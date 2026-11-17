"""Turn raw Meta Ad Library records into flat tables. Everything written here is REAL: copied or
counted from the public record, nothing estimated or simulated.

Usage:  python scripts/meta_ad_library/build_dataset.py full_eu full_us

Outputs under data/external/meta_ad_library/real/:
  ads.csv               one row per library ad: page, dates, format, targeting, EU reach, creative_id
  creatives.csv         one row per distinct creative (same copy and format on the same page)
  reach_by_segment.csv  one row per ad x country x age band x gender with the published reach
  real_manifest.json    counts, date range, source files and checksums

What the Ad Library does NOT publish for commercial ads, and is therefore absent here:
impressions, clicks, spend, CTR, CPM, budget. See simulate.py for how those are handled.
"""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/external/meta_ad_library"
RAW, REAL = BASE / "raw", BASE / "real"
GENDERS = ("female", "male", "unknown")


def clean(text):
    """Collapse whitespace and drop zero-width characters; wording is left untouched."""
    return re.sub(r"\s+", " ", re.sub(r"[​‌‍﻿]", "", text or "")).strip()


def day(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d") if epoch else None


def parse(record, page_key):
    snap, eu = record.get("snapshot") or {}, record.get("aaa_info") or {}
    cards = snap.get("cards") or []
    card_text = clean(" | ".join(dict.fromkeys(clean(" ".join(filter(None, [c.get("title"), c.get("body")]))) for c in cards)))
    body = clean((snap.get("body") or {}).get("text"))
    if not body and cards:                                  # some formats keep the copy on the cards
        body = clean(cards[0].get("body"))
    headline = clean(snap.get("title")) or (clean(cards[0].get("title")) if cards else "")
    age = eu.get("age_audience") or {}
    locations = [l for l in (eu.get("location_audience") or []) if not l.get("excluded")]
    ad = {
        "ad_id": str(record["ad_archive_id"]), "page_key": page_key, "page_id": str(record.get("page_id")), "page_name": record.get("page_name"),
        "start_date": day(record.get("start_date")), "end_date": day(record.get("end_date")), "is_active": bool(record.get("is_active")),
        "display_format": snap.get("display_format") or "UNKNOWN", "cta_type": snap.get("cta_type") or "NONE",
        "platforms": "|".join(sorted(record.get("publisher_platform") or [])),
        "collation_id": record.get("collation_id"), "collation_count": record.get("collation_count"),
        "headline": headline, "body": body, "link_description": clean(snap.get("link_description")), "card_text": card_text[:600],
        "n_cards": len(cards), "landing_domain": clean(snap.get("caption")),
        "has_eu_transparency": eu.get("eu_total_reach") is not None,
        "target_locations": "|".join(sorted(str(l.get("name")) for l in locations)), "target_location_types": "|".join(sorted({str(l.get("type")) for l in locations})),
        "target_gender": eu.get("gender_audience"), "target_age_min": age.get("min"), "target_age_max": age.get("max"),
        "eu_total_reach": eu.get("eu_total_reach"),
    }
    segments = []
    for country in eu.get("age_country_gender_reach_breakdown") or []:
        for band in country.get("age_gender_breakdowns") or []:
            for g in GENDERS:
                if band.get(g):
                    segments.append({"ad_id": ad["ad_id"], "country": country.get("country"), "age_range": band.get("age_range"), "gender": g, "reach": int(band[g])})
    return ad, segments


def main(tags):
    REAL.mkdir(parents=True, exist_ok=True)
    ads, segments, sources = [], [], []
    for tag in tags:
        log = json.loads((RAW / f"{tag}_fetch_log.json").read_text())
        for run in log["runs"]:
            content = (RAW / run["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != run["sha256"]:
                raise ValueError(f"Raw file changed since download: {run['file']}")
            records = json.loads(content)
            for r in records:
                if "ad_archive_id" not in r:
                    continue
                a, s = parse(r, run["page"])
                a["fetched_at_utc"] = log["fetched_at_utc"]
                ads.append(a); segments.extend(s)
            sources.append({k: run[k] for k in ("page", "page_url", "delivery_country", "actor", "actor_build", "run_id", "items", "file", "sha256")} | {"fetched_at_utc": log["fetched_at_utc"]})
    ads = pd.DataFrame(ads)
    duplicates = int(ads.duplicated("ad_id").sum())
    ads = ads.drop_duplicates("ad_id").reset_index(drop=True)
    segments = pd.DataFrame(segments).drop_duplicates(["ad_id", "country", "age_range", "gender"])
    ads["days_running"] = (pd.to_datetime(ads["end_date"]) - pd.to_datetime(ads["start_date"])).dt.days + 1
    # A creative is the same copy in the same format on the same page; one creative can run in several ad sets.
    key = ads[["page_key", "display_format", "cta_type", "headline", "body", "link_description", "card_text"]].astype(str).agg("\x1f".join, axis=1)
    ads["creative_id"] = ["CR_" + hashlib.sha1(k.encode()).hexdigest()[:12] for k in key]
    creatives = ads.groupby("creative_id").agg(page_key=("page_key", "first"), display_format=("display_format", "first"), cta_type=("cta_type", "first"),
                                               headline=("headline", "first"), body=("body", "first"), link_description=("link_description", "first"),
                                               card_text=("card_text", "first"), n_cards=("n_cards", "first"), n_ads=("ad_id", "size"),
                                               first_start_date=("start_date", "min")).reset_index()
    summed = segments.groupby("ad_id")["reach"].sum()
    ads["segment_reach_sum"] = ads["ad_id"].map(summed)
    ads.to_csv(REAL / "ads.csv", index=False)
    creatives.to_csv(REAL / "creatives.csv", index=False)
    segments.to_csv(REAL / "reach_by_segment.csv", index=False)
    eu = ads[ads["has_eu_transparency"]]
    manifest = {
        "what_this_is": "Public Meta Ad Library records, flattened. No value in real/ is estimated or simulated.",
        "not_published_by_the_ad_library": ["impressions", "clicks", "spend", "CTR", "CPM", "budget"],
        "sources": sources, "duplicate_ads_dropped": duplicates,
        "pages": {p: {"ads": int(len(g)), "creatives": int(g["creative_id"].nunique()), "first_start_date": g["start_date"].min(), "last_start_date": g["start_date"].max(),
                      "active_at_fetch": int(g["is_active"].sum()), "ads_with_eu_reach": int(g["has_eu_transparency"].sum()),
                      "eu_total_reach_sum": int(g["eu_total_reach"].fillna(0).sum()), "formats": g["display_format"].value_counts().to_dict(),
                      "targeting_specs": int(g.loc[g["has_eu_transparency"], ["target_locations", "target_gender", "target_age_min", "target_age_max"]].drop_duplicates().shape[0])}
                  for p, g in ads.groupby("page_key")},
        "caveats": ["Reach is lifetime-to-date and is censored for ads still active at fetch time",
                    "end_date of an active ad is the last day Meta reported, not a planned end",
                    "Records are the most recent ads per page up to the fetch limit, not a random sample of all ads",
                    "Meta reports small reach values with privacy rounding; segment reach may not sum exactly to eu_total_reach",
                    "Ads without EU delivery (the US page) carry no targeting or reach"],
        "reach_check": {"eu_ads": int(len(eu)), "ads_where_segment_sum_within_5pct_of_total": int(((eu["segment_reach_sum"] / eu["eu_total_reach"]).between(0.95, 1.05)).sum())},
        "outputs": {f: {"rows": int(len(d)), "sha256": hashlib.sha256((REAL / f).read_bytes()).hexdigest()}
                    for f, d in [("ads.csv", ads), ("creatives.csv", creatives), ("reach_by_segment.csv", segments)]},
    }
    (REAL / "real_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("pages", "reach_check", "duplicate_ads_dropped")}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or ["full_eu", "full_us"])
