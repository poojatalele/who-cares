"""Add the performance numbers the Ad Library does not publish, so the full engine can be run.

REAL (copied from real/): which ads ran, their copy and format, launch and end dates, the age x
gender audiences they reached and how many people in each.
SIMULATED (generated here from SPEC with a fixed seed): daily impressions, clicks and spend.

The recipe is fixed in SPEC before the engine is ever run on the result and is written, with every
drawn effect, to simulated/simulation_spec.json. Because the hidden "truth" is known, a forecast
can be scored against it. It measures whether the engine recovers a planted structure on real
creatives and real audiences. It does NOT measure accuracy on Sephora's actual results, which
are not public.

Usage:  python scripts/meta_ad_library/simulate.py
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/external/meta_ad_library"
REAL, SIM = BASE / "real", BASE / "simulated"

SPEC = {
    "seed": 20261114,
    "window": {"start": "2026-09-01", "end": "2026-11-13", "why": "last full day before the download; earlier months have too few French ads"},
    "test_share_of_days": 0.30,                       # deliberately larger than the usual 15%: more rows to review
    "audience": "country x gender x age band as published by Meta (female/male, six age bands)",
    "min_cell_reach": 1000,                            # ad x audience cells below this are too small to carry daily rows
    "impressions": {"frequency": 1.5, "day_noise_sd": 0.35,
                    "rule": "real lifetime reach of the ad in the audience x frequency, spread evenly over the ad's real live days, x log-normal day noise"},
    "ctr": {"base_pct": 1.0, "audience_sd": 0.30, "format_sd": 0.20, "promo_lift": 0.20, "creative_sd": 0.20, "ad_audience_sd": 0.10,
            "delivery_skew_gamma": 0.15, "fatigue_per_week": -0.05, "weekday_sd": 0.05, "day_noise_sd": 0.15, "max_rate": 0.5,
            "rule": "log CTR = log base + audience + format + promo + creative + ad-in-audience + gamma x real delivery skew + fatigue x weeks live + weekday + day noise; clicks ~ Binomial(impressions, CTR)"},
    "cpm": {"base_eur": {"sephora_france": 7.0, "sephora_italia": 5.0}, "audience_sd": 0.20, "format_sd": 0.10, "trend_per_day": 0.002, "day_noise_sd": 0.08,
            "rule": "log CPM = log base + audience + format + trend x day index + day noise; spend = impressions x CPM / 1000"},
    "promo_pattern": r"\d{1,2}\s?%|black\s*friday|promo|scont[oi]|offert[ae]|offres?\b|réduction|remise|saldi|soldes|\bsale\b|\bcod(?:e|ice)\b",
    "delivery_skew": "log of the ad's real reach share in the audience over the average share among same-page ads with the same targeting; standardised, clipped to +/-2",
    "assumptions_not_from_data": ["frequency", "every CTR and CPM parameter above", "even spread of delivery over live days",
                                  "that delivery skew signals response (gamma)", "the CPM trend standing in for Q4 price pressure"],
}
GENDERS, AGES = ["female", "male"], ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def main():
    SIM.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SPEC["seed"])
    ads = pd.read_csv(REAL / "ads.csv", parse_dates=["start_date", "end_date"], dtype={"ad_id": str})
    ads = ads[ads["has_eu_transparency"] & ads["page_key"].isin(SPEC["cpm"]["base_eur"])]
    seg = pd.read_csv(REAL / "reach_by_segment.csv", dtype={"ad_id": str})
    seg = seg[seg["gender"].isin(GENDERS) & seg["age_range"].isin(AGES)].merge(
        ads[["ad_id", "page_key", "creative_id", "display_format", "start_date", "end_date", "days_running", "target_gender", "target_age_min", "target_age_max"]], on="ad_id")
    home = {"sephora_france": "FR", "sephora_italia": "IT"}
    seg = seg[seg["country"] == seg["page_key"].map(home)].copy()
    seg["audience_id"] = seg["country"] + "|" + seg["gender"] + "|" + seg["age_range"]

    # real delivery skew: where an ad reached more of an audience than same-targeting ads usually do
    seg["share"] = seg["reach"] / seg.groupby("ad_id")["reach"].transform("sum")
    spec_key = ["page_key", "target_gender", "target_age_min", "target_age_max"]
    n_spec = seg.groupby(spec_key)["ad_id"].transform("nunique")
    expected = seg.groupby(spec_key + ["audience_id"])["share"].transform("sum") / n_spec
    raw = np.log((seg["share"] + 0.01) / (expected + 0.01))
    seg["delivery_skew"] = ((raw - raw.mean()) / raw.std()).clip(-2, 2)

    cells = seg[seg["reach"] >= SPEC["min_cell_reach"]].sort_values(["ad_id", "audience_id"]).reset_index(drop=True)
    cells["sim_ad_id"] = cells["ad_id"] + "|" + cells["gender"].str[0].str.upper() + cells["age_range"]

    creatives = pd.read_csv(REAL / "creatives.csv").set_index("creative_id")
    text = creatives[["headline", "body", "link_description", "card_text"]].fillna("").astype(str).agg(" ".join, axis=1)
    promo = text.str.contains(SPEC["promo_pattern"], case=False, regex=True)

    c, p = SPEC["ctr"], SPEC["cpm"]

    def draw(keys, sd):
        keys = sorted(set(keys))
        return dict(zip(keys, rng.normal(0, sd, len(keys))))
    eff = {"ctr_audience": draw(cells["audience_id"], c["audience_sd"]), "ctr_format": draw(cells["display_format"], c["format_sd"]),
           "ctr_creative": draw(cells["creative_id"], c["creative_sd"]), "ctr_ad_audience": draw(cells["sim_ad_id"], c["ad_audience_sd"]),
           "ctr_weekday": dict(zip(range(7), rng.normal(0, c["weekday_sd"], 7))),
           "cpm_audience": draw(cells["audience_id"], p["audience_sd"]), "cpm_format": draw(cells["display_format"], p["format_sd"])}

    lo, hi = pd.Timestamp(SPEC["window"]["start"]), pd.Timestamp(SPEC["window"]["end"])
    days = np.array([(e - s).days + 1 for s, e in zip(cells["start_date"], cells["end_date"])])
    rep = cells.loc[cells.index.repeat(days)].copy()
    rep["date"] = rep["start_date"] + pd.to_timedelta(np.concatenate([np.arange(n) for n in days]), unit="D")
    rep["age_days"] = (rep["date"] - rep["start_date"]).dt.days
    rep = rep[(rep["date"] >= lo) & (rep["date"] <= hi)].sort_values(["date", "sim_ad_id"]).reset_index(drop=True)
    n = len(rep)

    per_day = rep["reach"].values * SPEC["impressions"]["frequency"] / rep["days_running"].values
    impressions = np.maximum(np.round(per_day * np.exp(rng.normal(0, SPEC["impressions"]["day_noise_sd"], n))), 1).astype(int)

    parts = pd.DataFrame({
        "ctr_audience": rep["audience_id"].map(eff["ctr_audience"]).values, "ctr_format": rep["display_format"].map(eff["ctr_format"]).values,
        "ctr_promo": rep["creative_id"].map(promo).astype(float).values * c["promo_lift"], "ctr_creative": rep["creative_id"].map(eff["ctr_creative"]).values,
        "ctr_ad_audience": rep["sim_ad_id"].map(eff["ctr_ad_audience"]).values, "ctr_delivery_skew": c["delivery_skew_gamma"] * rep["delivery_skew"].values,
        "ctr_fatigue": c["fatigue_per_week"] * rep["age_days"].values / 7.0, "ctr_weekday": rep["date"].dt.weekday.map(eff["ctr_weekday"]).values,
        "cpm_audience": rep["audience_id"].map(eff["cpm_audience"]).values, "cpm_format": rep["display_format"].map(eff["cpm_format"]).values,
        "cpm_trend": p["trend_per_day"] * (rep["date"] - lo).dt.days.values})
    ctr_cols, cpm_cols = [k for k in parts if k.startswith("ctr_")], [k for k in parts if k.startswith("cpm_")]
    # expected values (what a perfect forecaster could know) exclude the unpredictable day noise
    expected_ctr = np.minimum(c["base_pct"] / 100 * np.exp(parts[ctr_cols].sum(axis=1).values), c["max_rate"])
    expected_cpm = rep["page_key"].map(p["base_eur"]).values * np.exp(parts[cpm_cols].sum(axis=1).values)
    day_ctr = np.minimum(expected_ctr * np.exp(rng.normal(0, c["day_noise_sd"], n)), c["max_rate"])
    day_cpm = expected_cpm * np.exp(rng.normal(0, p["day_noise_sd"], n))
    clicks = rng.binomial(impressions, day_ctr)
    spend = np.round(impressions * day_cpm / 1000, 4)

    all_days = pd.date_range(lo, hi)
    n_test = int(round(len(all_days) * SPEC["test_share_of_days"]))
    test_start = all_days[-n_test]
    split = np.where(rep["date"] >= test_start, "test", "train")

    daily = pd.DataFrame({"date": rep["date"].dt.strftime("%Y-%m-%d"), "ad_id": rep["sim_ad_id"].values, "library_ad_id": rep["ad_id"].values,
                          "audience_id": rep["audience_id"].values, "creative_id": rep["creative_id"].values, "advertiser_id": rep["page_key"].values,
                          "impressions": impressions, "clicks": clicks, "spend": spend,
                          "reach": np.maximum(np.round(impressions / SPEC["impressions"]["frequency"]), 1).astype(int), "split": split})
    truth = pd.concat([daily[["date", "ad_id", "split"]], pd.DataFrame({"true_ctr_pct": expected_ctr * 100, "true_cpm_eur": expected_cpm}), parts.round(5)], axis=1)
    launch = cells[["sim_ad_id", "ad_id", "audience_id", "creative_id", "start_date", "reach", "delivery_skew"]].rename(
        columns={"sim_ad_id": "ad_id", "ad_id": "library_ad_id", "start_date": "launch_date", "reach": "real_lifetime_reach"})
    launch["launch_date"] = launch["launch_date"].dt.strftime("%Y-%m-%d")

    daily.to_csv(SIM / "daily.csv.gz", index=False)
    truth.to_csv(SIM / "truth.csv.gz", index=False)
    launch.to_csv(SIM / "ad_launch.csv", index=False)
    train_creatives = set(daily.loc[daily["split"] == "train", "creative_id"])
    train_ads = set(daily.loc[daily["split"] == "train", "ad_id"])
    test = daily[daily["split"] == "test"]
    report = dict(SPEC, drawn_effects={k: {str(a): round(float(b), 5) for a, b in v.items()} for k, v in eff.items() if k not in ("ctr_creative", "ctr_ad_audience")},
                  creatives_with_promo_wording=int(promo.reindex(sorted(set(cells["creative_id"]))).sum()),
                  split={"train_days": [str(lo.date()), str((test_start - pd.Timedelta(days=1)).date())], "test_days": [str(test_start.date()), str(hi.date())],
                         "train_rows": int((split == "train").sum()), "test_rows": int((split == "test").sum()),
                         "test_rows_share": float((split == "test").mean()),
                         "test_ads_never_seen_in_train": int(test.loc[~test["ad_id"].isin(train_ads), "ad_id"].nunique()),
                         "test_creatives_never_seen_in_train": int(test.loc[~test["creative_id"].isin(train_creatives), "creative_id"].nunique())},
                  totals={"rows": n, "library_ads": int(daily["library_ad_id"].nunique()), "ad_x_audience": int(daily["ad_id"].nunique()),
                          "creatives": int(daily["creative_id"].nunique()), "audiences": int(daily["audience_id"].nunique()),
                          "impressions": int(impressions.sum()), "clicks": int(clicks.sum()), "spend_eur": float(spend.sum()),
                          "zero_click_rows": int((clicks == 0).sum()), "overall_ctr_pct": float(clicks.sum() / impressions.sum() * 100)},
                  real_inputs={f: hashlib.sha256((REAL / f).read_bytes()).hexdigest() for f in ("ads.csv", "creatives.csv", "reach_by_segment.csv")},
                  outputs={f: hashlib.sha256((SIM / f).read_bytes()).hexdigest() for f in ("daily.csv.gz", "truth.csv.gz", "ad_launch.csv")})
    (SIM / "simulation_spec.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("split", "totals", "creatives_with_promo_wording")}, indent=1))


if __name__ == "__main__":
    main()
