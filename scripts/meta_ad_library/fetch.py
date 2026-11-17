"""Download public Meta Ad Library records for a few advertiser pages through Apify.

Usage:  python scripts/meta_ad_library/fetch.py --limit 400 [--tag full]

The Apify token is read from the APIFY_TOKEN environment variable or from .env.apify in the
repository root (git-ignored). It is sent only to api.apify.com and never written to any output.
Raw responses are saved unmodified under data/external/meta_ad_library/raw/ (git-ignored).
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/external/meta_ad_library/raw"
ACTOR = "curious_coder~facebook-ads-library-scraper"
API = "https://api.apify.com/v2"
# page key -> (public Facebook page, Ad Library delivery country)
PAGES = {
    "sephora": ("https://www.facebook.com/sephora", "ALL"),
    "sephora_france": ("https://www.facebook.com/sephorafrance", "FR"),
    "sephora_italia": ("https://www.facebook.com/sephoraitalia", "IT"),
}


def token():
    if os.environ.get("APIFY_TOKEN"):
        return os.environ["APIFY_TOKEN"]
    for line in (ROOT / ".env.apify").read_text().splitlines():
        if line.startswith("APIFY_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("No APIFY_TOKEN in the environment or .env.apify")


def run(page, limit, sort_by, headers):
    url, country = PAGES[page]
    body = {"urls": [{"url": url}], "scrapeAdDetails": True, "count": limit, "limitPerSource": limit,
            "scrapePageAds.activeStatus": "all", "scrapePageAds.sortBy": sort_by, "scrapePageAds.countryCode": country}
    r = requests.post(f"{API}/acts/{ACTOR}/runs", headers=headers, json=body, params={"maxTotalChargeUsd": round(limit * 0.00075 * 1.3 + 0.01, 3)}, timeout=60)
    r.raise_for_status()
    info = r.json()["data"]
    while info["status"] in ("READY", "RUNNING"):
        time.sleep(5)
        info = requests.get(f"{API}/actor-runs/{info['id']}", headers=headers, timeout=60).json()["data"]
        print(f"  {page}: {info['status']}", flush=True)
    items = requests.get(f"{API}/datasets/{info['defaultDatasetId']}/items", headers=headers, params={"clean": "true", "format": "json"}, timeout=300)
    items.raise_for_status()
    return body, info, items.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, required=True, help="ads per page")
    ap.add_argument("--sort", default="most_recent", choices=["most_recent", "impressions_desc"])
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--pages", nargs="*", default=list(PAGES))
    a = ap.parse_args()
    headers = {"Authorization": "Bearer " + token()}
    RAW.mkdir(parents=True, exist_ok=True)
    log = []
    for page in a.pages:
        body, info, content = run(page, a.limit, a.sort, headers)
        path = RAW / f"{a.tag}_{page}.json"
        path.write_bytes(content)
        n = len(json.loads(content))
        log.append({"page": page, "page_url": PAGES[page][0], "delivery_country": PAGES[page][1], "actor": ACTOR.replace("~", "/"),
                    "actor_build": info.get("buildNumber"), "run_id": info["id"], "status": info["status"], "input": body,
                    "started_at": info.get("startedAt"), "finished_at": info.get("finishedAt"), "charged_usd": info.get("usageTotalUsd"),
                    "charged_events": info.get("chargedEventCounts"), "items": n, "file": path.name,
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
        print(f"{page}: {info['status']}, {n} ads, {len(content):,} bytes", flush=True)
    (RAW / f"{a.tag}_fetch_log.json").write_text(json.dumps({"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "runs": log}, indent=2))


if __name__ == "__main__":
    main()
