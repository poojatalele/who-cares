"""Independent reviewer: does not import the builder or its validator."""
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

BASE = Path(__file__).resolve().parents[1]
REV = "e11d69ae913b16cdd7387706a2133def0fdc6ced"
REPO = "criteo/CriteoClickLogs"
EXPECTED_FEATURES = [f"integer_feature_{i}" for i in range(1, 14)] + [f"categorical_feature_{i}" for i in range(1, 27)]
SOURCE_COLUMNS = ["label"] + EXPECTED_FEATURES
checks, failures, day_reports, output_reports = [], [], [], []


def check(condition, description):
    checks.append(description)
    if not bool(condition):
        failures.append(description)
        raise AssertionError(description)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = json.loads((BASE / "manifest.json").read_text())
schema = json.loads((BASE / "schema.json").read_text())
session = requests.Session()
resp = session.get(f"https://huggingface.co/api/datasets/{REPO}/revision/{REV}", timeout=30)
resp.raise_for_status()
upstream = resp.json()
check(upstream["sha"] == REV == manifest["revision"], "Pinned source revision matches independent live API")
check(manifest["license"] == upstream["cardData"]["license"] == "cc-by-nc-sa-4.0", "Source license recorded accurately")
groups = defaultdict(list)
for entry in upstream["siblings"]:
    name = entry["rfilename"]
    if name.endswith(".parquet"):
        groups[name.split("/")[1][4:]].append(name)
days = sorted(groups)
expected_days = [(date(2015, 2, 15) + timedelta(days=i)).isoformat() for i in range(24)]
check(days == expected_days == manifest["days"], "Exactly the upstream 24 consecutive day partitions, without invented dates")
check(schema["allowed_predictors"] == EXPECTED_FEATURES, "Predictor allowlist contains exactly the 39 original anonymous features")
check(set(schema["excluded_from_predictors"]) == {"label", "source_day", "source_file", "source_row"}, "Target and provenance are explicitly excluded from predictors")
expected_splits = {"train": days[:16], "validation": days[16:20], "test": days[20:]}
sources = {s["source_day"]: s for s in manifest["sources"]}
all_keys = set()
total_clicks = 0

for split, chosen_days in expected_splits.items():
    path = BASE / f"{split}.parquet"
    pf = pq.ParquetFile(path)
    record = manifest["outputs"][split]
    check(digest(path) == record["sha256"], f"{split}: independently recomputed output SHA256")
    check(path.stat().st_size == record["bytes"], f"{split}: output byte count")
    check(pf.metadata.num_rows == len(chosen_days) * 20000 == record["rows"], f"{split}: correct row count")
    check(record["days"] == chosen_days, f"{split}: correct temporal day allocation")
    check(pf.schema_arrow.names == SOURCE_COLUMNS + ["source_day", "source_file", "source_row"], f"{split}: no invented model fields")
    check(pf.metadata.num_row_groups == len(chosen_days), f"{split}: one row group per source day")
    split_clicks = 0

    for day_index, day in enumerate(chosen_days):
        out = pf.read_row_group(day_index)
        src = sources[day]
        # Derive selection from independently fetched complete inventory, not from the receipt.
        expected_file = sorted(groups[day], key=lambda x: hashlib.sha256((str(20261114) + "|" + x).encode()).hexdigest())[0]
        check(src["source_file"] == expected_file, f"{day}: deterministic shard selection matches live complete inventory")
        check(src["available_shards_on_day"] == len(groups[day]), f"{day}: recorded shard inventory count")
        check(set(out["source_day"].to_pylist()) == {day}, f"{day}: source_day matches output block")
        check(set(out["source_file"].to_pylist()) == {expected_file}, f"{day}: source_file matches output block")
        raw = BASE / src["raw_local_file"]
        raw_hash = digest(raw)
        head = session.head(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{expected_file}", allow_redirects=False, timeout=30)
        head.raise_for_status()
        check(head.headers.get("X-Repo-Commit") == REV, f"{day}: live object revision")
        check(raw_hash == head.headers.get("X-Linked-ETag", "").strip('"') == src["sha256"], f"{day}: retained source SHA256 matches live upstream object SHA256")
        check(raw.stat().st_size == int(head.headers["X-Linked-Size"]) == src["bytes"], f"{day}: retained source size matches live upstream")
        source = pq.ParquetFile(raw)
        check(source.schema_arrow.names == SOURCE_COLUMNS, f"{day}: source column identity")
        check(source.metadata.num_rows == src["source_rows"], f"{day}: source row count")
        row_seed = int(hashlib.sha256(f"20261114|{expected_file}|rows".encode()).hexdigest()[:16], 16)
        check(row_seed == src["row_sampling_seed"], f"{day}: row seed independently derived")
        positions = np.sort(np.random.Generator(np.random.PCG64(row_seed)).choice(source.metadata.num_rows, size=20000, replace=False))
        actual_positions = out["source_row"].to_numpy()
        check(np.array_equal(positions, actual_positions), f"{day}: all sampled row positions match seeded sampling without replacement")
        keys = {(expected_file, int(x)) for x in actual_positions}
        check(len(keys) == 20000 and not all_keys.intersection(keys), f"{day}: row identities unique within and across all splits")
        all_keys.update(keys)
        chunks, offset = [], 0
        # Independent streaming extraction, different from builder's full-column reads.
        for batch in source.iter_batches(batch_size=65536, columns=SOURCE_COLUMNS, use_threads=False):
            lo = np.searchsorted(positions, offset, side="left")
            hi = np.searchsorted(positions, offset + len(batch), side="left")
            if hi > lo:
                chunks.append(pa.Table.from_batches([batch]).take(pa.array(positions[lo:hi] - offset)))
            offset += len(batch)
        expected = pa.concat_tables(chunks)
        check(expected.equals(out.select(SOURCE_COLUMNS), check_metadata=False), f"{day}: all 20,000 labels and all 39 feature values/types/nulls match original source exactly")
        labels = out["label"].to_numpy()
        check(set(labels.tolist()) == {0, 1} and not out["label"].null_count, f"{day}: both outcomes retained and labels binary/nonmissing")
        nulls = {name: out[name].null_count for name in SOURCE_COLUMNS}
        check(nulls == src["null_counts"], f"{day}: recorded missing-value counts match output")
        clicks = int(labels.sum())
        check(clicks == src["clicks"], f"{day}: recorded click count")
        split_clicks += clicks
        total_clicks += clicks
        day_reports.append({"day": day, "split": split, "rows": len(out), "clicks": clicks, "nonclicks": len(out) - clicks, "source_sha256": raw_hash, "source_values_compared": len(out) * len(SOURCE_COLUMNS), "source_sha256_verified_against_live_upstream": True})
        print(f"Independently verified {day}: 20,000 rows x 40 original columns", flush=True)
        del chunks, expected, out, source
        gc.collect()
    output_reports.append({"split": split, "rows": record["rows"], "days": chosen_days, "clicks": split_clicks, "sample_click_rate": split_clicks / record["rows"], "sha256": record["sha256"]})

check(len(all_keys) == manifest["total_rows"] == 480000, "Exactly 480,000 distinct original source records")
report = {
    "status": "PASS_WITH_SCOPE_LIMITATIONS",
    "auditor": "Independent child agent /root/criteo_data_audit; standalone checker does not import builder or primary validator",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_revision": REV,
    "source_file_inventory_count": sum(map(len, groups.values())),
    "assertions_passed": len(checks),
    "failures": failures,
    "rows_checked": len(all_keys),
    "source_cell_values_checked": len(all_keys) * 40,
    "clicks": total_clicks,
    "nonclicks": len(all_keys) - total_clicks,
    "splits": output_reports,
    "days": day_reports,
    "benchmark_code_review": {
        "predictors": "Loads only the schema's feature allowlist; label, day and source identity are not passed as predictors.",
        "preprocessing": "Numeric medians, category-frequency maps and Ridge scaling are fitted on training data only.",
        "test_policy": "Default invocation evaluates validation only; --evaluate-test is explicit. Integrity checks inspect test labels without fitting or choosing models.",
        "scope": "Retrains numerical Ridge/tree baselines on impression labels; does not claim to evaluate the fitted adsim system, MiniLM, CPM or advertiser transfer.",
        "caveat": "Regression outputs are clipped for probability scoring and out-of-range counts are reported; frequency encoding loses category identity. Benchmark is a fixed smoke test, not a tuned classifier comparison."
    },
    "limitations": [
        "Only 24 upstream day partitions exist; the requested 8–12 weeks cannot be supplied without a different source.",
        "Dates are supplied by the official mirror's directory partitions; original event timestamps and within-day chronology cannot be independently established.",
        "Selecting one shard and 20,000 rows per day is a bounded cluster sample, not a globally representative traffic sample.",
        "Upstream positives and negatives were subsampled at unequal undisclosed rates; sample click rates/calibration must not be interpreted as population CTR.",
        "Anonymous columns do not identify advertisers, audiences, creatives, spend or creative text. Do not fabricate these or use the sample to claim CPM/MiniLM/advertiser-generalization validation.",
        "Row identities and dates are disjoint across splits, but recurring users/ads cannot be audited because column semantics are undisclosed.",
        "CC-BY-NC-SA-4.0 attribution/noncommercial/share-alike source terms apply."
    ]
}
(BASE / "audit/report.json").write_text(json.dumps(report, indent=2) + "\n")
lines = ["# Independent Criteo data audit", "", "**PASS with scope limitations.** Independently checked all 480,000 output rows against retained original source records and verified every raw-file SHA256 against live Hugging Face object metadata.", "", f"- Pinned revision: `{REV}`.", f"- {len(checks)} checks passed; no failed integrity checks.", "- 19,200,000 original label/feature values compared, including original types and missing values.", "- 24 real source day partitions, 2015-02-15 through 2015-03-10.", "- 320,000 training rows, 80,000 validation rows, 80,000 reserved test rows; no overlapping dates or source-row identities.", f"- {total_clicks:,} clicks and {len(all_keys)-total_clicks:,} nonclicks retained; no label balancing or zero removal.", "- Shards and all sampled positions independently reconstructed from the complete upstream inventory and seed.", "", "## Benchmark review", "", *[f"- {v}" for v in report["benchmark_code_review"].values()], "", "## Limits", "", *[f"- {v}" for v in report["limitations"]], "", "This audit is independent of `scripts/criteo/validate_dataset.py`. Its checker reads raw source batches rather than invoking the builder's extraction code. It examines reserved-test labels for integrity only; that does not authorize using them for model tuning.", ""]
(BASE / "audit/report.md").write_text("\n".join(lines))
print(json.dumps({"status": report["status"], "assertions": len(checks), "rows": len(all_keys), "clicks": total_clicks}, indent=2), flush=True)
