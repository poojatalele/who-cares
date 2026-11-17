"""Validate every sampled value against retained, checksum-verified source shards."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_dataset import FEATURES, ROOT, REVISION, SEED, sha256, source_selection, write_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/criteo")
    args = parser.parse_args()
    folder = args.data
    manifest = json.loads((folder / "manifest.json").read_text())
    schema = json.loads((folder / "schema.json").read_text())
    source_metadata = json.loads((folder / "raw/source_metadata.json").read_text())
    require(manifest["revision"] == source_metadata["sha"] == REVISION, "Source revision mismatch")
    selection = {day: path for day, path, _ in source_selection(source_metadata)}
    require(schema["allowed_predictors"] == FEATURES, "Predictor allowlist mismatch")
    require(not set(FEATURES) & set(schema["excluded_from_predictors"]), "Target/provenance leakage")
    days = manifest["days"]
    require(len(days) == len(set(days)) == 24 and days == sorted(days), "Invalid day partitions")
    require((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days == 23, "Days not consecutive")
    expected_days = {"train": days[:16], "validation": days[16:20], "test": days[20:]}
    sources = {x["source_day"]: x for x in manifest["sources"]}
    source_checks, split_checks = [], {}
    for split, expected in expected_days.items():
        output = manifest["outputs"][split]
        path = folder / output["file"]
        require(sha256(path) == output["sha256"], f"Output checksum mismatch: {split}")
        table = pq.ParquetFile(path).read()
        require(table.column_names == ["label"] + FEATURES + ["source_day", "source_file", "source_row"], "Unexpected columns")
        require(len(table) == output["rows"] == len(expected) * manifest["rows_per_day"], "Row count mismatch")
        require(sorted(set(table["source_day"].to_pylist())) == expected == output["days"], "Date split mismatch")
        require(not table["label"].null_count, "Null labels")
        require(set(table["label"].to_pylist()) == {0, 1}, "Both click classes must be present")
        for day in expected:
            subset = table.filter(pc.equal(table["source_day"], day))
            record = sources[day]
            raw_path = folder / record["raw_local_file"]
            require(record["source_file"] == selection[day], "Shard selection changed")
            require(set(subset["source_file"].to_pylist()) == {record["source_file"]}, "Source path mismatch")
            require(sha256(raw_path) == record["sha256"], f"Raw checksum mismatch: {day}")
            source = pq.ParquetFile(raw_path)
            require(source.metadata.num_rows == record["source_rows"], "Source count mismatch")
            positions = subset["source_row"].to_numpy()
            require(len(positions) == len(set(positions)) == record["sample_rows"], "Duplicate source positions")
            seed = int.from_bytes(hashlib.sha256(f"{SEED}|{record['source_file']}|rows".encode()).digest()[:8], "big")
            expected_positions = np.sort(np.random.default_rng(seed).choice(source.metadata.num_rows, len(subset), replace=False))
            require(np.array_equal(positions, expected_positions), "Row selection mismatch")
            for column in ["label"] + FEATURES:
                original = pc.take(source.read(columns=[column]).column(column), pa.array(positions))
                require(subset[column].equals(original), f"Original values/nulls changed: {day} {column}")
                require(subset[column].null_count == record["null_counts"][column], "Null count mismatch")
            require(pc.sum(subset["label"]).as_py() == record["clicks"], "Click count mismatch")
            source_checks.append({"day": day, "rows": len(subset), "all_40_source_columns_match": True,
                                  "raw_sha256_matches": True, "sampling_reproduced": True})
            print(f"Verified {day}: every sampled value and null matches source", flush=True)
        split_checks[split] = {"rows": len(table), "days": expected, "clicks": int(pc.sum(table["label"]).as_py()),
                               "sample_click_rate": float(pc.mean(table["label"]).as_py()),
                               "sha256_matches": True}
    report = {"status": "PASS", "total_rows": sum(x["rows"] for x in split_checks.values()),
              "source_revision": REVISION, "split_checks": split_checks, "source_checks": source_checks,
              "no_source_row_overlap": True, "labels_and_nulls_preserved": True,
              "note": "Source row identities are unique. Equal feature vectors may legitimately recur across days; that is not proof of duplicate source events.",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__}}
    write_json(folder / "validation_report.json", report)
    print("PASS: full source-to-sample validation", flush=True)


if __name__ == "__main__":
    main()
