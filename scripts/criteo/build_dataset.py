"""Build a provenance-preserving, bounded Criteo component benchmark.

Downloads one seeded shard per day, then samples rows without looking at labels.
No fabricated ad IDs, spend, creative descriptions, or extra dates are introduced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

REPO = "criteo/CriteoClickLogs"
REVISION = "e11d69ae913b16cdd7387706a2133def0fdc6ced"
SEED = 20261114
FEATURES = [f"integer_feature_{i}" for i in range(1, 14)] + [f"categorical_feature_{i}" for i in range(1, 27)]
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(url, dest):
    """Resume only completed downloads with an independently stored checksum."""
    receipt = dest.with_suffix(dest.suffix + ".json")
    if dest.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record["url"] == url and record["sha256"] == sha256(dest):
            return record
        raise ValueError(f"Existing download does not match its receipt: {dest}")
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f"Expected complete file, got {response.status_code}")
                h, size = hashlib.sha256(), 0
                with partial.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected and size != int(expected):
                    raise ValueError("Incomplete download")
            pq.read_metadata(partial)  # Reject invalid/truncated Parquet before publishing.
            partial.replace(dest)
            record = {"url": url, "bytes": size, "sha256": h.hexdigest()}
            write_json(receipt, record)
            return record
        except (requests.RequestException, ValueError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def source_selection(metadata):
    groups = {}
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if name.endswith(".parquet"):
            day = name.split("/")[1].removeprefix("day=")
            groups.setdefault(day, []).append(name)
    if len(groups) != 24:
        raise ValueError(f"Expected 24 source partitions, found {len(groups)}")
    return [(day, min(paths, key=lambda p: hashlib.sha256(f"{SEED}|{p}".encode()).digest()), len(paths))
            for day, paths in sorted(groups.items())]


def sample_file(path, source_path, day, n):
    source = pq.ParquetFile(path)
    if source.schema_arrow.names != ["label"] + FEATURES:
        raise ValueError(f"Unexpected source schema: {source.schema_arrow}")
    total = source.metadata.num_rows
    if total < n:
        raise ValueError(f"Source shard has only {total} rows; requested {n}")
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{source_path}|rows".encode()).digest()[:8], "big")
    indices = np.sort(np.random.default_rng(seed).choice(total, n, replace=False))
    # Read one column at a time to avoid materialising a full ~1.5M-row shard.
    columns = [pc.take(source.read(columns=[name]).column(name), pa.array(indices))
               for name in ["label"] + FEATURES]
    table = pa.Table.from_arrays(columns, names=["label"] + FEATURES)
    if table["label"].null_count or not set(table["label"].to_pylist()).issubset({0, 1}):
        raise ValueError("Missing or nonbinary source labels")
    table = table.append_column("source_day", pa.array([day] * n))
    table = table.append_column("source_file", pa.array([source_path] * n))
    table = table.append_column("source_row", pa.array(indices, type=pa.int64()))
    record = {"source_day": day, "source_file": source_path, "source_rows": total,
              "source_row_groups": source.metadata.num_row_groups, "row_sampling_seed": seed,
              "sample_rows": n, "clicks": int(pc.sum(table["label"]).as_py()),
              "sample_click_rate": float(pc.mean(table["label"]).as_py()),
              "null_counts": {c: table[c].null_count for c in ["label"] + FEATURES}}
    return table, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/external/criteo")
    ap.add_argument("--rows-per-day", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.rows_per_day <= 0 or not 1 <= args.workers <= 4:
        ap.error("rows-per-day must be positive; workers must be between 1 and 4")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw"
    raw.mkdir(exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}", timeout=45)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION:
        raise ValueError("Source revision mismatch")
    write_json(raw / "source_metadata.json", metadata)
    card = requests.get(f"https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md", timeout=45)
    card.raise_for_status()
    (raw / "SOURCE_README.md").write_text(card.text)
    selected = source_selection(metadata)
    # Download concurrently; serialise extraction to bound working memory.
    extraction_lock = threading.Lock()

    def work(item):
        day, name, shard_count = item
        path = raw / f"{day}.parquet"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
        print(f"Downloading {day}", flush=True)
        receipt = download(url, path)
        with extraction_lock:
            table, record = sample_file(path, name, day, args.rows_per_day)
        record.update(receipt)
        record["available_shards_on_day"] = shard_count
        record["raw_local_file"] = str(path.relative_to(args.out))
        print(f"Sampled {day}: {len(table):,} rows, {record['clicks']:,} clicks", flush=True)
        return day, table, record

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, x) for x in selected]
        for future in as_completed(futures):
            day, table, record = future.result()
            results[day] = (table, record)
    days = sorted(results)
    split_days = {"train": days[:16], "validation": days[16:20], "test": days[20:]}
    outputs = {}
    for split, chosen in split_days.items():
        table = pa.concat_tables([results[day][0] for day in chosen])
        path = args.out / f"{split}.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=args.rows_per_day)
        outputs[split] = {"file": path.name, "rows": len(table), "days": chosen,
                          "sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(args.out / "schema.json", {
        "target": "label", "target_meaning": "Observed clicked=1 / not-clicked=0 in the source's subsampled traffic",
        "allowed_predictors": FEATURES, "numeric_features": FEATURES[:13], "categorical_features": FEATURES[13:],
        "excluded_from_predictors": ["label", "source_day", "source_file", "source_row"],
        "null_policy": "Preserve original nulls; fit any imputation on training data only",
        "granularity": "one source impression record", "schema_version": 1})
    write_json(args.out / "manifest.json", {
        "dataset": REPO, "revision": REVISION, "seed": SEED, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        "license": metadata.get("cardData", {}).get("license"), "rows_per_day": args.rows_per_day,
        "total_rows": sum(x["rows"] for x in outputs.values()), "days": days, "outputs": outputs,
        "sampling": "One shard per day selected by minimum SHA256(seed|path); uniform seeded row sampling without replacement within that shard, independent of label. Rows sorted by source position; no claim about intraday chronology of the Parquet mirror.",
        "limits": ["24 days, not 8-12 weeks", "One-shard-per-day cluster sample, not a representative sample of all traffic",
                   "Original source subsampled positives and negatives differently; sample click rate is not population CTR",
                   "Anonymous feature semantics; no advertiser/ad/audience mapping", "No creative text or spend; MiniLM and CPM cannot be validated",
                   "Partition dates are supplied by the official Hugging Face mirror, not independently verified event timestamps",
                   "No rows were invented, labels balanced, zero outcomes removed or values imputed"],
        "sources": [results[day][1] for day in days]})
    print(f"Complete: {sum(x['rows'] for x in outputs.values()):,} rows at {args.out}", flush=True)


if __name__ == "__main__":
    main()
