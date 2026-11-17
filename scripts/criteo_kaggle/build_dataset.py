"""Download the historical Kaggle Criteo archive and build an ordered 1-in-50 sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
URL = "https://ndownloader.figshare.com/files/10082655"
SIZE = 4576820670
MD5 = "df9b1b3766d9ff91d5ca3eb3d23bed27"
EVERY, OFFSET = 50, 17
FEATURES = [f"integer_feature_{i}" for i in range(1, 14)] + [f"categorical_feature_{i}" for i in range(1, 27)]
SCHEMA = pa.schema([pa.field("label", pa.int32())] + [pa.field(c, pa.int32()) for c in FEATURES[:13]]
                   + [pa.field(c, pa.string()) for c in FEATURES[13:]]
                   + [pa.field("source_row", pa.int64()), pa.field("source_line_sha256", pa.string())])


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def download(dest, workers):
    if dest.exists():
        h = hashes(dest)
        if dest.stat().st_size == SIZE and h["md5"] == MD5:
            print("Using checksum-verified archive", flush=True)
            return h
        raise ValueError("Existing archive is not the expected source; refusing to overwrite")
    partial = dest.with_suffix(dest.suffix + ".part")
    with partial.open("wb") as f:
        f.truncate(SIZE)
    block = 64 * 1024 * 1024

    def part(start):
        end = min(start + block, SIZE) - 1
        for attempt in range(3):
            try:
                # Unique query avoids intermediaries serving a cached, different range.
                with requests.get(URL + f"?benchmark_range={start}", headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("Server did not honour exact requested byte range")
                    count = 0
                    with partial.open("r+b") as f:
                        f.seek(start)
                        for chunk in r.iter_content(1024 * 1024):
                            if count + len(chunk) > end - start + 1:
                                raise ValueError("Oversized byte range")
                            f.write(chunk)
                            count += len(chunk)
                if count != end - start + 1:
                    raise ValueError("Incomplete byte range")
                return count
            except (requests.RequestException, OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    received = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done in as_completed([pool.submit(part, start) for start in range(0, SIZE, block)]):
            received += done.result()
            print(f"Downloaded {received / 1e9:.2f}/{SIZE / 1e9:.2f} GB", flush=True)
    h = hashes(partial)
    if h["md5"] != MD5:
        raise ValueError("Archive MD5 does not match published Figshare checksum")
    partial.replace(dest)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/external/criteo_kaggle")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        ap.error("workers must be 1..8")
    folder, raw = args.out, args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    response = requests.get("https://api.figshare.com/v2/articles/5732310", timeout=45)
    response.raise_for_status()
    metadata = response.json()
    item = next(f for f in metadata["files"] if f["id"] == 10082655)
    if item["size"] != SIZE or item["computed_md5"] != MD5:
        raise ValueError("Mirror metadata changed")
    json_write(raw / "figshare_metadata.json", metadata)
    archive = raw / "dac.tar.gz"
    archive_hashes = download(archive, args.workers)
    buffers, batches, members = [[] for _ in SCHEMA.names], [], []
    total, selected, found, source_member = 0, 0, False, None
    # Never extract arbitrary archive paths. Read named members as streams only.
    with tarfile.open(archive, mode="r|gz") as tar, gzip.open(raw / "sampled_source_lines.tsv.gz", "wb") as retained:
        for member in tar:
            members.append({"name": member.name, "bytes": member.size})
            if member.name.lower().endswith("readme.txt") and member.isfile():
                (raw / "SOURCE_README.txt").write_bytes(tar.extractfile(member).read())
            if Path(member.name).name != "train.txt":
                continue
            if found:
                raise ValueError("Multiple training members")
            found, source_member = True, member.name
            print(f"Sampling {member.name}: {member.size / 1e9:.2f} GB uncompressed", flush=True)
            for index, line in enumerate(tar.extractfile(member)):
                total = index + 1
                if index % EVERY == OFFSET:
                    cells = line.rstrip(b"\r\n").split(b"\t")
                    if len(cells) != 40 or cells[0] not in (b"0", b"1"):
                        raise ValueError(f"Invalid labelled row {index}")
                    values = [int(cells[0])] + [int(v) if v else None for v in cells[1:14]]
                    values += [v.decode("ascii") if v else None for v in cells[14:]]
                    values += [index, hashlib.sha256(line).hexdigest()]
                    for buffer, value in zip(buffers, values):
                        buffer.append(value)
                    retained.write(line)
                    selected += 1
                    if selected % 25000 == 0:
                        batches.append(pa.Table.from_arrays([pa.array(v, type=f.type) for v, f in zip(buffers, SCHEMA)], schema=SCHEMA))
                        buffers = [[] for _ in SCHEMA.names]
                if total % 5000000 == 0:
                    print(f"Read {total:,} source rows; retained {selected:,}", flush=True)
            print(f"Finished labelled source: {total:,} rows", flush=True)
    if not found or total < 1000000:
        raise ValueError("Missing/truncated training source")
    if buffers[0]:
        batches.append(pa.Table.from_arrays([pa.array(v, type=f.type) for v, f in zip(buffers, SCHEMA)], schema=SCHEMA))
    sample = pa.concat_tables(batches)
    boundaries = [0, len(sample) * 70 // 100, len(sample) * 85 // 100, len(sample)]
    outputs = {}
    for i, split in enumerate(["train", "validation", "test"]):
        table = sample.slice(boundaries[i], boundaries[i + 1] - boundaries[i])
        path = folder / f"{split}.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=25000)
        outputs[split] = {"file": path.name, "rows": len(table), "clicks": int(pc.sum(table["label"]).as_py()),
                          "sample_click_rate": float(pc.mean(table["label"]).as_py()), "sha256": hashes(path)["sha256"],
                          "first_source_row": table["source_row"][0].as_py(), "last_source_row": table["source_row"][-1].as_py(),
                          "null_counts": {c: table[c].null_count for c in ["label"] + FEATURES}}
    json_write(folder / "schema.json", {"target": "label", "allowed_predictors": FEATURES,
                                       "numeric_features": FEATURES[:13], "categorical_features": FEATURES[13:],
                                       "excluded_from_predictors": ["label", "source_row", "source_line_sha256"]})
    json_write(folder / "manifest.json", {
        "name": "Criteo Kaggle Display Advertising Challenge labelled training sample", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_url": "https://www.kaggle.com/competitions/criteo-display-ad-challenge/data",
        "download_url": URL, "mirror_article": "https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310",
        "source_archive": {"file": "raw/dac.tar.gz", "bytes": SIZE, **archive_hashes}, "archive_members": members,
        "labelled_member": source_member, "labelled_source_rows": total, "sample_rows": len(sample),
        "sampling": {"method": "systematic original-row-order sample, independent of labels", "every": EVERY, "offset": OFFSET,
                     "rule": "zero_based_source_row % 50 == 17", "split": "first70%, next15%, last15% of sampled labelled source in original chronological row order"},
        "outputs": outputs, "limits": ["Seven-day labelled competition source, exact per-row dates unavailable",
        "Source order preserved, but boundaries are row positions, not known whole days", "Unlabelled official competition test is not used",
        "One-in-50 systematic sample is not a random uniform sample and may retain periodicity bias",
        "Upstream positive/negative sampling differs; sample probabilities are not population CTR", "No advertiser/ad/audience semantics, creative text or spend; no full adsim/MiniLM/CPM validation"],
        "licensing": "Figshare mirror claims CC0; that is not proof of rights to relicense Criteo/Kaggle data. Preserve source attribution and consult original competition/source terms before redistribution or commercial use."})
    print(f"Complete: {len(sample):,} labelled records", flush=True)


if __name__ == "__main__":
    main()
