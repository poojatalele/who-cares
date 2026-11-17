# Criteo external CTR benchmark

A reproducible **480,000-row sample across all 24 available day partitions**, for
checking numerical click-prediction learners on data independent of the Wevion
kit. This is not a drop-in input to `adsim`, a CPM dataset, or a creative-text test.

## Files and split

| File | Rows | Source partition dates | Purpose |
| --- | ---: | --- | --- |
| `train.parquet` | 320,000 | 2015-02-15 to 2015-03-02 | Fit models and preprocessing |
| `validation.parquet` | 80,000 | 2015-03-03 to 2015-03-06 | Compare candidate models |
| `test.parquet` | 80,000 | 2015-03-07 to 2015-03-10 | Reserved final evaluation |

There are **24 days, not 8–12 weeks**. No extra dates or observations were
invented. Dates come from the official Hugging Face mirror's directory names;
individual event timestamps are not available. Within-day order in these
Parquet shards is not asserted to be chronological.

The three sampled files total about 28 MB. `raw/` retains the 24 source shards
(about 1.13 GB), their download receipts, the pinned source inventory and source
README. Large data files are excluded from Git by this directory's `.gitignore`;
rebuild them using the script rather than expecting Git to transfer them.

## Provenance and licence

- Publisher: Criteo.
- Official description: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
- Dataset: https://huggingface.co/datasets/criteo/CriteoClickLogs
- Pinned revision: `e11d69ae913b16cdd7387706a2133def0fdc6ced`.
- Source card licence: **CC BY-NC-SA 4.0**. This sample is derived from that
  dataset and retains its attribution and noncommercial/share-alike terms:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
- Preparation: deterministic shard selection, row sampling and train/validation/
  test separation by date. No source feature values or click labels were changed.

`manifest.json` records every source URL, retained-file SHA256, source and sampled
row counts, click counts, null counts, per-shard random seed, output checksums and
split dates. `schema.json` defines the exact allowed predictor columns.

## Meaning of each record

Each row represents a source impression record:

- `label`: observed clicked (`1`) or not clicked (`0`).
- `integer_feature_1` to `integer_feature_13`: anonymous numeric features.
- `categorical_feature_1` to `categorical_feature_26`: anonymous categorical
  features, retained as strings with their original missing values.
- `source_day`, `source_file`, `source_row`: provenance, **not predictors**.
  `source_row` is the zero-based position in the retained source Parquet file.

The source does not disclose the feature meanings. None has been relabelled as
an advertiser, ad, audience, country, budget or creative. There is no spend,
CPM, copy, image or video field. Zero-click records are retained.

## Exact sampling procedure

1. Enumerate all 6,027 Parquet paths from the pinned repository revision.
2. Group by the 24 source-day directories.
3. In each group, select the path with minimum
   `SHA256("20261114|" + source_path)` (compared as digest bytes).
4. Derive a row seed from the first eight SHA256 digest bytes of
   `"20261114|" + source_path + "|rows"`, interpreted as an unsigned big-endian
   integer.
5. Use NumPy 2.3.5 `default_rng(seed)` (PCG64), `choice(source_rows, 20000,
   replace=False)`, and sort the resulting row positions.
6. Copy the original values and nulls at those positions, add provenance and
   concatenate by source day. Use the first 16 days for training, the next four
   for validation and the final four for testing.

Selection does not look at click labels. This is a **one-shard-per-day cluster
sample**, not uniform sampling over the full source corpus. Shard row counts
vary, while each day contributes equally. It must not be presented as a
representative estimate of all Criteo traffic.

Additionally, Criteo itself sampled clicked and unclicked records at different
rates. The necessary population sampling ratios are unavailable. Therefore the
observed click rate and probability calibration refer to **this sampled data**,
not real production CTR. No inverse-probability correction has been invented.

## Rebuild and verify

Run from the repository root, using a separate Python 3.12 environment:

```bash
python -m pip install -r scripts/criteo/requirements.txt
python scripts/criteo/build_dataset.py
python scripts/criteo/validate_dataset.py
```

The builder checks the pinned source revision and downloads three files at a
time by default. Extraction reads one column at a time to bound memory. Existing
completed downloads are reused only after validating their recorded checksums.

The validator checks every sampled value and null against the retained source,
reproduces seeded row selection, verifies checksums, checks both label classes
and validates disjoint date blocks and source-row identities. Its results are in
`validation_report.json`. The separately requested agent's findings are under
`audit/`; that report is independent of this validator.

## Run a numerical-component smoke benchmark

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/criteo/benchmark.py
```

This compares a training-click-rate baseline, Ridge regression and boosted-tree
regression on **validation only**. It writes `benchmark_validation.json`.
Preprocessing is fitted on training rows only: median imputation, signed log
numeric inputs, category-frequency encoding and missingness indicators. Ridge
also uses training-only standardisation. All 39 original feature columns are
eligible; neither labels nor provenance fields are predictors.

This is an intentionally simple, fixed baseline check. Frequency encoding loses
category identity, and these learners have not been tuned. Regression outputs
are bounded to valid probabilities for scoring; **labels are never clipped**.
Reported log loss, Brier score and ROC AUC measure performance on the sampled
click task. They are not daily CTR MAPE or CPM accuracy.

Only after finalising model selection, explicitly opt into the reserved test:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/criteo/benchmark.py --evaluate-test
```

That writes `benchmark_with_test.json`. Once scored and used to guide choices,
the test set must not continue to be described as untouched.

## What this can and cannot establish

**Can test:** numerical learner behaviour, zero-click retention, missing-data
handling, category novelty, throughput and performance across later source days.

**Cannot test:** the current trained `adsim` engine end to end, MiniLM creative
understanding, unseen-advertiser transfer, audience segmentation, CPM, budget
elasticity or creative winner selection. Those require a separate advertiser
dataset with the corresponding identifiers, creative assets and actual spend.

The original forecasting code, app and official hackathon predictions are not
modified by these scripts. No data has been uploaded to an external service.
