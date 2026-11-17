# Independent Criteo data audit

**PASS with scope limitations.** Independently checked all 480,000 output rows against retained original source records and verified every raw-file SHA256 against live Hugging Face object metadata.

- Pinned revision: `e11d69ae913b16cdd7387706a2133def0fdc6ced`.
- 408 checks passed; no failed integrity checks.
- 19,200,000 original label/feature values compared, including original types and missing values.
- 24 real source day partitions, 2015-02-15 through 2015-03-10.
- 320,000 training rows, 80,000 validation rows, 80,000 reserved test rows; no overlapping dates or source-row identities.
- 15,938 clicks and 464,062 nonclicks retained; no label balancing or zero removal.
- Shards and all sampled positions independently reconstructed from the complete upstream inventory and seed.

## Benchmark review

- Loads only the schema's feature allowlist; label, day and source identity are not passed as predictors.
- Numeric medians, category-frequency maps and Ridge scaling are fitted on training data only.
- Default invocation evaluates validation only; --evaluate-test is explicit. Integrity checks inspect test labels without fitting or choosing models.
- Retrains numerical Ridge/tree baselines on impression labels; does not claim to evaluate the fitted adsim system, MiniLM, CPM or advertiser transfer.
- Regression outputs are clipped for probability scoring and out-of-range counts are reported; frequency encoding loses category identity. Benchmark is a fixed smoke test, not a tuned classifier comparison.

## Limits

- Only 24 upstream day partitions exist; the requested 8–12 weeks cannot be supplied without a different source.
- Dates are supplied by the official mirror's directory partitions; original event timestamps and within-day chronology cannot be independently established.
- Selecting one shard and 20,000 rows per day is a bounded cluster sample, not a globally representative traffic sample.
- Upstream positives and negatives were subsampled at unequal undisclosed rates; sample click rates/calibration must not be interpreted as population CTR.
- Anonymous columns do not identify advertisers, audiences, creatives, spend or creative text. Do not fabricate these or use the sample to claim CPM/MiniLM/advertiser-generalization validation.
- Row identities and dates are disjoint across splits, but recurring users/ads cannot be audited because column semantics are undisclosed.
- CC-BY-NC-SA-4.0 attribution/noncommercial/share-alike source terms apply.

This audit is independent of `scripts/criteo/validate_dataset.py`. Its checker reads raw source batches rather than invoking the builder's extraction code. It examines reserved-test labels for integrity only; that does not authorize using them for model tuning.
