"""Smoke benchmark numerical learners on Criteo; this is NOT an adsim evaluation.

Uses training-only numeric imputation and category-frequency encoding. Never uses
the target, source identity, day or split metadata as a predictor. Defaults to
validation only so the final test partition is not used for model selection.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from build_dataset import ROOT, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/external/criteo")
    ap.add_argument("--evaluate-test", action="store_true", help="Explicitly score the reserved test set")
    args = ap.parse_args()
    schema = json.loads((args.data / "schema.json").read_text())
    train = pd.read_parquet(args.data / "train.parquet", columns=["label"] + schema["allowed_predictors"])
    numeric, categorical = schema["numeric_features"], schema["categorical_features"]
    medians = train[numeric].median().fillna(0)
    maps = {c: train[c].value_counts(dropna=True) / len(train) for c in categorical}

    def features(frame):
        nums = frame[numeric].fillna(medians).to_numpy(dtype=float)
        # Some anonymised integer values are negative; signed log preserves sign.
        nums = np.sign(nums) * np.log1p(np.abs(nums))
        missing = frame[numeric + categorical].isna().to_numpy(dtype=float)
        cats = np.column_stack([frame[c].map(maps[c]).fillna(0).to_numpy(dtype=float) for c in categorical])
        return np.column_stack([nums, cats, missing])

    x, y = features(train), train.label.to_numpy()
    scaler = StandardScaler().fit(x)
    start = time.monotonic()
    ridge = Ridge(alpha=10).fit(scaler.transform(x), y)
    ridge_seconds = time.monotonic() - start
    start = time.monotonic()
    tree = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, min_samples_leaf=50,
                                        learning_rate=0.05, l2_regularization=1, early_stopping=False, random_state=0).fit(x, y)
    tree_seconds = time.monotonic() - start
    results = {}
    for split in ["validation"] + (["test"] if args.evaluate_test else []):
        frame = pd.read_parquet(args.data / f"{split}.parquet", columns=["label"] + schema["allowed_predictors"])
        xx, yy = features(frame), frame.label.to_numpy()
        predictions = {"training_rate_baseline": np.full(len(frame), y.mean()),
                       "ridge_component": ridge.predict(scaler.transform(xx)), "boosted_tree_component": tree.predict(xx)}
        results[split] = {}
        for name, p in predictions.items():
            # Regression outputs are bounded only for probability scoring; labels never change.
            clipped = np.clip(p, 1e-7, 1 - 1e-7)
            results[split][name] = {"log_loss": float(log_loss(yy, clipped)), "brier_score": float(brier_score_loss(yy, clipped)),
                                    "roc_auc": float(roc_auc_score(yy, clipped)), "mean_prediction": float(clipped.mean()),
                                    "observed_sample_rate": float(yy.mean()), "rows": len(frame),
                                    "predictions_outside_0_1": int(((p < 0) | (p > 1)).sum())}
    report = {"scope": "Numerical learner component smoke benchmark, not the trained adsim system or MiniLM",
              "training_rows": len(train), "test_evaluated": args.evaluate_test,
              "preprocessing": "Training-only median imputation, signed log of numeric inputs, category frequency encoding and missingness flags. Ridge standardisation fitted only on training.",
              "limitations": "Frequency encoding discards category identity; these are simple fixed baselines, not tuned models. Source sampling prevents interpreting sample rates as production CTR.",
              "fit_seconds": {"ridge": ridge_seconds, "boosted_tree": tree_seconds}, "results": results}
    write_json(args.data / ("benchmark_with_test.json" if args.evaluate_test else "benchmark_validation.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
