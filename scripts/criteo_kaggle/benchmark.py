"""Test actual numerical learner classes from current adsim on Kaggle click labels.

This adapts anonymous feature tables to the numerical components, NOT the full
campaign forecaster. Train-only preprocessing; validation selects the learner;
the final row-order holdout is scored after selection and never used to choose.

Which learners are engine paths here: Engine switches CTR to the Poisson likelihood as soon as
valid zero-click rows exist, and every unclicked impression is one. So the Poisson GLM and the
Poisson tree are the engine's CTR learners on this data and are the only selectable candidates.
Engine uses Ridge only on log CTR when every row has clicks, which per-impression labels cannot
provide; Ridge on the raw 0/1 label is kept as a labelled linear reference, not an engine path.
Engine-path rates get the same 100% ceiling Engine.predict applies (adsim.engine.MAX_CTR_PCT).
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from build_dataset import ROOT, hashes, json_write

sys.path.insert(0, str(ROOT))
from adsim.engine import MAX_CTR_PCT, _Hier, _PoissonTree  # noqa: E402

REFERENCE = "ridge_on_binary_label_reference"          # not an engine path on zero-inflated data; never selectable
ENGINE_PATHS = ["adsim_poisson_component", "adsim_poisson_tree_component"]


def finalise(name, raw):
    """What the engine would report: engine-path rates are capped at the engine's CTR ceiling."""
    return np.minimum(raw * 100, MAX_CTR_PCT) / 100 if name in ENGINE_PATHS else raw


def score(y, prediction, raw=None):
    raw = prediction if raw is None else raw
    if not (np.isfinite(prediction).all() and np.isfinite(raw).all()):
        raise ValueError("Nonfinite model output")
    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    return {"log_loss": float(log_loss(y, p)), "brier_score": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "mean_prediction": float(p.mean()), "sample_click_rate": float(y.mean()), "rows": len(y),
            "raw_outputs_outside_0_1": int(((raw < 0) | (raw > 1)).sum()),
            "outputs_outside_0_1": int(((prediction < 0) | (prediction > 1)).sum())}


def main():
    folder = ROOT / "data/external/criteo_kaggle"
    schema = json.loads((folder / "schema.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    predictors = schema["allowed_predictors"]
    if set(predictors) & set(schema["excluded_from_predictors"]):
        raise ValueError("Target/provenance in predictors")
    for split, record in manifest["outputs"].items():
        if hashes(folder / record["file"])["sha256"] != record["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {split}")
    train = pd.read_parquet(folder / "train.parquet", columns=["label"] + predictors)
    nums, cats = schema["numeric_features"], schema["categorical_features"]
    medians = train[nums].median().fillna(0)
    maps = {c: train[c].value_counts(dropna=True) / len(train) for c in cats}

    def transform(frame):
        numeric = frame[nums].fillna(medians).to_numpy(dtype=float)
        numeric = np.sign(numeric) * np.log1p(np.abs(numeric))
        categories = np.column_stack([frame[c].map(maps[c]).fillna(0).to_numpy(dtype=float) for c in cats])
        missing = frame[predictors].isna().to_numpy(dtype=float)
        values = np.column_stack([numeric, categories, missing])
        return pd.DataFrame(values, columns=predictors + [c + "_missing" for c in predictors])

    x, y = transform(train), train.label.to_numpy()
    blocks = [("std", list(x.columns), 1.0)]
    print(f"Training {len(train):,} source impressions, {x.shape[1]} transformed features", flush=True)
    models, fit_times, convergence = {}, {}, {}
    for name in [REFERENCE] + ENGINE_PATHS:
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if name == REFERENCE:
                model = _Hier(blocks).fit(x, y)
            elif name == "adsim_poisson_component":
                # One impression per source record; same regularisation rule as Engine's Poisson path.
                model = _Hier(blocks).fit(x, y, np.ones(len(y)), poisson_alpha=float(y.mean()) / len(y))
            else:
                model = _PoissonTree().fit(x, y, np.ones(len(y)))
        models[name], fit_times[name] = model, time.monotonic() - start
        convergence[name] = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        print(f"Fitted {name} in {fit_times[name]:.1f}s; convergence warnings={len(convergence[name])}", flush=True)

    def predict(model, name, frame):
        values = model.predict(frame)
        return values if name == REFERENCE else np.exp(values)

    val = pd.read_parquet(folder / "validation.parquet", columns=["label"] + predictors)
    xv, yv = transform(val), val.label.to_numpy()
    validation = {"training_rate_baseline": score(yv, np.full(len(yv), y.mean()))}
    for name, model in models.items():
        raw = predict(model, name, xv)
        validation[name] = score(yv, finalise(name, raw), raw)
    winner = min(ENGINE_PATHS, key=lambda name: validation[name]["log_loss"])
    # Save the selection before loading/scoring test features or labels.
    json_write(folder / "model_selection.json", {
        "criterion": "Lowest validation log loss among engine-path candidates; fixed candidates, no hyperparameter search",
        "selectable_candidates": ENGINE_PATHS, "reference_only": ["training_rate_baseline", REFERENCE],
        "selected_model": winner, "validation": validation, "test_used_for_selection": False})
    print(f"Validation selection: {winner}", flush=True)
    test = pd.read_parquet(folder / "test.parquet", columns=["label"] + predictors)
    xt, yt = transform(test), test.label.to_numpy()
    raw_test = {"training_rate_baseline": np.full(len(yt), y.mean())}
    raw_test.update({name: predict(model, name, xt) for name, model in models.items()})
    predictions = {name: finalise(name, raw) for name, raw in raw_test.items()}
    test_results = {name: score(yt, predictions[name], raw_test[name]) for name in predictions}
    # Paired, descriptive bootstrap of squared-error improvement; rows may be correlated.
    chosen = np.clip(predictions[winner], 1e-7, 1 - 1e-7)
    difference = (yt - predictions["training_rate_baseline"]) ** 2 - (yt - chosen) ** 2
    rng = np.random.default_rng(20261114)
    bootstrap = [float(difference[rng.integers(0, len(yt), len(yt))].mean()) for _ in range(500)]
    report = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "uncommitted_adsim_changes": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "adsim"], cwd=ROOT, text=True).strip()),
        "engine_sha256": hashes(ROOT / "adsim/engine.py")["sha256"],
        "scope": "Actual adsim._Hier Poisson and _PoissonTree learner components (the engine's CTR path when zero-click rows exist) retrained on Kaggle, with the engine's CTR ceiling; not the full Engine, MiniLM, CPM or campaign/audience model",
        "engine_path_candidates": ENGINE_PATHS,
        "reference_models": {"training_rate_baseline": "constant training click rate",
                             REFERENCE: "adsim._Hier Ridge fitted on the raw 0/1 label. Engine fits Ridge only on log CTR when every row has clicks, so this is a linear reference, not an engine path, and is excluded from selection"},
        "ctr_ceiling": {"max_ctr_pct": MAX_CTR_PCT, "applied_to": ENGINE_PATHS, "raw_outputs_outside_0_1": "count before the ceiling", "outputs_outside_0_1": "count in the scored output"},
        "training_rows": len(train), "training_clicks": int(y.sum()), "zero_labels_retained": int((y == 0).sum()),
        "feature_allowlist": predictors, "preprocessing": "Training-only numeric median imputation, signed log1p numeric values, category frequency encoding, explicit missingness flags. _Hier applies training-only standardisation internally.",
        "limitations": ["Anonymous category identity is discarded by frequency encoding", "Log-rate components are exponentiated and capped at the engine's 100% CTR ceiling; all outputs additionally clipped to [1e-7, 1-1e-7] only for probability scoring",
                       "No full campaign-engine test because actual campaign/audience/creative/spend fields are unavailable", "Systematic one-in-50 sample; source probabilities reflect upstream label-dependent sampling",
                       "Row-order holdout from labelled train.txt, not a Kaggle leaderboard submission", "Test scored after model selection; do not reuse it for tuning",
                       "Test was first scored at commit 06d6f6c before the CTR ceiling existed and rescored after adding it; candidates, hyperparameters and the selection rule were not changed using test results"],
        "fit_seconds": fit_times, "convergence_warnings": convergence,
        "selection": {"model": winner, "criterion": "validation log loss", "test_used_for_selection": False},
        "validation": validation, "test": test_results,
        "selected_model_brier_improvement_over_baseline": {"mean": float(difference.mean()), "bootstrap_95_percentile_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                                                          "caution": "Descriptive row bootstrap; dependence among impressions is not accounted for"}}
    json_write(folder / "benchmark_results.json", report)
    provenance = pd.read_parquet(folder / "test.parquet", columns=["source_row", "source_line_sha256"])
    provenance["label"] = yt
    for name, p in predictions.items():
        provenance[name] = np.clip(p, 1e-7, 1 - 1e-7)
    provenance.to_parquet(folder / "test_predictions.parquet", index=False)
    print(json.dumps({"selected_model": winner, "test": test_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
