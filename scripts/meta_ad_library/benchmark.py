"""Test adsim on Sephora ads from the Meta Ad Library. Two parts that must not be confused.

PART A  REAL DATA ONLY. Uses only what Meta publishes. Two questions the real record can answer:
  A1  Does the creative's copy predict how long an ad ran and how many people it reached?
      (the engine's external-evidence path, adsim.creative.fit_external_prior, never validated before)
  A2  Does the creative say anything about WHICH audience an ad reaches, beyond its targeting?
      (the engine's creative representation, adsim.creative.CreativeSpace)

PART B  FULL ENGINE ON SIMULATED PERFORMANCE. Real creatives, launch dates and audiences; simulated
  impressions, clicks and spend (simulate.py). The engine is fitted on the training days and scored
  on the later test days against the simulated outcomes and against the planted truth, next to
  simple baselines. This shows whether the engine recovers a known structure. It is not evidence
  of accuracy on Sephora's actual results.

Usage:  python scripts/meta_ad_library/benchmark.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from adsim import Engine, load                                   # noqa: E402
from adsim.creative import CreativeSpace, embed, fit_external_prior  # noqa: E402
from adsim.data import flag_anomalies                             # noqa: E402
from adsim.evaluate import _ctr_block                             # noqa: E402
from adsim.simulate import audiences, simulate                    # noqa: E402

BASE = ROOT / "data/external/meta_ad_library"
REAL, SIM, OUT = BASE / "real", BASE / "simulated", BASE / "results"
CONFIG = ROOT / "configs/sephora_meta_ad_library.json"
TEXT = ["headline", "body", "link_description", "card_text"]
GENDERS, AGES = ["female", "male"], ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
HOME = {"sephora_france": "FR", "sephora_italia": "IT"}
REAL_TEST_SHARE = 0.30


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(spearmanr(a, b).statistic) if len(a) > 2 and np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")


# ------------------------------------------------------------------ PART A: real data only
def real_longevity(ads):
    """A1. Train on earlier finished ads, predict later finished ads, from copy alone."""
    done = ads[~ads["is_active"] & ads["page_key"].isin(HOME)].sort_values("start_date").reset_index(drop=True)
    cut = done["start_date"].iloc[int(len(done) * (1 - REAL_TEST_SHARE))]
    train, test = done[done["start_date"] < cut], done[done["start_date"] >= cut].copy()
    space = CreativeSpace({"creatives": {"text": TEXT}, "encoder": "auto"}).fit(train.drop_duplicates("creative_id"))
    texts = test[TEXT].fillna("").astype(str).agg(". ".join, axis=1).tolist()
    vecs = embed(texts, "hf")[0] if space.used.startswith("hf") else space.tfidf.transform(texts)
    out, rows = {"encoder": space.used, "train_ads": int(len(train)), "test_ads": int(len(test)), "test_starts_on": str(cut.date()),
                 "test_ads_with_copy_unseen_in_train": int((~test["creative_id"].isin(train["creative_id"])).sum()), "outcomes": {}}, test[["ad_id", "page_key", "creative_id", "start_date", "display_format"]].copy()
    unseen = ~test["creative_id"].isin(train["creative_id"]).values
    for outcome in ("days_running", "eu_total_reach"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.csv"
            train[TEXT + [outcome]].to_csv(path, index=False)
            model = fit_external_prior({"file": str(path), "text": TEXT, "outcome": outcome}, space)    # the engine's own code path
        y_train, y = np.log1p(train[outcome].values.astype(float)), np.log1p(test[outcome].values.astype(float))
        pred = model.predict(vecs) * (y_train.std() + 1e-9) + y_train.mean()
        by_format = train.assign(y=y_train).groupby(["page_key", "display_format"])["y"].mean()
        fmt = np.array([by_format.get((p, f), y_train.mean()) for p, f in zip(test["page_key"], test["display_format"])])

        def block(mask):
            sse = lambda q: float(((y[mask] - q[mask]) ** 2).sum())
            base = sse(np.full(len(y), y_train.mean()))
            return {"ads": int(mask.sum()), "spearman_copy_model": rho(pred[mask], y[mask]), "spearman_format_average": rho(fmt[mask], y[mask]),
                    "r2_copy_model_vs_train_mean": 1 - sse(pred) / base, "r2_format_average_vs_train_mean": 1 - sse(fmt) / base}
        out["outcomes"][outcome] = {"all_test_ads": block(np.ones(len(y), bool)), "copy_unseen_in_train": block(unseen)}
        rows["actual_" + outcome], rows["predicted_" + outcome], rows["format_average_" + outcome] = test[outcome].values, np.expm1(pred).round(1), np.expm1(fmt).round(1)
    out["caveats"] = ["Only finished ads are scored; recent ads that are still running are excluded, so later ads skew short",
                      "Longevity and reach are advertiser decisions (budget, schedule), not a measure of response"]
    return out, rows


def real_audience_share(ads, segments):
    """A2. Predict each later ad's real reach split across age x gender audiences."""
    seg = segments[segments["gender"].isin(GENDERS) & segments["age_range"].isin(AGES)].merge(ads[["ad_id", "page_key"]], on="ad_id")
    seg = seg[seg["country"] == seg["page_key"].map(HOME)]
    seg["audience"] = seg["gender"] + "|" + seg["age_range"]
    wide = seg.pivot_table(index="ad_id", columns="audience", values="reach", aggfunc="sum", fill_value=0)
    cols = [g + "|" + a for g in GENDERS for a in AGES]
    wide = wide.reindex(columns=cols, fill_value=0)
    wide = wide[wide.sum(axis=1) >= 1000]
    share = wide.div(wide.sum(axis=1), axis=0)
    a = ads.set_index("ad_id").loc[share.index].sort_values("start_date")
    share = share.loc[a.index]
    cut = a["start_date"].iloc[int(len(a) * (1 - REAL_TEST_SHARE))]
    tr, te = (a["start_date"] < cut).values, (a["start_date"] >= cut).values
    a["spec"] = a["page_key"] + "|" + a["target_gender"].astype(str) + "|" + a["target_age_min"].astype(str) + "-" + a["target_age_max"].astype(str)
    page_mean = share[tr].groupby(a.loc[tr, "page_key"].values).mean()
    spec_mean = share[tr].groupby(a.loc[tr, "spec"].values).mean()
    base_page = page_mean.loc[a["page_key"]].values
    base_spec = np.array([spec_mean.loc[s].values if s in spec_mean.index else page_mean.loc[p].values for s, p in zip(a["spec"], a["page_key"])])
    # creative features: the engine's own text projection (fitted on training creatives) plus format
    space = CreativeSpace({"creatives": {"text": TEXT}, "encoder": "auto"}).fit(a[tr].drop_duplicates("creative_id"))
    feats = space.transform(a.assign(creative_id=a.index)).values
    fmt = pd.get_dummies(a["display_format"]).reindex(columns=sorted(a.loc[tr, "display_format"].unique()), fill_value=0).values.astype(float)
    X = np.column_stack([feats, fmt, pd.get_dummies(a["page_key"]).values.astype(float)])
    eps = 0.005
    resid = np.log(share.values + eps) - np.log(base_spec + eps)
    model = RidgeCV(alphas=[1, 10, 100, 1000]).fit(X[tr], resid[tr])
    raw = (base_spec + eps) * np.exp(model.predict(X)) - eps
    pred = np.clip(raw, 0, None) * (base_spec > 0)                       # never predict reach outside what the targeting has ever reached
    pred = pred / np.where(pred.sum(axis=1, keepdims=True) > 0, pred.sum(axis=1, keepdims=True), 1)
    y = share.values

    def score(q, mask):
        tv = 0.5 * np.abs(y[mask] - q[mask]).sum(axis=1)
        return {"ads": int(mask.sum()), "mean_share_misallocated": float(tv.mean()), "median_share_misallocated": float(np.median(tv)),
                "largest_audience_named_correctly": float((y[mask].argmax(1) == q[mask].argmax(1)).mean())}
    unseen = te & ~a["creative_id"].isin(a.loc[tr, "creative_id"]).values
    out = {"encoder": space.used, "train_ads": int(tr.sum()), "test_ads": int(te.sum()), "test_starts_on": str(cut.date()), "audiences": cols,
           "metric": "share misallocated = half the summed absolute difference between predicted and real reach shares (0 = perfect, 1 = entirely wrong)",
           "all_test_ads": {"page_average": score(base_page, te), "same_targeting_average": score(base_spec, te), "same_targeting_plus_creative": score(pred, te)},
           "copy_unseen_in_train": {"page_average": score(base_page, unseen), "same_targeting_average": score(base_spec, unseen), "same_targeting_plus_creative": score(pred, unseen)},
           "test_ads_whose_targeting_was_never_seen_in_train": int((te & ~a["spec"].isin(a.loc[tr, "spec"]).values).sum()),
           "caveat": "Reach split is mostly set by the advertiser's targeting; what is left is Meta's delivery, which favours likely responders but is not a response measure"}
    rows = a.loc[te, ["page_key", "creative_id", "start_date", "display_format", "spec"]].copy()
    for j, c in enumerate(cols):
        rows["real_" + c], rows["pred_" + c], rows["targeting_avg_" + c] = y[te][:, j].round(4), pred[te][:, j].round(4), base_spec[te][:, j].round(4)
    return out, rows.reset_index()


# ------------------------------------------------------------------ PART B: engine on simulated performance
def baselines(train, test):
    """Simple forecasts any analyst could make from the training days."""
    def pooled(frame, key):
        g = frame.groupby(key)[["clicks", "impressions", "spend"]].sum()
        return g["clicks"] / g["impressions"] * 100, g["spend"] / g["impressions"] * 1000
    adv_ctr, adv_cpm = pooled(train, "advertiser_id")
    aud_ctr, aud_cpm = pooled(train, "audience_id")
    last = train[train["date"] > train["date"].max() - pd.Timedelta(days=7)]
    l_aud_ctr, l_aud_cpm = pooled(last, "audience_id")
    l_ad_ctr, l_ad_cpm = pooled(last, "ad_id")
    l_ad_ctr = l_ad_ctr[l_ad_ctr > 0]                                   # an ad with no clicks last week falls back to its audience
    out = {"advertiser_average": (test["advertiser_id"].map(adv_ctr), test["advertiser_id"].map(adv_cpm)),
           "audience_average": (test["audience_id"].map(aud_ctr), test["audience_id"].map(aud_cpm))}
    out["last_7_days"] = (test["ad_id"].map(l_ad_ctr).fillna(test["audience_id"].map(l_aud_ctr)).fillna(out["audience_average"][0]),
                          test["ad_id"].map(l_ad_cpm).fillna(test["audience_id"].map(l_aud_cpm)).fillna(out["audience_average"][1]))
    return {k: (c.values.astype(float), m.values.astype(float)) for k, (c, m) in out.items()}


def accuracy(test, ctr, cpm, mask=None):
    mask = np.ones(len(test), bool) if mask is None else mask
    t = test[mask]
    b = _ctr_block(t, pd.DataFrame({"ctr": ctr[mask]}))
    w = t["impressions"].values
    return {"rows": b["rows"], "zero_click_rows": b["zero_click_rows_scored"],
            "ctr_click_error_share": b["ctr_wape_clicks"], "ctr_total_clicks_bias": b["ctr_pooled_bias"], "ctr_mae_pp": b["ctr_mae_pp"],
            "ctr_log_error_vs_planted_truth": float(np.mean(np.abs(np.log(ctr[mask] / t["true_ctr_pct"].values)))),
            "ctr_log_error_vs_planted_truth_impression_weighted": float(np.average(np.abs(np.log(ctr[mask] / t["true_ctr_pct"].values)), weights=w)),
            "cpm_log_error_vs_observed": float(np.mean(np.abs(np.log(cpm[mask] / t["cpm"].values)))),
            "cpm_log_error_vs_planted_truth": float(np.mean(np.abs(np.log(cpm[mask] / t["true_cpm_eur"].values)))),
            "cpm_bias_vs_observed": float(np.mean(np.log(cpm[mask] / t["cpm"].values)))}


def audience_ranking(test, preds):
    """Per advertiser: does the forecast order the audiences the way the test period did?"""
    out, tables = {}, []
    assert (test.index.values == np.arange(len(test))).all()
    for adv, t in test.groupby("advertiser_id"):
        g = t.groupby("audience_id").apply(lambda x: pd.Series({
            "observed_ctr": x["clicks"].sum() / x["impressions"].sum() * 100,
            "true_ctr": np.average(x["true_ctr_pct"], weights=x["impressions"]),
            **{name: np.average(preds[name][0][x.index.values], weights=x["impressions"]) for name in preds}}), include_groups=False)   # test has a 0..n-1 index
        tables.append(g.round(4).reset_index().assign(advertiser_id=adv))
        res = {"audiences": int(len(g)), "observed_best_audience": g["observed_ctr"].idxmax()}
        for name in preds:
            top3 = len(set(g[name].nlargest(3).index) & set(g["observed_ctr"].nlargest(3).index))
            res[name] = {"rank_correlation_with_observed": rho(g[name], g["observed_ctr"]), "rank_correlation_with_planted_truth": rho(g[name], g["true_ctr"]),
                         "best_audience_named": g[name].idxmax(), "best_audience_correct": bool(g[name].idxmax() == g["observed_ctr"].idxmax()), "top3_overlap": top3}
        out[adv] = res
    return out, pd.concat(tables)


def new_creative_audiences(test, preds, train_creatives):
    """For creatives that did not exist in training: which audience suits each one best?"""
    t = test[~test["creative_id"].isin(train_creatives)]
    pos = test.index.get_indexer(t.index)
    frame = t[["creative_id", "advertiser_id", "audience_id", "impressions", "true_ctr_pct"]].copy()
    for name in preds:
        frame[name] = preds[name][0][pos]
    wavg = lambda x, c: np.average(x[c], weights=x["impressions"])
    g = frame.groupby(["creative_id", "advertiser_id", "audience_id"]).apply(lambda x: pd.Series({"true": wavg(x, "true_ctr_pct"), **{n: wavg(x, n) for n in preds}, "impressions": x["impressions"].sum()}), include_groups=False).reset_index()
    rows, picks = [], []
    for (cid, adv), x in g.groupby(["creative_id", "advertiser_id"]):
        if len(x) < 3:
            continue
        x = x.set_index("audience_id")
        best = x["true"].idxmax()
        r = {"creative_id": cid, "advertiser_id": adv, "audiences": len(x), "true_best_audience": best, "chance_of_guessing": 1 / len(x)}
        for n in preds:
            pick = x[n].idxmax()
            r.update({n + "_pick": pick, n + "_correct": pick == best, n + "_in_top2": best in x[n].nlargest(2).index,
                      n + "_ctr_kept": x.loc[pick, "true"] / x.loc[best, "true"], n + "_rank_corr": rho(x[n], x["true"])})
        rows.append(r)
    d = pd.DataFrame(rows)
    summary = {"new_creatives_scored": int(len(d)), "mean_audiences_per_creative": float(d["audiences"].mean()), "chance_level": float(d["chance_of_guessing"].mean())}
    for n in preds:
        summary[n] = {"best_audience_correct": float(d[n + "_correct"].mean()), "true_best_in_top2": float(d[n + "_in_top2"].mean()),
                      "share_of_best_ctr_kept": float(d[n + "_ctr_kept"].mean()), "mean_rank_correlation": float(d[n + "_rank_corr"].mean())}
    return summary, d


def engine_on_simulated():
    t0 = time.monotonic()
    b = load(CONFIG)
    h, creatives, cfg = b["history"], b["creatives"], b["cfg"]
    truth = pd.read_csv(SIM / "truth.csv.gz", parse_dates=["date"])
    h = h.merge(truth[["date", "ad_id", "true_ctr_pct", "true_cpm_eur"]], on=["date", "ad_id"], how="left", validate="one_to_one")
    train = h[h["split"] == "train"].drop(columns=["true_ctr_pct", "true_cpm_eur"])        # the engine never sees the truth columns
    e = Engine(cfg).calibrate(train, creatives).fit(train, creatives)
    flags = flag_anomalies(h.drop(columns=["quality"]))                                  # same rule as adsim.evaluate.backtest
    test = flags[(flags["split"] == "test") & (flags["quality"] == "")].reset_index(drop=True)
    excluded = flags.loc[(flags["split"] == "test") & (flags["quality"] != ""), "quality"].value_counts().to_dict()
    p = e.predict(test)
    preds = {"adsim_engine": (p["ctr"].values, p["cpm"].values), **baselines(train, test)}
    new_ad, new_cr = p["new_ad"].values, p["new_creative"].values
    acc = {name: {"all_test_rows": accuracy(test, c, m), "ads_seen_in_training": accuracy(test, c, m, ~new_ad),
                  "new_ads_known_creative": accuracy(test, c, m, new_ad & ~new_cr), "new_creatives": accuracy(test, c, m, new_cr)} for name, (c, m) in preds.items()}
    per_week = {}
    week = ((test["date"] - test["date"].min()).dt.days // 7 + 1).values
    for wk in sorted(set(week)):
        per_week[f"test_week_{wk}"] = {name: {k: v for k, v in accuracy(test, c, m, week == wk).items() if k in ("rows", "ctr_click_error_share", "ctr_log_error_vs_planted_truth", "cpm_log_error_vs_observed", "cpm_bias_vs_observed")}
                                       for name, (c, m) in preds.items()}
    s = simulate(e, test.assign(budget=test["spend"].values * test.groupby(["date", "audience_id"])["ad_id"].transform("size").values), draws=200)
    obs = (test["clicks"] / test["impressions"] * 100).values
    cov = lambda lo, hi, y, mask: float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask]))) if mask.any() else float("nan")
    coverage = {"target": 0.80, "interval_status": e.interval_status,
                "ctr_simulated_range_seen_ads": cov(s["ctr_observed_p10"].values, s["ctr_observed_p90"].values, obs, ~new_ad),
                "ctr_simulated_range_new_ads": cov(s["ctr_observed_p10"].values, s["ctr_observed_p90"].values, obs, new_ad),
                "cpm_band_seen_ads": cov(p["cpm_lo"].values, p["cpm_hi"].values, test["cpm"].values, ~new_ad),
                "cpm_band_new_ads": cov(p["cpm_lo"].values, p["cpm_hi"].values, test["cpm"].values, new_ad)}
    ranking, ranking_table = audience_ranking(test, preds)
    picks_summary, picks = new_creative_audiences(test, preds, set(train["creative_id"]))
    spec = json.loads((SIM / "simulation_spec.json").read_text())
    arch = audiences(e)
    arch["planted_ctr_effect"] = arch["audience_id"].map(spec["drawn_effects"]["ctr_audience"])
    arch["planted_cpm_effect"] = arch["audience_id"].map(spec["drawn_effects"]["cpm_audience"])
    archetypes = {adv: {"rank_correlation_engine_ctr_level_vs_planted": rho(g["ctr_level_pct"], g["planted_ctr_effect"]),
                        "rank_correlation_engine_cpm_level_vs_planted": rho(g["cpm_level_eur"], g["planted_cpm_effect"]),
                        "audiences_per_archetype": g["archetype"].value_counts().sort_index().to_dict()} for adv, g in arch.groupby("advertiser_id")}

    OUT.mkdir(parents=True, exist_ok=True)
    audit = test[["date", "advertiser_id", "audience_id", "ad_id", "library_ad_id", "creative_id", "age_days", "impressions", "clicks", "spend"]].copy()
    audit["observed_ctr_pct"], audit["observed_cpm_eur"] = obs.round(4), test["cpm"].round(4).values
    audit["planted_true_ctr_pct"], audit["planted_true_cpm_eur"] = test["true_ctr_pct"].round(4).values, test["true_cpm_eur"].round(4).values
    for col in ("ctr", "ctr_lo", "ctr_hi", "cpm", "cpm_lo", "cpm_hi"):
        audit["engine_" + col] = p[col].round(4).values
    audit["sim_ctr_p10"], audit["sim_ctr_p90"] = s["ctr_observed_p10"].round(4).values, s["ctr_observed_p90"].round(4).values
    for name in ("advertiser_average", "audience_average", "last_7_days"):
        audit[name + "_ctr"], audit[name + "_cpm"] = preds[name][0].round(4), preds[name][1].round(4)
    audit["new_ad"], audit["new_creative"] = new_ad, new_cr
    audit["date"] = audit["date"].dt.strftime("%Y-%m-%d")
    audit.to_csv(OUT / "test_predictions.csv.gz", index=False)
    # a readable sample for manual review: same number of rows from every advertiser x known/new group
    cr = creatives.set_index("creative_id")
    grp = audit["advertiser_id"] + np.where(new_cr, "|new creative", np.where(new_ad, "|new ad", "|seen ad"))
    sample = audit.groupby(grp, group_keys=False).apply(lambda x: x.sample(min(len(x), 100), random_state=0)).sort_values(["advertiser_id", "date", "ad_id"])
    sample.insert(6, "display_format", cr["display_format"].reindex(sample["creative_id"]).values)
    sample.insert(7, "headline", cr["headline"].reindex(sample["creative_id"]).values)
    sample.insert(8, "body", cr["body"].reindex(sample["creative_id"]).fillna("").str.slice(0, 200).values)
    sample.to_csv(OUT / "test_review_sample.csv", index=False)
    picks.to_csv(OUT / "new_creative_audience_picks.csv", index=False)
    ranking_table.to_csv(OUT / "audience_ranking.csv", index=False)
    arch.to_csv(OUT / "audience_archetypes.csv", index=False)
    return {"what_is_real": "creatives (copy, format), launch and end dates, age x gender audiences, reach per audience",
            "what_is_simulated": "daily impressions, clicks, spend (see simulated/simulation_spec.json)",
            "engine": {"ctr_model": e.ctr_mode, "encoder": e.space.used, "creative_protocol": e.creative_protocol, "text_projection_fitted_on_creatives": e.space.n_fit_creatives,
                       "trained_through": str(e.trained_through.date()), "budget_elasticity": e.k, "tree_inputs_left_to_linear_model": e.tree_dropped,
                       "missing_optional_inputs": b["missing_optional_inputs"], "train_rows_excluded": e.excluded, "calibration": getattr(e, "calibration", None)},
            "split": {"train_rows": int(len(train)), "test_rows_scored": int(len(test)), "test_rows_excluded_by_quality_rule": excluded,
                      "test_days": [str(test["date"].min().date()), str(test["date"].max().date())],
                      "test_rows_seen_ads": int((~new_ad).sum()), "test_rows_new_ads_known_creative": int((new_ad & ~new_cr).sum()), "test_rows_new_creatives": int(new_cr.sum())},
            "accuracy": acc, "accuracy_by_test_week": per_week, "interval_coverage": coverage,
            "audience_ranking": ranking, "best_audience_for_new_creatives": picks_summary, "archetypes_vs_planted": archetypes,
            "outputs_above_100pct_ctr": int((p["ctr"] > 100).sum()), "seconds": round(time.monotonic() - t0, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ads = pd.read_csv(REAL / "ads.csv", parse_dates=["start_date", "end_date"], dtype={"ad_id": str})
    segments = pd.read_csv(REAL / "reach_by_segment.csv", dtype={"ad_id": str})
    print("A1 longevity and reach from copy (real)...", flush=True)
    a1, a1_rows = real_longevity(ads)
    a1_rows.to_csv(OUT / "real_longevity_predictions.csv", index=False)
    print("A2 audience reach split from creative (real)...", flush=True)
    a2, a2_rows = real_audience_share(ads, segments)
    a2_rows.to_csv(OUT / "real_audience_share_predictions.csv", index=False)
    print("B  full engine on simulated performance...", flush=True)
    part_b = engine_on_simulated()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "uncommitted_adsim_changes": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "adsim"], cwd=ROOT, text=True).strip()),
        "engine_sha256": sha(ROOT / "adsim/engine.py"),
        "source": "Meta Ad Library public records for the Sephora France and Sephora Italy pages (see real/real_manifest.json)",
        "read_this_first": ["Part A uses real published numbers only. Part B uses simulated clicks, impressions and spend.",
                            "The Ad Library publishes no clicks, spend, CTR or CPM for commercial ads, so no result here measures accuracy on Sephora's actual performance.",
                            "The US Sephora page has no audience or reach data (EU-only disclosure) and is not part of any test."],
        "part_a_real_only": {"A1_longevity_and_reach_from_copy": a1, "A2_audience_reach_split_from_creative": a2},
        "part_b_engine_on_simulated_performance": part_b,
        "inputs": {"real/ads.csv": sha(REAL / "ads.csv"), "real/reach_by_segment.csv": sha(REAL / "reach_by_segment.csv"), "real/creatives.csv": sha(REAL / "creatives.csv"),
                   "simulated/daily.csv.gz": sha(SIM / "daily.csv.gz"), "simulated/truth.csv.gz": sha(SIM / "truth.csv.gz")},
    }
    (OUT / "benchmark_results.json").write_text(json.dumps(report, indent=2, default=float))
    print("done ->", OUT / "benchmark_results.json", flush=True)


if __name__ == "__main__":
    main()
