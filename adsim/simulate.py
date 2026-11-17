"""What-if and Monte Carlo on top of a fitted Engine. Structural, so every curve can be inspected."""
import numpy as np
import pandas as pd

from .engine import MAX_CTR_PCT


def what_if_budget(engine, rows, multipliers=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)):
    """Response of one audience-day (or any set of rows) to scaling its budget.

    Spend is assumed to follow budget and to be split equally across the rows of an audience-day.
    Returns per multiplier: CPM, CTR, impressions, clicks, CPC and clicks gained per extra euro.
    Rows outside the budget range seen in training are marked and carry a limitation note.
    """
    out = []
    n = rows.groupby(["date", "audience_id"])["ad_id"].transform("size")
    for m in multipliers:
        p = engine.predict(rows, budget_multiplier=m)
        spend = rows["budget"].values * m / n.values
        impr = spend / p["cpm"].values * 1000
        clicks = impr * p["ctr"].values / 100
        out.append(dict(multiplier=m, budget=float(spend.sum()), cpm=float(np.average(p["cpm"], weights=impr)),
                        ctr=float(clicks.sum() / impr.sum() * 100), impressions=float(impr.sum()), clicks=float(clicks.sum()),
                        cpc=float(spend.sum() / clicks.sum()), extrapolated=bool(p["budget_extrapolated"].any())))
    t = pd.DataFrame(out)
    t["marginal_clicks_per_euro"] = t["clicks"].diff() / t["budget"].diff()
    basis = "%d training rows with a budget change; elasticities CPM %.2f, CTR %.2f (priors %.2f, %.2f)" % (
        engine.budget_change_rows, engine.k["cpm"], engine.k["ctr"], engine.k0["cpm"], engine.k0["ctr"])
    t["note"] = np.where(t["extrapolated"], "outside the budget range seen in training: constant-elasticity extrapolation; " + basis, basis)
    return t


def simulate(engine, rows, draws=2000, seed=0):
    """Draw outcomes for each row: rate uncertainty (tau, log-normal) then click-count noise (Poisson).

    tau is the forecast error of the underlying rate with sampling noise already removed during
    calibration, so Poisson noise enters exactly once. If the engine could not be calibrated, only
    click-count noise is simulated and the `uncertainty` column says so.

    `p_best_in_audience` is the share of draws in which the ad has the highest observed CTR among
    the rows of its audience-day; draws with a tie split the credit equally among the tied ads.
    It is a model-based probability, validated only to the extent `adsim validate` reports.
    """
    rng = np.random.default_rng(seed)
    p = engine.predict(rows)
    n = rows.groupby(["date", "audience_id"])["ad_id"].transform("size").values
    spend = rows["budget"].values / n
    if engine.tau is None:
        s_ctr = s_cpm = np.zeros(len(p))
        label = "click-count noise only: forecast error not calibrated (" + engine.interval_status + ")"
    else:
        cold = p["new_ad"].values
        s_ctr = np.where(cold, engine.tau["ctr"]["cold"], engine.tau["ctr"]["warm"])
        s_cpm = np.where(cold, engine.tau["cpm"]["cold"], engine.tau["cpm"]["warm"])
        label = "rate error (held-out, count noise removed) + Poisson click noise"
    ctr = np.minimum(p["ctr"].values * np.exp(rng.normal(0, 1, (draws, len(p))) * s_ctr), MAX_CTR_PCT)
    cpm = p["cpm"].values * np.exp(rng.normal(0, 1, (draws, len(p))) * s_cpm)
    impr = spend / cpm * 1000
    clicks = np.minimum(rng.poisson(impr * ctr / 100), np.floor(impr))      # count noise cannot produce more clicks than impressions
    obs = clicks / impr * 100
    res = p[["date", "ad_id", "audience_id", "creative_id"]].copy()
    for name, arr in [("impressions", impr), ("clicks", clicks), ("ctr_observed", obs)]:
        res[name + "_p10"], res[name + "_p50"], res[name + "_p90"] = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
    freq = rows["audience_id"].map(engine.freq).astype(float).values if engine.freq else np.full(len(p), np.nan)
    res["reach_p50"] = res["impressions_p50"] / np.where(np.isfinite(freq), freq, 1.0)
    grp = pd.factorize(rows["date"].astype(str) + "|" + rows["audience_id"].astype(str))[0]
    win = np.zeros(len(p))
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        m = obs[:, idx]
        tied = np.isclose(m, m.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
        win[idx] = (tied / tied.sum(axis=1, keepdims=True)).sum(axis=0) / draws      # ties share the win
    res["p_best_in_audience"] = win
    res["uncertainty"] = label
    return res


def audiences(engine):
    """Behavioural description of each audience from its fingerprint; no demographics are inferred."""
    f = engine.fingerprint.copy()

    def describe(r):
        bits = []
        bits.append("responds %s its advertiser's average" % ("well above" if r.z_ctr > 0.8 else "above" if r.z_ctr > 0.2 else "below" if r.z_ctr < -0.2 else "near"))
        bits.append("costs %s to reach" % ("much more" if r.z_cpm > 0.8 else "more" if r.z_cpm > 0.2 else "less" if r.z_cpm < -0.2 else "about average"))
        if "z_freq" in r and np.isfinite(r.z_freq):
            bits.append("sees ads repeatedly (small pool)" if r.z_freq > 0.8 else "rarely sees an ad twice (large pool)")
        return "; ".join(bits)
    f["behaviour"] = f.apply(describe, axis=1)
    f["ctr_level_pct"], f["cpm_level_eur"] = np.exp(f["ctr"]), np.exp(f["cpm"])
    return f[["advertiser_id", "audience_id", "archetype", "ctr_level_pct", "cpm_level_eur", "behaviour"]]
