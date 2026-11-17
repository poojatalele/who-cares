"""The forecasting and simulation engine. Company-agnostic: it sees only the canonical frame.

log KPI = audience level + creative effect + ad effect + fatigue + log(season) + k * log(budget / base budget) + noise

* Levels are partially pooled (group-wise ridge penalties): an ad with little data falls back on
  its creative, its audience archetype and its advertiser.
* Season enters only by division and multiplication, so no learner can cancel or amplify it.
* k (budget elasticity) starts at a shipped prior and moves toward the customer's own budget
  changes in proportion to how much budget variation their history contains.
* Audience archetypes are clustered from behaviour on the training rows; no IDs or names are read.
* CTR likelihood: log-normal when every training row has clicks (the large-count regime), Poisson
  on click counts with impressions as exposure as soon as valid zero-click rows exist. Zero-click
  days are therefore modelled, never dropped and never replaced by a small constant.
* Everything that is learned (quality flags, archetypes, text projection, elasticities, bands) is
  learned from the rows passed to fit(), so a time split cannot leak.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

from .creative import CreativeSpace
from .data import flag_anomalies

KPIS = ("ctr", "cpm")
Z80 = 1.2816
MAX_TREE_LEVELS = 200          # sklearn's histogram trees accept at most 255 categories
DENSE_CELLS = 2e7
MAX_CTR_PCT = 100.0             # CTR is a percentage: a rate above 100% would mean more clicks than impressions


class _Hier:
    def __init__(self, blocks):
        self.blocks = blocks                      # list of (kind, name, scale)

    def _X(self, d, fit=False):
        if fit:
            self.cats, self.stats = {}, {}
        out, n = [], len(d)
        for kind, name, scale in self.blocks:
            if kind == "cat":
                s = d[name].astype(str)
                if fit:
                    self.cats[name] = {c: i for i, c in enumerate(sorted(s.unique()))}
                idx = self.cats[name]
                cols = s.map(idx)                 # unseen levels -> NaN -> an all-zero row
                keep = cols.notna().values
                out.append(sparse.csr_matrix((np.full(keep.sum(), scale), (np.flatnonzero(keep), cols[keep].astype(int).values)),
                                             shape=(n, len(idx))))
            elif kind == "std":
                v = d[name].values.astype(float)
                if fit:
                    self.stats[tuple(name)] = (v.mean(0), v.std(0) + 1e-9)
                mu, sd = self.stats[tuple(name)]
                out.append(sparse.csr_matrix((v - mu) / sd * scale))
            else:
                out.append(sparse.csr_matrix(d[name].values.astype(float) * scale))
        X = sparse.hstack(out, format="csr")
        return X.toarray() if X.shape[0] * X.shape[1] <= DENSE_CELLS else X      # sparse only when it has to be

    def fit(self, d, y, weight=None, poisson_alpha=None):
        X = self._X(d, True)
        self.poisson = poisson_alpha is not None
        if self.poisson:
            self.model = PoissonRegressor(alpha=poisson_alpha, max_iter=3000, tol=1e-7).fit(X, y, sample_weight=weight)
        else:
            self.model = Ridge(alpha=1.0).fit(X, y)
        return self

    def predict(self, d):
        p = self.model.predict(self._X(d))
        return np.log(p) if self.poisson else p


class Engine:
    def __init__(self, cfg, level_days=7):
        self.cfg, self.level_days = cfg, level_days
        self.blend = cfg.get("blend", {"ctr": ["hier"], "cpm": ["hier", "tree", "level"]})
        self.k0 = cfg.get("priors", {}).get("budget_elasticity", {"ctr": -0.1, "cpm": 0.3})
        self.prior_weight = cfg.get("priors", {}).get("prior_weight", 1.0)
        self.creative_cats = (cfg.get("creatives") or {}).get("categorical", [])
        self.creative_protocol = cfg.get("creative_protocol", "train_only")     # or "transductive"
        ai = cfg.get("audience_interactions", 0.0)                               # penalty scale; 0 or false = off
        self.audience_interactions = float(0.7 if ai is True else ai or 0.0)
        self.sigma = self.tau = None
        self.interval_status = "not calibrated: call calibrate() first"

    # ---------- features ----------
    def _archetypes(self, h):
        """Cluster audiences on behaviour relative to their own advertiser (pooled rates, so
        zero-click days count)."""
        a = h.groupby(["advertiser_id", "audience_id"]).agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"),
                                                            spend=("spend", "sum")).reset_index()
        a["ctr"] = np.log((a["clicks"] + 0.5) / a["impressions"] * 100)           # +0.5 only to describe audiences
        a["cpm"] = np.log((a["spend"] / a["impressions"] * 1000).where(a["spend"] > 0))     # no spend recorded -> unknown price
        a["cpm"] = a["cpm"].fillna(a.groupby("advertiser_id")["cpm"].transform("mean")).fillna(0.0)
        cols = ["ctr", "cpm"]
        if "reach" in h and h["reach"].notna().all() and (h["reach"] > 0).all():
            r = h.groupby("audience_id").agg(i=("impressions", "sum"), r=("reach", "sum"))
            a["freq"] = np.log(a["audience_id"].map(r["i"] / r["r"]))
            cols.append("freq")
        g = a[["advertiser_id", "audience_id"] + cols].copy()
        z = g[cols] - g.groupby("advertiser_id")[cols].transform("mean")
        z = (z / z.std(ddof=0).replace(0, 1.0)).fillna(0.0)                         # one audience -> all zeros, not NaN
        n = int(max(1, min(self.cfg.get("archetypes", 4), len(g) // 2)))
        if n < 2 or len(np.unique(z.values.round(9), axis=0)) < n:
            g["archetype"] = "A0"                                                  # too few audiences to tell types apart
        else:
            km = KMeans(n, n_init=20, random_state=0).fit(z.values)
            order = np.argsort(np.argsort(km.cluster_centers_[:, 0]))
            g["archetype"] = ["A%d" % order[c] for c in km.labels_]
        self.fingerprint = g.join(z.add_prefix("z_"))
        # Keyed by advertiser AND audience: the same audience name in two accounts is two audiences.
        return dict(zip(g["advertiser_id"].astype(str) + "\x1f" + g["audience_id"].astype(str), g["archetype"]))

    def _design(self, df):
        d = df.copy()
        d["adv_aud"] = d["advertiser_id"].astype(str) + "\x1f" + d["audience_id"].astype(str)
        d["archetype"] = d["adv_aud"].map(self.arch).fillna("A?")
        d["weeks"] = d["age_days"] / 7.0
        d["log_age"] = np.log1p(d["age_days"])
        d["wk0"] = (d["age_days"] < 7).astype(float)
        d["weekday"] = d["date"].dt.weekday.astype(str)
        feats = self.cfeat.reindex(d["creative_id"].values)
        if len(self.cfeat.columns) and feats.isna().any().any():
            raise ValueError("Unknown creative: add it to the creatives table (or call add_creatives) before forecasting.")
        for c in self.cfeat.columns:
            d[c] = feats[c].values
        cats = self.ccat.reindex(d["creative_id"].values)
        for c in self.creative_cats:
            d["c_" + c] = cats[c].astype(str).values
            d["ax_" + c] = d["archetype"] + "|" + d["c_" + c]
            d["vx_" + c] = d["advertiser_id"] + "|" + d["c_" + c]
            d["ux_" + c] = d["adv_aud"] + "|" + d["c_" + c]           # this audience's own response to the lever / format
        for a in self.arch_levels:
            d["fat_" + a] = d["weeks"] * (d["archetype"] == a)
        return d

    def _blocks(self, with_budget):
        b = [("cat", "advertiser_id", 3.0), ("cat", "audience_id", 3.0), ("cat", "creative_id", 1.0), ("cat", "ad_id", 0.7), ("cat", "weekday", 0.5)]
        if self.shared_audiences:
            # The same audience definition is used by several advertisers: each advertiser keeps its
            # own level and its own deviation from the shared audience effect.
            b += [("cat", "adv_aud", 1.5)]
        for c in self.creative_cats:
            b += [("cat", "c_" + c, 1.0), ("cat", "ax_" + c, 0.7), ("cat", "vx_" + c, 0.7)]
            if self.audience_interactions:
                # Each campaign targets a specific audience, and audiences respond differently to the same
                # lever. Partially pooled: with little evidence it shrinks to the archetype and brand effect.
                b += [("cat", "ux_" + c, self.audience_interactions)]
        if len(self.cfeat.columns):
            b += [("std", list(self.cfeat.columns), 0.5)]
        b += [("raw", ["weeks", "log_age", "wk0"] + ["fat_" + a for a in self.arch_levels], 1.0)]
        if with_budget:
            b += [("raw", ["lbr"], 5.0 / max(self.prior_weight, 1e-6) ** 0.5)]
        return b

    # ---------- fit ----------
    def fit(self, history, creatives):
        flagged = flag_anomalies(history.drop(columns=["quality"], errors="ignore"))     # training rows only
        self.excluded = flagged.loc[flagged["quality"] != "", "quality"].value_counts().to_dict()
        h = flagged[flagged["quality"] == ""].copy()
        if h.empty:
            raise ValueError("No usable history rows to fit on (%d supplied, exclusions: %s). The engine needs daily rows "
                             "with positive impressions for at least one ad." % (len(history), self.excluded or "none"))
        zero = int((h["clicks"] == 0).sum())
        mode = self.cfg.get("ctr_likelihood", "auto")
        self.ctr_mode = "poisson" if mode == "poisson" or (mode == "auto" and zero > 0) else "lognormal"
        self.zero_click_rows = {"valid_zero_click_rows": zero, "kept_in_training": zero if self.ctr_mode == "poisson" else 0}
        if self.ctr_mode == "lognormal" and zero:
            h = h[h["clicks"] > 0]                      # only reachable when the config forces log-normal
        self.trained_through = h["date"].max()
        self.shared_audiences = bool((h.groupby("audience_id")["advertiser_id"].nunique() > 1).any())
        self.arch = self._archetypes(h)
        self.arch_levels = sorted(set(self.arch.values()))
        creatives = creatives.assign(creative_id=creatives["creative_id"].astype(str))
        in_training = creatives[creatives["creative_id"].isin(h["creative_id"].unique())]
        basis = creatives if self.creative_protocol == "transductive" or in_training.empty else in_training
        self.space = CreativeSpace(self.cfg).fit(basis)
        self.cfeat = self.space.transform(creatives)
        self.ccat = creatives.set_index("creative_id")[self.creative_cats] if self.creative_cats else pd.DataFrame(index=creatives["creative_id"])
        self.cfeat = self.cfeat.reindex(self.cfeat.index.union(h["creative_id"].unique()))
        d = self._design(h)
        self.k, self.m, self.level, self.nonpositive_rows = {}, {}, {}, {}
        tcat = ["advertiser_id", "audience_id", "creative_id", "archetype"] + ["c_" + c for c in self.creative_cats]
        # IDs with more levels than a histogram tree can take are left to the partially pooled linear
        # model, which holds them sparsely with shrinkage; the tree still sees their features.
        self.tree_dropped = [c for c in tcat if d[c].astype(str).nunique() > MAX_TREE_LEVELS]
        self.tcat = [c for c in tcat if c not in self.tree_dropped]
        self.tnum = list(self.cfeat.columns) + ["weeks", "log_age"]
        self.tlevels = {c: sorted(d[c].astype(str).unique()) for c in self.tcat}
        scale = self._blocks(True)[-1][2]
        recent = d["date"] > self.trained_through - pd.Timedelta(days=self.level_days)
        for kpi in KPIS:
            season = np.log(d[f"season_{kpi}"].values / 100)
            if kpi == "ctr" and self.ctr_mode == "poisson":
                def exposure(k):                                    # impressions on the budget-neutral, de-seasonalised scale
                    return d["impressions"].values * np.exp(season + k * d["lbr"].values) / 100
                e0 = exposure(self.k0[kpi])
                alpha = float(d["clicks"].sum() / e0.sum()) / len(d)       # same prior-to-data ratio as the ridge at an average row
                s1 = _Hier(self._blocks(True)).fit(d, d["clicks"].values / e0, e0 / e0.mean(), alpha)
                self.k[kpi] = float(self.k0[kpi] + s1.model.coef_[-1] * scale)
                ex = exposure(self.k[kpi])
                hier = _Hier(self._blocks(False)).fit(d, d["clicks"].values / ex, ex / ex.mean(), alpha)
                tree = None
                if "tree" in self.blend[kpi]:
                    tree = _PoissonTree().fit(self._tx(d), d["clicks"].values / ex, ex / ex.mean())
                last = d.assign(ex=ex)[recent.values]
                lv = lambda key: np.log((last.groupby(key)["clicks"].sum() / last.groupby(key)["ex"].sum()).replace(0, np.nan)).to_dict()
                self.level[kpi] = (lv("ad_id"), lv("audience_id"))
            else:
                # A log-normal rate needs a positive value. Zero-spend rows occur when the account
                # is billed per click; they cannot enter a log price model and are counted, not hidden.
                pos = d[kpi].values > 0
                self.nonpositive_rows[kpi] = int((~pos).sum())
                if not pos.any():
                    raise ValueError(f"No rows with a positive {kpi.upper()} to fit on.")
                dk, sk = d[pos], season[pos]
                y = np.log(dk[kpi].values) - sk
                s1 = _Hier(self._blocks(True)).fit(dk, y - self.k0[kpi] * dk["lbr"].values)
                self.k[kpi] = float(self.k0[kpi] + s1.model.coef_[-1] * scale)
                yn = y - self.k[kpi] * dk["lbr"].values
                hier = _Hier(self._blocks(False)).fit(dk, yn)
                tree = None
                if "tree" in self.blend[kpi]:
                    tree = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=15,
                                                         l2_regularization=1.0, categorical_features="from_dtype", random_state=0).fit(self._tx(dk), yn)
                last = dk.assign(yn=yn)[recent.values[pos]]
                self.level[kpi] = (last.groupby("ad_id")["yn"].mean().to_dict(), last.groupby("audience_id")["yn"].mean().to_dict())
            self.m[kpi] = (hier, tree)
        self.seen_ads, self.seen_creatives = set(d["ad_id"]), set(d["creative_id"])
        self.seen_lbr = (float(d["lbr"].min()), float(d["lbr"].max()))
        self.seen_season = {k: (float(d[f"season_{k}"].min()), float(d[f"season_{k}"].max())) for k in KPIS}
        self.budget_variation = float((d["lbr"].abs() > 1e-9).mean())
        self.budget_change_rows = int((d["lbr"].abs() > 1e-9).sum())
        self.freq = (h.groupby("audience_id")["impressions"].sum() / h.groupby("audience_id")["reach"].sum()).to_dict() if "reach" in h else {}
        self.n_train_rows = len(d)
        return self

    def add_creatives(self, creatives):
        """Register creatives that did not exist at fit time (no refit: encoder and model are frozen)."""
        creatives = creatives.assign(creative_id=creatives["creative_id"].astype(str))
        self.cfeat = pd.concat([self.cfeat[~self.cfeat.index.isin(creatives["creative_id"])], self.space.transform(creatives)])
        if self.creative_cats:
            self.ccat = pd.concat([self.ccat[~self.ccat.index.isin(creatives["creative_id"])], creatives.set_index("creative_id")[self.creative_cats]])
        return self

    def _tx(self, d):
        X = d[self.tcat + self.tnum].copy()
        for c in self.tcat:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.tlevels[c])       # unseen level -> missing, which trees accept
        return X

    # ---------- predict ----------
    def predict(self, df, budget_multiplier=1.0):
        d = self._design(df)
        lbr = d["lbr"].values + np.log(budget_multiplier)
        cold = ~d["ad_id"].isin(self.seen_ads).values
        out = df[["date", "ad_id", "audience_id", "creative_id"]].copy()
        for kpi in KPIS:
            hier, tree = self.m[kpi]
            parts = {"hier": hier.predict(d)}
            if tree is not None:
                parts["tree"] = tree.predict(self._tx(d))
            lv = d["ad_id"].map(self.level[kpi][0]).fillna(d["audience_id"].map(self.level[kpi][1])).values.astype(float)
            use = [parts[p] for p in self.blend[kpi] if p in parts]
            base = np.mean(use, axis=0)
            if "level" in self.blend[kpi]:
                ok = np.isfinite(lv) & ~cold
                base = np.where(ok, (np.sum(use, axis=0) + np.nan_to_num(lv)) / (len(use) + 1), base)
            out[kpi] = np.exp(base + np.log(d[f"season_{kpi}"].values / 100) + self.k[kpi] * lbr)
            band = self._band(kpi)
            if band is None:
                out[kpi + "_lo"] = out[kpi + "_hi"] = np.nan
            else:
                q = np.where(cold, band["cold"], band["warm"])
                out[kpi + "_lo"], out[kpi + "_hi"] = out[kpi] * np.exp(-q), out[kpi] * np.exp(q)
            if kpi == "ctr":                        # the log-linear model is unbounded above; probability is not
                for col in ("ctr", "ctr_lo", "ctr_hi"):
                    out[col] = np.minimum(out[col], MAX_CTR_PCT)
        out["new_ad"] = cold
        out["new_creative"] = ~d["creative_id"].isin(self.seen_creatives).values
        out["budget_extrapolated"] = (lbr > self.seen_lbr[1] + 1e-9) | (lbr < self.seen_lbr[0] - 1e-9)
        out["season_extrapolated"] = (d["season_cpm"].values > self.seen_season["cpm"][1] * 1.02) | (d["season_ctr"].values > self.seen_season["ctr"][1] * 1.02)
        return out

    def _band(self, kpi):
        """Half-width (log scale) of the 80% band that predict() reports, and what it refers to."""
        if kpi == "ctr" and getattr(self, "ctr_mode", "lognormal") == "poisson":
            return None if self.tau is None else {w: Z80 * self.tau["ctr"][w] for w in ("warm", "cold")}
        return None if self.sigma is None else self.sigma[kpi]

    @property
    def interval_kind(self):
        if self.sigma is None and self.tau is None:
            return {"ctr": "none", "cpm": "none"}
        return {"ctr": "underlying click rate (use simulate() for observed clicks)" if self.ctr_mode == "poisson" else "observed daily value",
                "cpm": "observed daily value"}

    def calibrate(self, history, creatives, holdout_days=14, coverage=0.8, min_train_days=7, min_holdout_days=3):
        """Empirical error from the most recent days, forecast by a model that never saw them.

        The hold-out adapts to the history available. With too little history no interval is
        produced and `interval_status` says so. Two quantities are estimated, for known and new ads:
          sigma  spread of observed daily values around the forecast (what a daily band needs)
          tau    spread of the underlying rate around the forecast, i.e. sigma with click-count
                 noise removed by moments: Var(clicks) = m + m^2 (exp(tau^2) - 1) for a Poisson
                 count with log-normal rate error. simulate() draws tau and adds Poisson noise
                 once, so sampling noise is not counted twice.
        """
        days = np.sort(history["date"].unique())
        hold = min(holdout_days, len(days) // 3)
        self.sigma = self.tau = None
        if hold < min_holdout_days or len(days) - hold < min_train_days:
            self.interval_status = ("not calibrated: %d days of history; need at least %d training days plus %d held-out days. "
                                    "Point forecasts are still produced; no interval is reported." % (len(days), min_train_days, min_holdout_days))
            return self
        cut = days[-hold]
        probe = Engine(self.cfg, self.level_days).fit(history[history["date"] < cut], creatives)
        held = flag_anomalies(history.drop(columns=["quality"], errors="ignore"))
        held = held[(held["date"] >= cut) & (held["quality"] == "")]
        if len(held) < 20:
            self.interval_status = "not calibrated: only %d valid held-out rows." % len(held)
            return self
        p = probe.predict(held)
        cold = p["new_ad"].values
        self.sigma, self.tau, counts = {}, {}, {"warm_rows": int((~cold).sum()), "cold_rows": int(cold.sum())}

        def by_group(fn, min_rows=10):
            warm = fn(~cold) if (~cold).sum() >= min_rows else fn(np.ones(len(cold), bool))
            cold_v = fn(cold) if cold.sum() >= min_rows else warm
            return {"warm": warm, "cold": max(cold_v, warm)}
        r_cpm = np.abs(np.log(held["cpm"].values / p["cpm"].values))
        self.sigma["cpm"] = by_group(lambda m: float(np.quantile(r_cpm[m], coverage)))
        self.tau["cpm"] = {w: v / Z80 for w, v in self.sigma["cpm"].items()}          # no count noise in price
        pos = held["clicks"].values > 0
        r_ctr = np.abs(np.log(np.where(pos, held["ctr"].values, 1.0) / p["ctr"].values))
        self.sigma["ctr"] = by_group(lambda m: float(np.quantile(r_ctr[m & pos], coverage)) if (m & pos).sum() else float("nan"))
        mexp = held["impressions"].values * p["ctr"].values / 100
        zstat = ((held["clicks"].values - mexp) ** 2 - mexp) / mexp ** 2
        self.tau["ctr"] = by_group(lambda m: float(np.sqrt(np.log1p(max(zstat[m].mean(), 0.0)))))
        self.calibration = dict(counts, holdout_start=str(pd.Timestamp(cut).date()), holdout_days=int(hold), nominal_coverage=coverage,
                                cold_source="held-out new ads" if cold.sum() >= 10 else "fewer than 10 new-ad rows: known-ad error used as a floor")
        self.interval_status = ("empirical %d%% bands from %d held-out days starting %s; coverage on later data is reported by "
                                "`adsim validate`, not assumed" % (round(coverage * 100), hold, pd.Timestamp(cut).date()))
        return self


class _PoissonTree:
    """Boosted trees on click rate with impressions as weight (Poisson deviance); predicts log rate."""

    def fit(self, X, rate, weight):
        self.m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=15,
                                               l2_regularization=1.0, categorical_features="from_dtype", random_state=0).fit(X, rate, sample_weight=weight)
        return self

    def predict(self, X):
        return np.log(self.m.predict(X))
