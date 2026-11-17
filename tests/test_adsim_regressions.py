"""One regression test per bug reproduced on 14 Nov 2026 (see docs/ADOPTION.md, 'Robustness fixes')."""
import copy
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from adsim import Engine, load                                  # noqa: E402
from adsim.data import flag_anomalies                            # noqa: E402
from adsim.evaluate import backtest                              # noqa: E402
from adsim.simulate import simulate                              # noqa: E402

B = load(ROOT / "configs" / "wevion_bighack.json")
B["cfg"]["encoder"] = "tfidf"                                    # no model download in tests
H, C, CFG, FUT = B["history"], B["creatives"], B["cfg"], B["future"]


def thin(h, factor=40, seed=0):
    """Low-volume copy of real rows: keep 1 impression in `factor`, clicks thinned binomially."""
    rng = np.random.default_rng(seed)
    t = h.copy()
    t["impressions"] = np.maximum((t["impressions"] / factor).round(), 20)
    t["clicks"] = rng.binomial(h["clicks"].astype(int).values, np.minimum(t["impressions"] / h["impressions"], 1.0))
    t["spend"] = t["spend"] * t["impressions"] / h["impressions"]
    if "reach" in t:
        t["reach"] = np.maximum((t["reach"] / factor).round(), 1)
    t["ctr"] = t["clicks"] / t["impressions"] * 100
    return t


class SmallData(unittest.TestCase):
    def test_one_audience(self):
        e = Engine(CFG).fit(H[H.audience_id == "NDL-AS1"], C)
        self.assertEqual(set(e.arch.values()), {"A0"})
        self.assertTrue(np.isfinite(e.predict(FUT[FUT.audience_id == "NDL-AS1"])[["ctr", "cpm"]].values).all())

    def test_one_creative(self):
        h, c = H[H.creative_id == "NDL-S2"], C[C.creative_id == "NDL-S2"]
        e = Engine(CFG).fit(h, c)
        self.assertEqual([x for x in e.cfeat.columns if x.startswith("emb")], [])     # no zero-dimensional projection
        self.assertTrue(np.isfinite(e.predict(h.tail(5))[["ctr", "cpm"]].values).all())

    def test_seven_days_of_history(self):
        h = H[H.date >= H.date.max() - pd.Timedelta(days=6)]
        e = Engine(CFG).calibrate(h, C).fit(h, C)
        self.assertIsNone(e.sigma)
        self.assertIn("not calibrated", e.interval_status)
        p = e.predict(FUT.head(20))
        self.assertTrue(np.isfinite(p[["ctr", "cpm"]].values).all() and p["ctr_lo"].isna().all())   # points yes, invented bands no
        self.assertIn("click-count noise only", simulate(e, FUT.head(20), draws=50)["uncertainty"].iloc[0])

    def test_adaptive_holdout_with_three_weeks(self):
        h = H[H.date >= H.date.max() - pd.Timedelta(days=20)]
        e = Engine(CFG).calibrate(h, C)
        self.assertIsNotNone(e.sigma)
        self.assertEqual(e.calibration["holdout_days"], 7)

    def test_empty_history_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "No usable history rows"):
            Engine(CFG).fit(H.iloc[:0], C)


class ZeroClicks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low = thin(H[H.quality == ""])
        cls.cut = pd.Timestamp("2026-11-02")
        cls.e = Engine(CFG).fit(cls.low[cls.low.date < cls.cut], C)

    def test_valid_zero_days_are_kept_and_modelled(self):
        flags = flag_anomalies(self.low.drop(columns="quality"))
        zeros = flags[flags.clicks == 0]
        self.assertGreater(len(zeros), 200)
        self.assertGreater((zeros.quality == "").mean(), 0.95)              # legitimate, not anomalies
        self.assertEqual(self.e.ctr_mode, "poisson")
        self.assertEqual(self.e.zero_click_rows["kept_in_training"], self.e.zero_click_rows["valid_zero_click_rows"])

    def test_no_upward_bias_against_full_volume_truth(self):
        """Dropping zero-click days (the old behaviour, still available by forcing log-normal)
        inflates CTR; modelling them as counts does not. Truth = the same rows before thinning."""
        val = self.low[self.low.date >= self.cut]
        truth = H[(H.quality == "") & (H.date >= self.cut)]
        clicks_true = (truth["ctr"].values * truth["impressions"].values).sum()

        def bias(engine):
            return (engine.predict(val)["ctr"].values * truth["impressions"].values).sum() / clicks_true - 1
        forced = copy.deepcopy(CFG)
        forced["ctr_likelihood"] = "lognormal"
        dropped = Engine(forced).fit(self.low[self.low.date < self.cut], C)
        self.assertEqual(dropped.zero_click_rows["kept_in_training"], 0)
        self.assertLess(abs(bias(self.e)), 0.15)
        self.assertGreater(bias(dropped), bias(self.e) + 0.15)

    def test_suspicious_zero_still_flagged(self):
        flags = flag_anomalies(H.drop(columns="quality"))
        self.assertEqual(set(flags[flags.quality == "suspicious_zero_clicks"]["date"].dt.strftime("%m-%d")), {"10-14", "11-09"})
        bad = H.head(3).assign(impressions=[0, 100, 100], clicks=[0, 500, -1])
        self.assertTrue((flag_anomalies(bad.drop(columns="quality")).quality == "invalid").all())

    def test_backtest_scores_zero_rows(self):
        r = backtest({"history": self.low, "creatives": C, "cfg": CFG}, cutoffs=["2026-11-02"], draws=100).iloc[0]
        self.assertGreater(r["zero_click_rows_scored"], 50)
        self.assertTrue(np.isfinite([r["ctr_mae_pp"], r["ctr_wape_clicks"], r["ctr_poisson_deviance"]]).all())


class HighCardinality(unittest.TestCase):
    def test_300_creatives_and_unseen_levels(self):
        rng = np.random.default_rng(0)
        ids = [f"CR{i:03d}" for i in range(300)]
        big = H[H.quality == ""].copy()
        big["creative_id"] = rng.choice(ids, len(big))
        big["ad_id"] = big["audience_id"] + "|" + big["creative_id"]
        cr = pd.concat([C.iloc[[i % len(C)]].assign(creative_id=ids[i]) for i in range(300)])
        e = Engine(CFG).fit(big, cr)
        self.assertIn("creative_id", e.tree_dropped)
        e.add_creatives(C.head(1).assign(creative_id="NEVER-SEEN"))
        rows = big.tail(10).assign(creative_id="NEVER-SEEN", ad_id="NEW", audience_id="UNSEEN-AUDIENCE")
        self.assertTrue(np.isfinite(e.predict(rows)[["ctr", "cpm"]].values).all())


class SimulationTies(unittest.TestCase):
    def test_identical_ads_get_equal_probability_in_any_order(self):
        e = Engine(CFG).calibrate(H, C).fit(H, C)
        r = FUT[(FUT.audience_id == "NDL-AS3") & (FUT.date == "2026-11-27")].head(1)
        twins = pd.concat([r.assign(ad_id=a) for a in ["T1", "T2", "T3"]]).assign(budget=3.0)     # tiny volume: many 0-0-0 draws
        for frame in (twins, twins.iloc[::-1]):
            s = simulate(e, frame, draws=6000).set_index("ad_id")["p_best_in_audience"]
            self.assertAlmostEqual(s.sum(), 1.0, places=6)
            self.assertLess(s.max() - s.min(), 0.03)


class Uncertainty(unittest.TestCase):
    def test_rate_error_excludes_count_noise(self):
        e = Engine(CFG).calibrate(H, C)
        self.assertLess(e.tau["ctr"]["warm"], e.sigma["ctr"]["warm"] / 1.2816)     # count noise removed, so strictly smaller

    def test_later_window_coverage_is_reported_by_group(self):
        r = backtest(B, cutoffs=["2026-11-02"], draws=300).iloc[0]
        for k in ["cov80_ctr_simulated_known", "cov80_ctr_simulated_new", "cov80_cpm_band_known", "cov80_cpm_band_new"]:
            self.assertTrue(0 <= r[k] <= 1)
        self.assertLess(r["cov80_ctr_simulated_known"], 0.90)                       # was 0.85-0.91 when noise was counted twice


class CtrCeiling(unittest.TestCase):
    """CTR is a share of impressions. The log-linear forecast has no upper limit of its own, so a
    high-rate history plus a budget cut (CTR elasticity is negative) used to forecast above 100%."""

    @classmethod
    def setUpClass(cls):
        hot = H[H.quality == ""].copy()
        hot["clicks"] = (hot["impressions"] * 0.9).round()                      # valid rows: clicks <= impressions
        hot["ctr"] = hot["clicks"] / hot["impressions"] * 100
        cls.hot = hot

    def _engine(self, likelihood):
        cfg = copy.deepcopy(CFG)
        cfg["ctr_likelihood"] = likelihood
        return Engine(cfg).calibrate(self.hot, C).fit(self.hot, C)

    def test_forecast_and_band_never_exceed_100_percent(self):
        for likelihood in ("lognormal", "poisson"):
            e = self._engine(likelihood)
            self.assertEqual(e.ctr_mode, likelihood)
            p = e.predict(FUT, budget_multiplier=0.01)                          # deep budget cut pushes CTR up
            for col in ("ctr", "ctr_lo", "ctr_hi"):
                self.assertTrue(np.isfinite(p[col]).all(), (likelihood, col))
                self.assertLessEqual(p[col].max(), 100.0, (likelihood, col))
            self.assertGreater(p["ctr"].max(), 99.0)                            # the ceiling is actually reached here
            self.assertTrue((p["ctr_lo"] <= p["ctr"]).all() and (p["ctr"] <= p["ctr_hi"]).all())

    def test_simulated_clicks_never_exceed_impressions(self):
        for likelihood in ("lognormal", "poisson"):
            s = simulate(self._engine(likelihood), FUT.assign(budget=FUT["budget"] * 0.01), draws=400)
            self.assertLessEqual(s["ctr_observed_p90"].max(), 100.0, likelihood)
            self.assertTrue((s["clicks_p90"] <= s["impressions_p90"]).all(), likelihood)

    def test_ordinary_forecasts_are_untouched(self):
        p = Engine(CFG).calibrate(H, C).fit(H, C).predict(FUT)
        self.assertLess(p["ctr_hi"].max(), 50.0)                                # nowhere near the ceiling on real data


class ValidationBoundaries(unittest.TestCase):
    def test_training_cutoff_is_respected(self):
        cut = pd.Timestamp("2026-11-02")
        train = H[H.date < cut]
        e = Engine(CFG).fit(train, C)
        self.assertEqual(e.creative_protocol, "train_only")
        self.assertEqual(e.space.n_fit_creatives, train.creative_id.nunique())      # projection never saw later creatives
        poisoned = pd.concat([train, H[H.date >= cut].assign(clicks=0)])            # later rows must not change training flags
        again = flag_anomalies(train.drop(columns="quality"))
        self.assertEqual(e.excluded, again.loc[again.quality != "", "quality"].value_counts().to_dict())
        self.assertEqual(Engine(CFG).fit(poisoned[poisoned.date < cut], C).excluded, e.excluded)

    def test_transductive_protocol_is_opt_in_and_labelled(self):
        cfg = copy.deepcopy(CFG)
        cfg["creative_protocol"] = "transductive"
        e = Engine(cfg).fit(H[H.date < "2026-11-02"], C)
        self.assertEqual(e.space.n_fit_creatives, len(C))

    def test_reference_budget_fallback_uses_first_observed_value(self):
        m = load(ROOT / "configs" / "minimal_example.json")["history"]
        first = m.sort_values("date").groupby("audience_id")["budget"].first()
        self.assertTrue((m["base_budget"].values == m["audience_id"].map(first).values).all())


if __name__ == "__main__":
    unittest.main()


class SharedAudiencesAcrossAdvertisers(unittest.TestCase):
    """Found by adversarial review on the Kaggle file: with no advertiser level, two campaigns that
    share audience names got identical predicted rates and the smaller ones were biased by -19% to -38%."""

    def test_each_advertiser_keeps_its_own_level(self):
        a = H[(H.quality == "") & (H.advertiser_id == "NDL-C1")].copy()
        b = a.copy()
        b["advertiser_id"] = "TWIN"
        b["ad_id"] = "T-" + b["ad_id"]
        b["clicks"] = (b["clicks"] * 0.5).round()                       # same audiences and creatives, half the response
        b["ctr"] = b["clicks"] / b["impressions"] * 100
        b = b[b["clicks"] > 0]
        e = Engine(CFG).fit(pd.concat([a, b]), C)
        self.assertTrue(e.shared_audiences)
        pa, pb = e.predict(a.tail(200))["ctr"].values, e.predict(b.tail(200))["ctr"].values
        self.assertLess(abs(np.median(pb / pa) - 0.5), 0.08)
        self.assertEqual(len(e.arch), 2 * a["audience_id"].nunique())   # archetypes not overwritten across advertisers
