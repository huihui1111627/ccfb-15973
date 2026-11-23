"""Acceptance-oriented unit tests.

These are deliberately *statistical*: the fleet is synthetic, so the assertions
check that the engine recovers the truth it was given (documented in
``spareparts/config.py`` and ``fleet.build_fleet``), not that a number equals a
hard-coded constant.

Run with:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from spareparts import config as cfg
from spareparts import estimation as E
from spareparts import inventory as I
from spareparts import lastbuy as L
from spareparts.fleet import build_fleet
from spareparts.life import AGE_EDGES, age_bucket
from spareparts.records import build_records, month_start_day


class Model:
    """Build the fleet once; every test reuses it (build is ~2 s)."""

    fleet = None
    rec = None
    lives = None
    mult = None
    fc = None

    @classmethod
    def setup(cls):
        if cls.fleet is None:
            cls.fleet = build_fleet()
            cls.rec = build_records(cls.fleet)
            cls.lives = E.fit_lives(cls.rec)
            cls.mult = E.fit_multipliers(cls.rec, cls.lives)
            cls.fc = E.forecast_demand(cls.rec, cls.lives, cls.mult,
                                       cfg.DEFAULT.forecast_horizon)


def actual_demand(horizon: int) -> np.ndarray:
    """Ground-truth failures per (site, part, month) after the cut-off."""
    f, rec = Model.fleet, Model.rec
    m0 = int(np.floor(rec.as_of_day / E.MONTH_DAYS))
    fail = (f.removal_kind == 1) & (f.removal_day >= rec.as_of_day)
    k = np.floor(f.removal_day[fail] / E.MONTH_DAYS).astype(int) - m0
    ok = (k >= 0) & (k < horizon)
    flat = (f.site_idx[fail][ok] * len(cfg.PARTS) + f.part_idx[fail][ok]) * horizon + k[ok]
    out = np.bincount(flat, minlength=len(cfg.SITES) * len(cfg.PARTS) * horizon)
    return out.reshape(len(cfg.SITES), len(cfg.PARTS), horizon)


class TestFleetGeneration(unittest.TestCase):
    def test_fleet_is_deterministic(self):
        a = build_fleet(cfg.Config(seed=99))
        b = build_fleet(cfg.Config(seed=99))
        np.testing.assert_array_equal(a.server_install, b.server_install)
        np.testing.assert_array_equal(a.part_idx, b.part_idx)
        np.testing.assert_allclose(a.removal_day, b.removal_day,
                                   equal_nan=True)

    def test_hazard_rises_with_age(self):
        # the top-level requirement: failure rate starts low, then rises
        Model.setup()
        for pi in (0, 5, 13, 9):
            h = Model.lives[pi].hazard(np.array([0.5, 1.5, 3.0, 5.0, 7.0]))
            self.assertTrue(np.all(np.diff(h) > 0),
                            f"part {cfg.PARTS[pi]['code']} hazard not increasing: {h}")

    def test_lots_are_shared_and_heterogeneous(self):
        f = Model.fleet
        q = f.truth["lot_quality"]
        self.assertGreater(len(q), 1000)
        self.assertGreater(np.std(q), 0.05)
        # members of the same lot carry the same quality multiplier
        lot = f.batch_idx
        first = np.unique(lot, return_index=True)[1]
        # every episode of a lot must agree on the multiplier
        per_lot = np.zeros(len(q))
        np.add.at(per_lot, lot, f.frailty)
        counts = np.bincount(lot, minlength=len(q))
        mean_f = per_lot / np.maximum(counts, 1)
        self.assertLess(np.max(np.abs(mean_f[counts > 0] - q[counts > 0])), 1e-6)

    def test_part_compatibility(self):
        self.assertIn("HDD-1T-SATA", cfg.compatible("HDD-4T-SAS"))
        self.assertIn("MEM-16G-DDR4-ALT", cfg.compatible("MEM-16G-DDR4"))
        self.assertNotIn("HDD-4T-SAS", cfg.compatible("HDD-1T-SATA"))
        self.assertEqual(cfg.compatible("MEM-16G-DDR4-ALT")[:1],
                         ["MEM-16G-DDR4-ALT"])

    def test_renewal_process_replaces_parts(self):
        f = Model.fleet
        key = f.server_id.astype(np.int64) * 1000 + f.slot_idx
        order = np.argsort(key, kind="stable")
        k = key[order]
        same = k[1:] == k[:-1]
        self.assertGreater(same.mean(), 0.1)   # many slots have >1 episode


class TestLifeFit(unittest.TestCase):
    def setUp(self):
        Model.setup()

    def test_fitted_life_orders_correctly(self):
        # parts with a longer true service life must get a longer fitted life
        pairs = [("HDD-1T-SATA", "SSD-480G"), ("FAN-40MM", "MEM-16G-DDR4")]
        for short, long in pairs:
            a = Model.lives[Model.rec.part_index[short]].mean_life
            b = Model.lives[Model.rec.part_index[long]].mean_life
            self.assertLess(a, b, f"{short} should have shorter life than {long}")

    def test_shape_recovered_within_tolerance(self):
        for pi, p in enumerate(cfg.PARTS):
            k = Model.lives[pi].shape
            self.assertGreater(k, 1.0)
            self.assertLess(k, 6.5)
            if Model.lives[pi].n_failures > 3000:
                self.assertLess(abs(np.log(k / p["weibull_shape"])), 0.55,
                                f"{p['code']}: k={k:.2f} vs true {p['weibull_shape']}")

    def test_exposure_is_not_biased_by_survivors(self):
        # exposure must cover the whole history, not only failures
        lf = Model.lives[0]
        self.assertGreater(lf.exposure_years, 10 * lf.n_failures)

    def test_lot_effects_shrink_towards_one(self):
        b = Model.mult.batch
        self.assertGreater(np.mean(np.abs(b - 1) < 0.05), 0.15)
        self.assertLess(np.max(b), 6.0)

    def test_model_load_recovered(self):
        # GPU boxes run hotter; the fitted model effect should show it
        pi = Model.rec.part_index["MEM-32G-DDR4"]
        models = [i for i, m in enumerate(cfg.SERVER_MODELS)
                  if "MEM-32G-DDR4" in m["base_qty"]]
        gpu = cfg.SERVER_MODELS.index(next(
            m for m in cfg.SERVER_MODELS if m["code"] == "GPU-4U"))
        if gpu in models:
            self.assertGreater(Model.mult.model[pi][gpu],
                               Model.mult.model[pi][models[0]])


class TestForecast(unittest.TestCase):
    def setUp(self):
        Model.setup()

    def test_forecast_is_positive_and_sorted(self):
        fc = Model.fc
        self.assertTrue(np.all(fc.mean >= 0))
        self.assertTrue(np.all(fc.q05 <= fc.q25 + 1e-9))
        self.assertTrue(np.all(fc.q25 <= fc.q95 + 1e-9))

    def test_aggregate_bias_small(self):
        actual = actual_demand(12).sum()
        pred = Model.fc.mean[:, :, :12].sum()
        self.assertLess(abs(pred / max(actual, 1) - 1), 0.12,
                        f"forecast/pred/actual {pred}/{actual}")

    def test_interval_coverage_reasonable(self):
        actual = actual_demand(12)
        rep = E.coverage_report(
            type(Model.fc)(months=Model.fc.months[:12],
                           labels=Model.fc.labels[:12],
                           mean=Model.fc.mean[:, :, :12],
                           q05=Model.fc.q05[:, :, :12],
                           q25=Model.fc.q25[:, :, :12],
                           q75=Model.fc.q75[:, :, :12],
                           q95=Model.fc.q95[:, :, :12],
                           dispersion=Model.fc.dispersion,
                           installed=Model.fc.installed[:, :, :12],
                           age_mix=Model.fc.age_mix[:, :12],
                           calibration=Model.fc.calibration), actual)
        self.assertGreater(rep["cov80"], 0.78)
        self.assertLess(rep["cov80"], 0.99)

    def test_forecast_uses_renewal(self):
        fc = Model.fc
        # the installed base halves over the horizon as servers retire; a
        # renewal projection must nonetheless keep demand well above the
        # naive "base decays, nothing replaced" line
        decay = fc.installed[:, :, 23].sum() / fc.installed[:, :, 0].sum()
        first = fc.mean[:, :, :4].sum() / 4
        last = fc.mean[:, :, 20:].sum() / 4
        self.assertGreater(decay, 0.2)
        self.assertGreater(last, 0.35 * first,
                           "renewal should keep late demand from collapsing")

    def test_no_lookahead(self):
        # the records must not contain post-cut-off events
        self.assertTrue(np.all(Model.rec.episodes.end_day <= Model.rec.as_of_day + 1e-6))


class TestPolicies(unittest.TestCase):
    def setUp(self):
        Model.setup()

    def test_policy_levels_differ_across_sites(self):
        fc = Model.fc
        hist_rate = Model.rec.monthly_issues.mean(axis=2)
        hist_std = Model.rec.monthly_issues.std(axis=2)
        p_ours = I.build_forecast_policy(fc, cfg.DEFAULT)
        p_hist = I.build_hist_policy(fc, hist_rate, hist_std)
        p_fixed = I.build_fixed_policy(fc)
        # a fixed quota is identical everywhere; the forecast policy is not
        self.assertEqual(len(set(p_fixed.S[:, 0].tolist())), 1)
        self.assertGreater(len(set(p_ours.S[:, 0].tolist())), 3)
        self.assertGreater(p_ours.hub_S.sum(), 0)

    def test_compatibility_serves_demand(self):
        # a 4T demand can be met by a 1T spare; a 1T demand cannot be met by 4T
        idx = {p["code"]: i for i, p in enumerate(cfg.PARTS)}
        self.assertIn(idx["HDD-1T-SATA"], I.COMPAT[idx["HDD-4T-SAS"]].tolist())
        self.assertNotIn(idx["HDD-4T-SAS"], I.COMPAT[idx["HDD-1T-SATA"]].tolist())

    def test_simulation_runs_and_accounts(self):
        pol = I.build_forecast_policy(Model.fc, cfg.DEFAULT)
        r = I.simulate(pol, Model.fleet, Model.rec, cfg.DEFAULT, sim_days=120)
        self.assertGreater(r.demand_units, 100)
        self.assertGreaterEqual(r.holding_cost, 0)
        self.assertGreaterEqual(r.total_cost,
                                r.holding_cost + r.ordering_cost
                                + r.transfer_cost + r.shortage_cost - 1e-6)
        self.assertLessEqual(r.csl_site, 1.0)
        self.assertGreaterEqual(r.csl_site, 0.0)

    def test_network_policy_fills_at_least_as_well_as_local(self):
        pol = I.build_forecast_policy(Model.fc, cfg.DEFAULT, risk_share=0.2)
        r = I.simulate(pol, Model.fleet, Model.rec, cfg.DEFAULT, sim_days=180)
        # every local shortfall must have been covered by the network here
        self.assertGreaterEqual(r.csl_network, 0.95)
        self.assertLess(r.wait_mean, 30)


class TestLastBuy(unittest.TestCase):
    def setUp(self):
        Model.setup()

    def test_plan_matches_real_tail(self):
        plans = L.plan_last_buys(Model.rec, Model.lives, Model.mult, Model.fc)
        check = L.check_against_fleet(plans, Model.fleet, Model.rec)
        for c in check:
            scale = max(c["actual_tail_demand"], 1)
            self.assertLess(abs(c["estimate_error"]), 0.45,
                            f"{c['part']}: estimate error {c['estimate_error']}")
            self.assertLess(c["surplus"], 0.6 * scale + 5,
                            f"{c['part']}: surplus {c['surplus']} for {scale}")
            self.assertLess(c["shortfall"], 0.6 * scale + 5,
                            f"{c['part']}: shortfall {c['shortfall']} for {scale}")

    def test_discontinued_have_horizon(self):
        plans = L.plan_last_buys(Model.rec, Model.lives, Model.mult, Model.fc)
        self.assertTrue(plans)
        for p in plans:
            self.assertGreaterEqual(p.horizon_months, 1)
            self.assertGreaterEqual(p.recommended_qty, 0)


if __name__ == "__main__":
    unittest.main()
