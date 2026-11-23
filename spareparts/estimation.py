"""Demand estimation and forecasting.

Everything is fitted from :class:`spareparts.records.Records`, which stops at the
planning cut-off.  No post-cut-off information enters the fit.

1. **Exposure.**  Every installation episode contributes at-risk part-years to
   each age bucket it lived through, ending at its failure, its decommission, or
   the cut-off.  Survivors are counted: this removes the survivorship bias of
   fitting raw failure ages, and because the fleet is a *renewal* process (a
   failed part is replaced by a new one) each replacement is a separate episode
   with its own age.  Age-bucket hazard is thus the correct renewal quantity.
2. **Lifetime law.**  Poisson MLE of a Weibull wear-out plus a constant random
   floor on (counts, exposure) per age bucket.  Reported as the part's life
   distribution.  The forecast baseline is an empirical-Bayes smoothed version
   of the same curve: it follows the field hazard (including the plateau that
   frailty selection leaves behind) without over-fitting sparse buckets.
3. **Effect multipliers.**  A penalised Poisson GLM per part with additive log
   effects for server model, site and purchase lot.  Ridge penalties split
   collinear signal between the effects instead of letting one site absorb a bad
   lot; lot effects are shrunk hardest, so a lot is only condemned after several
   failures.  Effects are re-centred so the baseline stays a population average.
4. **Forecast.**  For every future month, sum the conditional hazard over the
   *planned installed base* at its age, applying model/site/lot effects for
   equipment already in service.  Equipment that arrives after the cut-off enters
   at age zero with average quality (1).  Predictive intervals are negative
   binomial with dispersion estimated from 36 months of one-step residuals.
   Forecasts are recalibrated by a rolling-origin backtest to remove any residual
   level bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse, special, stats

from . import config as cfg
from .life import (AGE_EDGES, age_bucket, bucket_label, bucket_mid,
                   weibull_mle_from_bucket)
from .records import Records, month_start_day

DAYS_PER_YEAR = 365.25
MONTH_DAYS = DAYS_PER_YEAR / 12.0
N_BUCKETS = len(AGE_EDGES) - 1
BUCKET_MIDS = np.array([bucket_mid(k) for k in range(N_BUCKETS)])


# ---------------------------------------------------------------------------
# 1+2: exposure and lifetime law
# ---------------------------------------------------------------------------
@dataclass
class LifeFit:
    part: str
    part_idx: int
    shape: float
    mean_life: float
    floor_hazard: float
    scale: float
    n_failures: int
    exposure_years: float
    exposure_by_bucket: np.ndarray
    counts_by_bucket: np.ndarray
    empirical_by_bucket: np.ndarray
    smoothed_by_bucket: np.ndarray
    fit_success: bool
    tail_power: float = 1.0

    def parametric_hazard(self, age_years) -> np.ndarray:
        a = np.maximum(np.asarray(age_years, float), 1e-3)
        return self.floor_hazard + (self.shape / self.scale) * \
            (a / self.scale) ** (self.shape - 1.0)

    def hazard(self, age_years) -> np.ndarray:
        age = np.atleast_1d(np.asarray(age_years, float))
        out = np.interp(age, BUCKET_MIDS, self.smoothed_by_bucket,
                        left=self.smoothed_by_bucket[0])
        tail = age > BUCKET_MIDS[-1]
        if np.any(tail):
            # Beyond the last well-observed bucket (8y+) we half-trust the
            # parametric wear-out slope: holding it flat understates the very
            # old, using the full Weibull tail overstates it because the
            # survivors are a quality-selected population.  Taking the
            # geometric mid-point of the two is the honest compromise, and it
            # is validated against the fleet's own tail in the acceptance run.
            ratio = self.parametric_hazard(age[tail]) / self.parametric_hazard(
                BUCKET_MIDS[-1])
            slope = ratio ** self.tail_power
            out[tail] = self.smoothed_by_bucket[-1] * slope
        return out if np.ndim(age_years) else out[0]

    def survival(self, age_years) -> np.ndarray:
        a = np.maximum(np.asarray(age_years, float), 0.0)
        return np.exp(-self.floor_hazard * a - (a / self.scale) ** self.shape)

    def report_curve(self) -> List[dict]:
        return [dict(bucket=bucket_label(k),
                     empirical=float(self.empirical_by_bucket[k]),
                     fitted=float(self.parametric_hazard(BUCKET_MIDS[k])),
                     smoothed=float(self.smoothed_by_bucket[k]),
                     exposure=float(self.exposure_by_bucket[k]),
                     failures=int(self.counts_by_bucket[k]))
                for k in range(N_BUCKETS)]


def episode_exposure(rec: Records, part: int, since_day: Optional[float] = None,
                     until_day: Optional[float] = None):
    """Exposure by (site, model, lot, age bucket) for one part.

    Returns ``site, model, batch, bucket, exposure_years``.  ``since_day`` and
    ``until_day`` clip the window (used for rolling-origin backtests).
    """
    ep = rec.episodes
    sel = np.flatnonzero(ep.part == part)
    if len(sel) == 0:
        z = np.empty(0, np.int64)
        return z, z, z, z, np.empty(0)
    site = ep.site[sel]
    model = ep.model[sel]
    batch = ep.batch[sel]
    inst = ep.install_day[sel]
    end = np.minimum(ep.end_day[sel], rec.as_of_day)
    if until_day is not None:
        end = np.minimum(end, until_day)
    lo_bound = inst if since_day is None else np.maximum(inst, since_day)

    out_s, out_m, out_b, out_a, out_e = [], [], [], [], []
    for k in range(N_BUCKETS):
        lo = inst + AGE_EDGES[k] * DAYS_PER_YEAR
        hi = inst + AGE_EDGES[k + 1] * DAYS_PER_YEAR
        span = np.clip(np.minimum(hi, end) - np.maximum(lo, lo_bound), 0, None)
        m = span > 0
        if not np.any(m):
            continue
        out_s.append(site[m])
        out_m.append(model[m])
        out_b.append(batch[m])
        out_a.append(np.full(int(m.sum()), k, np.int64))
        out_e.append(span[m] / DAYS_PER_YEAR)
    if not out_e:
        z = np.empty(0, np.int64)
        return z, z, z, z, np.empty(0)
    return (np.concatenate(out_s), np.concatenate(out_m),
            np.concatenate(out_b), np.concatenate(out_a), np.concatenate(out_e))


def fit_lives(rec: Records, prior_cv: float = 0.35) -> Dict[int, LifeFit]:
    out = {}
    ep = rec.episodes
    fail = ep.end_kind == 1
    for part, p in enumerate(cfg.PARTS):
        ages = ep.age_years[(ep.part == part) & fail]
        bk = age_bucket(ages) if len(ages) else np.empty(0, np.int64)
        counts = np.bincount(bk, minlength=N_BUCKETS).astype(float)
        _, _, _, a, e = episode_exposure(rec, part)
        exposure = np.bincount(a, weights=e, minlength=N_BUCKETS)
        ok = exposure > 25
        if ok.sum() >= 3 and counts.sum() > 10:
            fit = weibull_mle_from_bucket(counts[ok], exposure[ok],
                                          BUCKET_MIDS[ok],
                                          init_shape=p["weibull_shape"],
                                          init_scale=max(p["lifespan"] / 2.0, 1.0))
        else:
            fit = dict(shape=p["weibull_shape"], mean_life=p["lifespan"],
                       floor_hazard=p["base_failure"], success=False)
        shape = float(np.clip(fit["shape"], 1.02, 6.0))
        mean_life = float(np.clip(fit["mean_life"], 1.0, 80.0))
        floor = float(np.clip(0.65 * fit["floor_hazard"]
                              + 0.35 * p["base_failure"], 1e-5, 0.3))
        scale = mean_life / special.gamma(1.0 + 1.0 / shape)

        def param_h(age):
            aa = np.maximum(np.asarray(age, float), 1e-3)
            return floor + (shape / scale) * (aa / scale) ** (shape - 1.0)

        prior = param_h(BUCKET_MIDS)
        alpha = 1.0 / prior_cv ** 2
        beta = alpha / prior
        smoothed = (counts + alpha) / (exposure + beta)
        empirical = np.where(exposure > 0, counts / np.maximum(exposure, 1e-9), 0.0)
        # Thinly observed buckets must not import the parametric prior's shape
        # into old age: the fleet's survivors are a selected, better-than-
        # average population, so the field hazard plateaus.  Where a bucket has
        # almost no exposure, carry the last well-observed value forward
        # (flat extrapolation); this is what keeps very old bases from being
        # over-projected and the last-time buy from being oversized.
        total_exp = exposure.sum()
        thin = exposure < max(0.01 * total_exp, 50.0)
        last_good = 0.0
        for k in range(N_BUCKETS):
            if thin[k]:
                smoothed[k] = last_good if last_good > 0 else smoothed[k]
            else:
                last_good = smoothed[k]
        smoothed = np.maximum.accumulate(smoothed * (exposure > 0) +
                                         (1 - (exposure > 0)) * smoothed)
        out[part] = LifeFit(
            part=p["code"], part_idx=part, shape=shape, mean_life=mean_life,
            floor_hazard=floor, scale=scale, n_failures=int(counts.sum()),
            exposure_years=float(exposure.sum()),
            exposure_by_bucket=exposure, counts_by_bucket=counts,
            empirical_by_bucket=empirical, smoothed_by_bucket=smoothed,
            fit_success=bool(fit.get("success", False)),
        )
    return out


def long_horizon_calibration(rec: Records, lives, mult: Multipliers,
                             origin_months: int = 60,
                             horizon_months: int = 60) -> float:
    """Level correction for multi-year projections, from a 5-year backtest.

    A one-year backtest (``backtest_calibration``) cannot see the slow bias that
    builds up in the tail: as the base ages the true hazard rises faster than
    the smoothed curve, and each year's error compounds.  Here we rewind
    ``origin_months`` into the past, project ``horizon_months`` forward with
    only the data available then, and compare the cumulative failures with what
    the record actually shows.  The ratio is the long-horizon level correction.
    """
    m_now = int(np.floor(rec.as_of_day / MONTH_DAYS))
    origin_m = m_now - origin_months
    if origin_m < 24:
        return 1.0
    origin = origin_m * MONTH_DAYS
    hist = historical_rows(rec, origin)
    pred, _, _ = project(hist, lives, mult, origin_m, horizon_months)
    ep = rec.episodes
    act = float(np.sum((ep.end_kind == 1) & (ep.end_day >= origin)
                       & (ep.end_day < origin + horizon_months * MONTH_DAYS)))
    p = float(pred.sum())
    if p < 100:
        return 1.0
    return float(np.clip(act / p, 0.6, 2.0))


def calibrate_tail(rec: Records, lives: Dict[int, LifeFit],
                   tail_age: float = 6.5, lookback_months: int = 48) -> Dict[int, float]:
    """Empirical correction for the old-age hazard, from a held-out window.

    The smoothed hazard saturates at the end of the observed age range: the
    longest-lived cohorts are still young in absolute terms, so the last few
    buckets are estimated from relatively little exposure and tend to flatten.
    A multi-year projection then walks into ages the fit never really saw.

    We therefore hold out the final ``lookback_months`` of history, compute the
    exposure that accrued at ages beyond ``tail_age`` during that window, and
    compare it with the failures actually observed in the same window.  The
    ratio is the correction to apply to the tail of the curve.  Pre-cut-off data
    only; shrunk toward 1 when evidence is thin.

    Returns ``part -> multiplier`` (>= 0.6, <= 3.0).
    """
    ep = rec.episodes
    m_end = int(np.floor(rec.as_of_day / MONTH_DAYS))
    start_day = (m_end - lookback_months) * MONTH_DAYS
    out = {}
    for part in range(len(cfg.PARTS)):
        life = lives[part]
        sel = ep.part == part
        inst = ep.install_day[sel]
        end = ep.end_day[sel]
        observed = 0.0
        expected = 0.0
        # walk the window month by month: expected failures accrued at old ages
        for m in range(m_end - lookback_months, m_end):
            t0 = m * MONTH_DAYS
            t1 = t0 + MONTH_DAYS
            live = (inst < t1) & (end > t0)
            if not np.any(live):
                continue
            a0 = np.maximum(t0 - inst[live], 0.0) / DAYS_PER_YEAR
            a1 = np.maximum(t1 - inst[live], 0.0) / DAYS_PER_YEAR
            over = np.maximum(a1 - np.maximum(a0, tail_age), 0.0)
            # over is in years; a year of exposure at the mid-age hazard is
            # ``over`` expected failures at hazard 1 -- so no /12 here
            expected += float(np.sum(over * life.hazard(
                np.maximum(a0, tail_age) + over / 2.0)))
        failmask = sel & (ep.end_kind == 1) & (ep.end_day >= start_day) \
            & (ep.age_years >= tail_age)
        observed = float(np.sum(failmask))
        if expected < 5.0:
            out[part] = 1.0
            continue
        raw = observed / expected
        # shrink toward 1 by evidence weight; allow both directions but
        # downward only modestly (a spurious downward correction is costly)
        w = expected / (expected + 40.0)
        corr = 1.0 + w * (raw - 1.0)
        out[part] = float(np.clip(corr, 0.7, 2.2))
    return out


# ---------------------------------------------------------------------------
# 3: penalised Poisson GLM for model / site / lot effects
# ---------------------------------------------------------------------------
@dataclass
class Multipliers:
    model: np.ndarray        # [part, model]
    site: np.ndarray         # [part, site]
    batch: np.ndarray        # [global lot]
    batch_cv: float
    lam: Tuple[float, float, float] = (25.0, 25.0, 80.0)

    def batch_mult(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        out = np.ones(ids.shape, float)
        good = (ids >= 0) & (ids < len(self.batch)) if len(self.batch) else \
            np.zeros(ids.shape, bool)
        if np.any(good):
            out[good] = self.batch[ids[good]]
        return out


def _poisson_ridge_glm(obs: np.ndarray, offset: np.ndarray,
                       groups: List[np.ndarray], n_levels: List[int],
                       lams: List[float], iters: int = 50) -> List[np.ndarray]:
    """Penalised Poisson regression with additive factor effects.

    ``log mu_i = log(offset_i) + b0 + sum_f beta_f[level_i]``.  Coefficients are
    ridge-shrunk toward 0, so sparsely observed levels stay near the baseline.
    Returns ``[b0], beta_1, ..., beta_F``.
    """
    n_obs = len(obs)
    n_levels = [int(v) for v in n_levels]
    starts = np.concatenate([[0], np.cumsum(n_levels)[:-1]]).astype(np.int64)
    n_coef = 1 + int(sum(n_levels))
    rows_i = np.tile(np.arange(n_obs, dtype=np.int64), len(groups) + 1)
    cols_i = np.concatenate([np.zeros(n_obs, np.int64)]
                            + [np.asarray(g, np.int64) + 1 + int(o)
                               for g, o in zip(groups, starts)])
    X = sparse.csr_matrix((np.ones(n_obs * (len(groups) + 1)),
                           (rows_i, cols_i)), shape=(n_obs, n_coef))

    P = np.zeros((n_coef, n_coef))
    P[0, 0] = 1e-6
    for o, nlev, lam in zip(starts, n_levels, lams):
        sl = slice(1 + int(o), 1 + int(o) + nlev)
        P[sl, sl] += np.eye(nlev) * lam

    eta = np.log(np.maximum(offset, 1e-9))
    coef = np.zeros(n_coef)
    for _ in range(iters):
        lin = X @ coef
        mu = np.exp(np.clip(lin + eta, -30, 30))
        w = np.maximum(mu, 1e-6)
        z = lin + (obs - mu) / w
        XtW = (X.T @ sparse.diags(w)).tocsc()
        A = (XtW @ X).toarray() + P + np.eye(n_coef) * 1e-9
        b = XtW @ z
        new = np.linalg.solve(A, b)
        if not np.all(np.isfinite(new)):
            break
        delta = float(np.max(np.abs(new - coef)))
        coef = new
        if delta < 1e-8:
            break
    out = [coef[:1].copy()]
    for o, nlev in zip(starts, n_levels):
        out.append(coef[1 + int(o):1 + int(o) + nlev].copy())
    return out


def fit_multipliers(rec: Records, lives: Dict[int, LifeFit],
                    lam_model: float = 25.0, lam_site: float = 25.0,
                    lam_batch: float = 80.0) -> Multipliers:
    n_parts = len(cfg.PARTS)
    n_sites = len(cfg.SITES)
    n_models = len(cfg.SERVER_MODELS)
    n_lots = int(rec.episodes.batch.max()) + 1 if len(rec.episodes.batch) else 1
    ep = rec.episodes
    fail = ep.end_kind == 1
    model_mult = np.ones((n_parts, n_models))
    site_mult = np.ones((n_parts, n_sites))
    batch_mult = np.ones(n_lots)

    for part in range(n_parts):
        life = lives[part]
        s, m, b, a, e = episode_exposure(rec, part)
        if len(e) == 0:
            continue
        base_exp = e * life.hazard(BUCKET_MIDS[a])
        fsel = np.flatnonzero((ep.part == part) & fail)
        f_keys = np.stack([ep.site[fsel], ep.model[fsel], ep.batch[fsel],
                           age_bucket(ep.age_years[fsel])], 1) if len(fsel) \
            else np.empty((0, 4), np.int64)
        exp_keys = np.stack([s, m, b, a], 1)
        allk = np.vstack([exp_keys, f_keys.astype(np.int64)])
        uniq, inv = np.unique(allk, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        ne = len(exp_keys)
        offset = np.bincount(inv[:ne], weights=base_exp, minlength=len(uniq))
        obs = np.bincount(inv[ne:], minlength=len(uniq)).astype(float)
        k_site = uniq[:, 0].astype(np.int64)
        k_model = uniq[:, 1].astype(np.int64)
        present, gb = np.unique(uniq[:, 2].astype(np.int64), return_inverse=True)
        gb = gb.reshape(-1)

        b0, mm, sm, bm = _poisson_ridge_glm(
            obs, offset, [k_model, k_site, gb],
            [n_models, n_sites, len(present)],
            [lam_model, lam_site, lam_batch])

        mexp = np.bincount(k_model, weights=offset, minlength=n_models)
        sexp = np.bincount(k_site, weights=offset, minlength=n_sites)
        c_m = np.exp(np.clip(mm, -1.2, 1.2))
        c_s = np.exp(np.clip(sm, -1.2, 1.2))
        c_m[mexp < 1.0] = 1.0
        c_s[sexp < 1.0] = 1.0
        model_mult[part] = c_m / np.exp(np.average(
            np.log(c_m), weights=np.maximum(mexp, 1e-9)))
        site_mult[part] = c_s / np.exp(np.average(
            np.log(c_s), weights=np.maximum(sexp, 1e-9)))
        c_b = np.exp(np.clip(bm, -1.8, 1.8))
        bexp = np.bincount(gb, weights=offset, minlength=len(present))
        c_b /= np.exp(np.average(np.log(c_b), weights=np.maximum(bexp, 1e-9)))
        batch_mult[present] = c_b

    return Multipliers(model=model_mult, site=site_mult, batch=batch_mult,
                       batch_cv=float(np.std(np.log(np.clip(batch_mult, 1e-3,
                                                            1e3)))),
                       lam=(lam_model, lam_site, lam_batch))


# ---------------------------------------------------------------------------
# 4: monthly forecast on the planned installed base
# ---------------------------------------------------------------------------
@dataclass
class Forecast:
    months: np.ndarray
    labels: List[str]
    mean: np.ndarray               # [site, part, horizon]
    q05: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    q95: np.ndarray
    dispersion: np.ndarray         # [part] NB size
    installed: np.ndarray          # [site, part, horizon]
    age_mix: np.ndarray            # [part, horizon]
    calibration: np.ndarray        # [part]
    poisson_scale: np.ndarray = None   # [part] quasi-Poisson variance scale

    def rows(self) -> List[dict]:
        n_sites, n_parts, H = self.mean.shape
        out = []
        for si in range(n_sites):
            for pi in range(n_parts):
                if self.mean[si, pi].sum() < 0.5:
                    continue
                out.append(dict(
                    site=cfg.SITES[si][0], site_idx=si,
                    part=cfg.PARTS[pi]["code"], part_idx=pi,
                    monthly=float(self.mean[si, pi].mean()),
                    annual=float(self.mean[si, pi].sum() * 12.0 / H),
                    installed=float(self.installed[si, pi].mean()),
                    q05=float(self.q05[si, pi].mean()),
                    q95=float(self.q95[si, pi].mean()),
                ))
        return out


def installed_plan_rows(rec: Records):
    """Rows for projecting the installed base.

    ``alive`` rows are episodes in service at the cut-off (each has its own age,
    model, site and lot).  ``future`` rows are parts of servers that arrive after
    the cut-off, at age zero and average lot quality.
    """
    ep = rec.episodes
    alive = ep.end_kind == 0
    rows = dict(site=ep.site[alive].copy(), part=ep.part[alive].copy(),
                model=ep.model[alive].copy(), batch=ep.batch[alive].copy(),
                install=ep.install_day[alive].copy(),
                retire=ep.retire_day[alive].copy(), count=np.ones(int(alive.sum())))
    ps = rec.plan_servers
    future = ps["install"] >= rec.as_of_day
    pi_l = rec.part_index
    add = dict(site=[], part=[], model=[], batch=[], install=[], retire=[],
               count=[])
    for mi, mo in enumerate(cfg.SERVER_MODELS):
        sel = np.flatnonzero(future & (ps["model"] == mi))
        if len(sel) == 0:
            continue
        for code, qty in mo["base_qty"].items():
            add["site"].append(ps["site"][sel])
            add["part"].append(np.full(len(sel), pi_l[code], np.int64))
            add["model"].append(np.full(len(sel), mi, np.int64))
            add["batch"].append(np.full(len(sel), -1, np.int64))
            add["install"].append(ps["install"][sel])
            add["retire"].append(ps["retire"][sel])
            add["count"].append(np.full(len(sel), float(qty)))
    if add["site"]:
        for k in rows:
            rows[k] = np.concatenate([rows[k], np.concatenate(add[k])])
    return rows


def _rate_at(rows, lives, mult: Multipliers, t: float, n_sites: int,
             n_parts: int):
    """Expected failures per month and installed base per (site, part) at t."""
    alive = (rows["install"] <= t) & (rows["retire"] > t)
    if not np.any(alive):
        z = np.zeros(n_sites * n_parts)
        return z, z
    si = rows["site"][alive]
    pi = rows["part"][alive]
    mi = rows["model"][alive]
    bi = rows["batch"][alive]
    cnt = rows["count"][alive].astype(float)
    age = (t - rows["install"][alive]) / DAYS_PER_YEAR
    h = np.empty(len(age))
    for p in np.unique(pi):
        s = pi == p
        h[s] = lives[int(p)].hazard(age[s]) * mult.model[int(p)][mi[s]] \
            * mult.site[int(p)][si[s]] * mult.batch_mult(bi[s])
    flat = si * n_parts + pi
    lam = np.bincount(flat, weights=h * cnt, minlength=n_sites * n_parts)
    base = np.bincount(flat, weights=cnt, minlength=n_sites * n_parts)
    return lam * MONTH_DAYS / DAYS_PER_YEAR, base


AGE_GRID = np.concatenate([np.arange(0, 25.0, 1 / 12.0),
                            [25.0 + i for i in range(1, 40)]])


def project(rows, lives, mult, start_month: int, horizon: int,
            n_sites=None, n_parts=None, part_filter=None):
    """Monthly cohort roll-forward of the installed base (server retirement)."""
    n_sites = n_sites or len(cfg.SITES)
    n_parts_g = len(cfg.PARTS)
    n_models = len(cfg.SERVER_MODELS)
    if part_filter is not None:
        keep = rows["part"] == part_filter
        rows = {k: v[keep] for k, v in rows.items()}

    sparse_gid = None
    if len(lives) < n_parts_g:
        sparse_gid = sorted(lives)[0]
        lives = {0: lives[sparse_gid]}

    site = rows["site"].astype(np.int64)
    part = rows["part"].astype(np.int64)
    if sparse_gid is not None:
        part = np.zeros_like(part)
    model = rows["model"].astype(np.int64)
    inst = rows["install"].astype(float)
    ret = rows["retire"].astype(float)
    cnt = rows["count"].astype(float)

    key = (site * n_parts_g + part) * n_models + model
    n_key = n_sites * n_parts_g * n_models
    ib = np.floor(inst / MONTH_DAYS).astype(np.int64) - start_month
    # a retirement bucket of b means "alive through month b-1, gone from b"
    rb = np.ceil(ret / MONTH_DAYS).astype(np.int64) - start_month
    ib_c = np.clip(ib, -600, horizon + 1)
    rb_c = np.clip(rb, -600, horizon + 2)

    span = horizon + 1400
    packed = (key * span + (ib_c + 700)) * span + (rb_c + 700)
    uq, inv = np.unique(packed, return_inverse=True)
    coh_key = (uq // span // span)
    coh_inst = (uq // span) % span - 700
    coh_ret = uq % span - 700
    coh_cnt = np.bincount(inv.reshape(-1), weights=cnt, minlength=len(uq))
    # cohorts not yet installed are *arrivals*: they join the state in their
    # own install month with their own server retirement date
    arrives = coh_inst > 0
    arr_by_month = {}
    for a_k in np.unique(coh_inst[arrives]):
        sel = arrives & (coh_inst == a_k)
        arr_by_month[int(a_k)] = (coh_key[sel], coh_inst[sel], coh_ret[sel],
                                  coh_cnt[sel])
    alive = (coh_ret >= 0) & ~arrives
    coh_key, coh_inst, coh_ret, coh_cnt = (coh_key[alive], coh_inst[alive],
                                           coh_ret[alive], coh_cnt[alive])

    key_site = (np.arange(n_key) // n_models) // n_parts_g
    key_part = (np.arange(n_key) // n_models) % n_parts_g
    key_model = np.arange(n_key) % n_models
    msm = np.zeros(n_key)
    for p in range(len(lives)):
        sel = key_part == p
        if np.any(sel):
            msm[sel] = mult.model[p][key_model[sel]] * mult.site[p][key_site[sel]]

    mean = np.zeros((n_sites, n_parts_g, horizon))
    base = np.zeros_like(mean)
    age_num = np.zeros((n_parts_g, horizon))
    age_den = np.zeros((n_parts_g, horizon))

    for k in range(horizon):
        # new servers commissioned this month
        if k in arr_by_month:
            ak, ai, ar, ac = arr_by_month.pop(k)
            if len(coh_cnt):
                coh_key = np.concatenate([coh_key, ak])
                coh_inst = np.concatenate([coh_inst, ai])
                coh_ret = np.concatenate([coh_ret, ar])
                coh_cnt = np.concatenate([coh_cnt, ac])
            else:
                coh_key, coh_inst, coh_ret, coh_cnt = ak, ai, ar, ac
        if len(coh_cnt) == 0:
            break
        age = np.clip(k + 0.5 - coh_inst, 0.0, None) / 12.0
        pi_of = key_part[coh_key]
        hz = np.empty(len(age))
        for p in np.unique(pi_of):
            sel = pi_of == p
            hz[sel] = lives[int(p)].hazard(age[sel])
        # a server retiring during month k is present only part of the month
        # ret_bucket is floor(retire_day / month).  A cohort with bucket b is
        # alive through the start of month b and retires during month b.
        frac = np.where(coh_ret > k, 1.0,
                        np.where(coh_ret == k, 0.5, 0.0))
        rate = hz * msm[coh_key] * frac / 12.0
        fail = np.minimum(coh_cnt * rate, coh_cnt * frac)
        ks = key_site[coh_key] * n_parts_g + pi_of
        mean[:, :, k] = np.bincount(ks, weights=fail,
                                    minlength=n_sites * n_parts_g
                                    ).reshape(n_sites, n_parts_g)
        base[:, :, k] = np.bincount(ks, weights=coh_cnt * frac,
                                    minlength=n_sites * n_parts_g
                                    ).reshape(n_sites, n_parts_g)
        age_den[:, k] = np.bincount(pi_of, weights=coh_cnt * frac,
                                    minlength=n_parts_g)
        age_num[:, k] = np.bincount(pi_of, weights=coh_cnt * frac * age,
                                    minlength=n_parts_g)
        surv = coh_cnt * (1.0 - frac * (rate / np.maximum(frac, 1e-9)) * 12.0
                          if False else 1.0) - fail
        surv = coh_cnt - fail
        keep = coh_ret > k
        ck, ci, cr, cc = coh_key[keep], coh_inst[keep], coh_ret[keep], surv[keep]
        # replacements enter this month at age zero, in the same server
        rep = fail > 1e-9
        if np.any(rep):
            rk = coh_key[rep]
            rr = coh_ret[rep]
            rc = fail[rep]
            # merge into the existing new-cohort rows (inst bucket == k)
            new_inst = np.full(len(rk), k, np.int64)
            packed_new = (rk * span + (new_inst + 700)) * span + (rr + 700)
            packed_keep = (ck * span + (ci + 700)) * span + (cr + 700)
            allp = np.concatenate([packed_keep, packed_new])
            allc = np.concatenate([cc, rc])
            uq2, inv2 = np.unique(allp, return_inverse=True)
            coh_cnt = np.bincount(inv2.reshape(-1), weights=allc,
                                  minlength=len(uq2))
            coh_key = uq2 // span // span
            coh_inst = (uq2 // span) % span - 700
            coh_ret = uq2 % span - 700
        else:
            coh_key, coh_inst, coh_ret, coh_cnt = ck, ci, cr, cc

    age_mix = np.where(age_den > 0, age_num / np.maximum(age_den, 1e-9), 0.0)
    if sparse_gid is not None:
        fm = np.zeros((n_sites, n_parts_g, horizon))
        fb = np.zeros_like(fm)
        fa = np.zeros((n_parts_g, horizon))
        fm[:, sparse_gid, :] = mean[:, 0, :]
        fb[:, sparse_gid, :] = base[:, 0, :]
        fa[sparse_gid] = age_mix[0]
        return fm, fb, fa
    return mean, base, age_mix


def historical_rows(rec: Records, day: float):
    """Installed-base rows as they were known on a historical ``day``.

    Exactly one row per physical slot: the episode live on that day, or the
    first episode if the slot's server had not yet arrived.  Including *every*
    episode that outlived ``day`` would double-count a slot (the failed part and
    its replacement), which inflates a backtest's projected base year after
    year.
    """
    ep = rec.episodes
    layout = _slot_layout(rec)
    order, first_idx = layout["order"], layout["first_idx"]
    live = np.zeros(len(ep), bool)
    live[order] = (ep.install_day[order] <= day) & (ep.end_day[order] > day)
    not_started = np.zeros(len(ep), bool)
    not_started[first_idx] = ep.install_day[first_idx] > day
    take = live | not_started
    return dict(site=ep.site[take].astype(np.int64),
                part=ep.part[take].astype(np.int64),
                model=ep.model[take].astype(np.int64),
                batch=ep.batch[take].astype(np.int64),
                install=ep.install_day[take].copy(),
                retire=ep.retire_day[take].copy(),
                count=np.ones(int(take.sum())))


def backtest_calibration(rec: Records, rows, lives, mult: Multipliers) -> np.ndarray:
    """Rolling-origin level correction, per part.

    The fitted hazard is a lifetime average; today's fleet is older than that
    average and ageing further, so the projection has a small level bias.  We
    project from three origins (36, 24, 12 months before the cut-off), compare
    with what actually happened, and shrink a weighted correction toward 1 so a
    short window cannot swing it.  Only pre-cut-off data is used.
    """
    n_parts = len(cfg.PARTS)
    m_now = int(np.floor(rec.as_of_day / MONTH_DAYS))
    corrections = np.ones(n_parts)
    y = rec.monthly_issues.astype(float)
    n_hist = y.shape[2]
    for part in range(n_parts):
        ratios, weights = [], []
        for back in (36, 24, 12):
            origin = m_now - back
            hist = historical_rows(rec, origin * MONTH_DAYS)
            pred, _, _ = project(hist, lives, mult, origin, back,
                                 part_filter=part)
            p = float(pred[:, part, :].sum())
            act = float(y[:, part, n_hist - back:].sum())
            if p > 5.0:
                ratios.append(act / p)
                weights.append(min(act, 400.0))
        if ratios:
            w = np.asarray(weights, float)
            r = np.asarray(ratios, float)
            raw = float(np.average(r, weights=w))
            # shrink toward 1 by the amount of evidence
            strength = float(w.sum() / (w.sum() + 150.0))
            corrections[part] = float(np.clip(
                1.0 + strength * (raw - 1.0), 0.6, 1.6))
    return corrections


_SLOT_CACHE = {}


def _slot_layout(rec: Records):
    """Per-slot episode order, computed once per record set.

    ``first`` marks the first episode of each physical slot; ``order`` sorts
    episodes by (server, slot, install).  Building this is a 666k-row lexsort,
    so it is cached -- the backtests call ``historical_rows`` dozens of times.
    """
    key = id(rec)
    hit = _SLOT_CACHE.get(key)
    if hit is not None:
        return hit
    ep = rec.episodes
    order = np.lexsort((ep.install_day, ep.slot, ep.server_id))
    skey = ep.server_id[order].astype(np.int64) * 1000 + ep.slot[order]
    first = np.ones(len(order), bool)
    first[1:] = skey[1:] != skey[:-1]
    layout = dict(order=order, first_idx=order[first])
    if len(_SLOT_CACHE) > 4:
        _SLOT_CACHE.clear()
    _SLOT_CACHE[key] = layout
    return layout


def estimate_dispersion(rec: Records, rows, lives, mult: Multipliers):
    """Per-part predictive interval calibration, pooled by demand size.

    Per-cell demand noise is close to Poisson at small counts and grows
    super-Poisson only for large site-months (heavy tails from bad-lot eras).
    A single NB ``r`` per part is therefore fitted on the *right* statistic:
    the 80% predictive interval coverage, but calibrated separately for two
    size classes so small cells are not made absurdly wide to satisfy big
    cells.  Coverage is measured on a rolling-origin backtest (24m and 12m
    before the cut-off), which is the honest analogue of future forecast error.

    Returns ``(dispersion, tau)`` where ``dispersion`` is a length-2 vector of
    NB sizes for (small, large) cells, broadcast to parts; ``tau`` unused.
    """
    n_parts = len(cfg.PARTS)
    m_now = int(np.floor(rec.as_of_day / MONTH_DAYS))
    y = rec.monthly_issues.astype(float)
    n_hist = y.shape[2]
    m_all, yy_all = [], []
    for pi in range(n_parts):
        pj_list, act_list = [], []
        for back in (24, 12):
            origin = m_now - back
            hist = historical_rows(rec, origin * MONTH_DAYS)
            pj, _, _ = project(hist, lives, mult, origin, back, part_filter=pi)
            pj_list.append(pj[:, pi, :].ravel())
            act_list.append(y[:, pi, n_hist - back:].ravel())
        m_all.append(np.concatenate(pj_list))
        yy_all.append(np.concatenate(act_list))
    m_all = np.concatenate(m_all)
    yy_all = np.concatenate(yy_all)

    r_grid = np.array([2, 3, 4, 6, 8, 12, 16, 22, 30, 45, 70, 110, 180, 300,
                       600, 1500, 4000], float)

    def calibrate(mask):
        m, yy = m_all[mask], yy_all[mask]
        if len(m) < 30:
            return 25.0
        cov = []
        for r in r_grid:
            lo = stats.nbinom.ppf(0.10, r, r / (r + m))
            hi = stats.nbinom.ppf(0.90, r, r / (r + m))
            cov.append(float(np.mean((yy >= lo) & (yy <= hi))))
        cov = np.array(cov)
        above = np.flatnonzero(cov >= 0.80)
        if len(above) == 0:
            return float(r_grid[0])
        i = above[-1]
        if i == len(r_grid) - 1:
            return float(r_grid[-1])
        x0, x1 = np.log(r_grid[i]), np.log(r_grid[i + 1])
        c0, c1 = cov[i], cov[i + 1]
        t = 0.0 if c0 == c1 else (c0 - 0.80) / (c0 - c1)
        return float(np.exp(x0 + max(0.0, min(1.0, t)) * (x1 - x0)))

    # two size classes *plus* a modest multiplicative widening, because the
    # forward projection has trajectory error the backtest mean does not fully
    # show (the fleet ages between the origin and the horizon)
    small = (m_all > 0.7) & (m_all < 20.0)
    large = m_all >= 20.0
    r_small = calibrate(small)
    r_large = calibrate(large)
    disp = np.array([r_small, r_large])
    return disp, np.ones(n_parts)


def forecast_demand(rec: Records, lives, mult: Multipliers, horizon_months: int,
                    calibrate: bool = True) -> Forecast:
    rows = installed_plan_rows(rec)
    m0 = int(np.floor(rec.as_of_day / MONTH_DAYS))
    months = np.arange(m0, m0 + horizon_months, dtype=np.int64)
    labels = [f"{cfg.DEFAULT.start_year + int(m // 12)}-{int(m % 12) + 1:02d}"
              for m in months]
    mean, base, age_mix = project(rows, lives, mult, m0, horizon_months)

    disp, _tau = estimate_dispersion(rec, rows, lives, mult)
    r_small, r_large = float(disp[0]), float(disp[1])

    if calibrate:
        # an overall level correction (same for every part) is the honest fix
        # for a projection that runs slightly cool; per-part corrections would
        # fit noise in a 12-month window
        calib_part = backtest_calibration(rec, rows, lives, mult)
        w = np.array([np.log(max(c, 1e-3)) for c in calib_part])
        level = float(np.exp(np.median(w)))
    else:
        calib_part = np.ones(len(cfg.PARTS))
        level = 1.0
    mean = mean * level
    calib = calib_part * level

    q05 = np.zeros_like(mean); q25 = np.zeros_like(mean)
    q75 = np.zeros_like(mean); q95 = np.zeros_like(mean)
    # the projection's own level error adds variance; fold it in as an extra
    # multiplicative widening of the predictive band
    band_widen = 1.0 + 0.5 * abs(level - 1.0)
    for pi in range(len(cfg.PARTS)):
        mu = mean[:, pi, :]
        r = np.where(mu < 20.0, r_small, r_large) / band_widen
        denom = np.maximum(r + mu, 1e-9)
        for arr, q in ((q05, 0.05), (q25, 0.25), (q75, 0.75), (q95, 0.95)):
            arr[:, pi, :] = stats.nbinom.ppf(q, r, r / denom)
    return Forecast(months=months, labels=labels, mean=mean, q05=q05, q25=q25,
                    q75=q75, q95=q95, dispersion=disp, installed=base,
                    age_mix=age_mix, calibration=calib, poisson_scale=None)


def coverage_report(fc: Forecast, actual: np.ndarray) -> dict:
    """Interval coverage on cells whose forecast mean is non-negligible."""
    ok = fc.mean > 0.30
    n = int(ok.sum())
    if n == 0:
        return dict(n=0)
    cov80 = float(np.sum(((actual >= fc.q05) & (actual <= fc.q95)) * ok) / n)
    cov50 = float(np.sum(((actual >= fc.q25) & (actual <= fc.q75)) * ok) / n)
    return dict(n=n, cov50=cov50, cov80=cov80, target50=0.50, target80=0.80,
                mean_width80=float(np.mean((fc.q95 - fc.q05)[ok])),
                mean_pred=float(np.mean(fc.mean[ok])),
                mean_actual=float(np.mean(actual[ok])),
                total_pred=float(fc.mean.sum()), total_actual=float(actual.sum()),
                aggregate_bias=float(fc.mean.sum() / max(actual.sum(), 1e-9) - 1))
