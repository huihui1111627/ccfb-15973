"""Lifetime distributions for field-replaced parts.

We use a competing-risks model:

* a Weibull wear-out component (shape > 1 -> hazard starts low and rises), whose
  scale is calibrated from the part's expected service life;
* a constant "random" hazard floor (the flat middle of the bathtub curve);
* a Gamma frailty multiplier shared by every unit of the same purchase batch,
  which makes bad batches fail together (spatial/temporal clustering);
* a load multiplier per server model (GPU boxes run hotter than 1U web boxes).

The model is used both to generate synthetic ground truth and as the parametric
family the estimator fits back from the failure records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import special

DAYS_PER_YEAR = 365.25

# Age bucket edges in years used by the estimator / reporting.
AGE_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])


def age_bucket(age_years: np.ndarray) -> np.ndarray:
    """0-based index of the age bucket for each age."""
    idx = np.searchsorted(AGE_EDGES, age_years, side="right") - 1
    return np.clip(idx, 0, len(AGE_EDGES) - 2)


def bucket_label(i: int) -> str:
    lo, hi = AGE_EDGES[i], AGE_EDGES[i + 1]
    return f"{lo:g}-{hi:g}y"


def bucket_mid(i: int) -> float:
    lo, hi = AGE_EDGES[i], AGE_EDGES[i + 1]
    return (lo + hi) / 2.0 if hi < 10 else 9.0


@dataclass(frozen=True)
class WeibullLife:
    """Population lifetime law for one part code."""
    shape: float
    mean_life: float          # Weibull component mean, in years
    floor_hazard: float       # annual random-failure hazard
    batch_cv: float           # Gamma frailty coefficient of variation
    load: float = 1.0

    @property
    def scale(self) -> float:
        # scale = mean / Gamma(1 + 1/k)
        return self.mean_life / special.gamma(1.0 + 1.0 / self.shape)

    def hazard(self, age: np.ndarray | float) -> np.ndarray:
        age = np.asarray(age, dtype=float)
        a = np.maximum(age, 1e-6)
        h = (self.shape / self.scale) * (a / self.scale) ** (self.shape - 1.0)
        return (self.floor_hazard + h) * self.load

    def hazard_population(self, age: np.ndarray | float) -> np.ndarray:
        """Hazard averaged over the batch-frailty distribution (mean 1)."""
        return self.hazard(age)

    def survival(self, age: np.ndarray | float) -> np.ndarray:
        age = np.asarray(age, dtype=float)
        a = np.maximum(age, 0.0)
        return np.exp(-self.floor_hazard * a
                      - (a / self.scale) ** self.shape)

    def sample(self, u_wear: np.ndarray, u_floor: np.ndarray,
               frailty: np.ndarray | float = 1.0) -> np.ndarray:
        """Sample failure age in years from uniforms (vectorised).

        ``u_wear`` drives the Weibull component, ``u_floor`` the exponential
        floor; failure occurs at the earlier of the two.  ``frailty`` scales the
        wear-out hazard only.
        """
        u_wear = np.clip(u_wear, 1e-12, 1 - 1e-12)
        u_floor = np.clip(u_floor, 1e-12, 1 - 1e-12)
        f = np.asarray(frailty, dtype=float)
        if self.floor_hazard <= 1e-9:
            t_floor = np.full_like(u_floor, np.inf)
        else:
            t_floor = -np.log(u_floor) / self.floor_hazard
        t_wear = self.scale * (-np.log(u_wear)) ** (1.0 / self.shape) \
            / np.maximum(f, 1e-6) ** (1.0 / self.shape)
        return np.minimum(t_floor, t_wear) / self.load


def gamma_frailty(rng: np.random.Generator, n: int, cv: float) -> np.ndarray:
    """Batch quality multiplier: mean 1, coefficient of variation ``cv``."""
    if cv <= 1e-9:
        return np.ones(n)
    shape = 1.0 / (cv * cv)
    scale = cv * cv
    return rng.gamma(shape, scale, size=n)


def weibull_mle_from_bucket(counts: np.ndarray, exposure: np.ndarray,
                            ages: np.ndarray, init_shape: float = 2.0,
                            init_scale: float = 6.0,
                            fix_floor: float | None = None):
    """Poisson MLE for (shape, scale, floor) from binned counts/exposure.

    ``counts`` and ``exposure`` are arrays over age buckets, ``ages`` the bucket
    midpoints in years.  Returns a dict with shape, mean_life, floor_hazard.
    """
    from scipy.optimize import minimize

    counts = np.asarray(counts, float)
    exposure = np.asarray(exposure, float)
    ages = np.asarray(ages, float)
    sel = exposure > 0
    counts, exposure, ages = counts[sel], exposure[sel], ages[sel]

    def nll(theta):
        logk, logscale, logfloor = theta
        k = np.exp(logk)
        scale = np.exp(logscale)
        floor = np.exp(logfloor)
        a = np.maximum(ages, 1e-3)
        h = floor + (k / scale) * (a / scale) ** (k - 1.0)
        mu = exposure * h
        return float(np.sum(mu - counts * np.log(np.maximum(mu, 1e-12))))

    x0 = np.array([np.log(init_shape), np.log(init_scale), np.log(0.008)])
    bounds = [(np.log(1.05), np.log(6.0)), (np.log(0.5), np.log(60.0)),
              (np.log(1e-5), np.log(0.2))]
    if fix_floor is not None:
        def nll2(theta):
            logk, logscale = theta
            k = np.exp(logk)
            scale = np.exp(logscale)
            a = np.maximum(ages, 1e-3)
            h = fix_floor + (k / scale) * (a / scale) ** (k - 1.0)
            mu = exposure * h
            return float(np.sum(mu - counts * np.log(np.maximum(mu, 1e-12))))
        res = minimize(nll2, x0[:2], method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-7, fatol=1e-7))
        k, scale = np.exp(res.x[0]), np.exp(res.x[1])
        floor = fix_floor
    else:
        res = minimize(nll, x0, method="Nelder-Mead",
                       options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
        k, scale, floor = np.exp(res.x)
    mean_life = scale * special.gamma(1.0 + 1.0 / k)
    return dict(shape=float(k), mean_life=float(mean_life),
                floor_hazard=float(floor), scale=float(scale),
                nll=float(res.fun), success=bool(res.success))
