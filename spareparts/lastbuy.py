"""Last-time buy for discontinued parts.

When a part goes end-of-life the vendor takes one final order.  Order too much
and the surplus is scrapped at end of service; order too little and the shortfall
has to be sourced as a substitute, refurbished, or (worst case) the server is
retired early.

Total remaining demand for a part is the expected number of failures from the
planning date until the last server carrying it is decommissioned.  We get it
from the same renewal projection the demand forecast uses:

* the installed base and its planned retirements (``records.plan_servers``),
* the fitted hazard by age, with model/site effects,
* renewal: each failure is replaced by a fresh part, which can fail again --

so the integral over the tail is not just "today's stock times hazard", which
would badly under-count a long tail.

The order quantity is the ``target`` quantile of a compound (NB) process, so the
plan has a stated probability of not running short.  The result is compared with
what actually happened in the simulator: units left over versus units short.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from . import config as cfg
from .estimation import (DAYS_PER_YEAR, MONTH_DAYS, Forecast, LifeFit,
                         Multipliers, historical_rows, installed_plan_rows,
                         project)
from .records import Records, month_start_day

N_PARTS = len(cfg.PARTS)
N_SITES = len(cfg.SITES)


@dataclass
class LastBuy:
    part: str
    part_idx: int
    discontinued_year: int
    buy_month: int                  # absolute month index of the last order
    horizon_months: int
    expected_demand: float
    q10: float
    q50: float
    q90: float
    recommended_qty: float
    installed: float
    last_retirement: float          # year the last server carrying it retires
    target: float
    stock_on_hand: float = 0.0
    expected_within_sim: float = 0.0

    def row(self) -> dict:
        return dict(part=self.part, eol_year=self.discontinued_year,
                    expected_within_sim=round(self.expected_within_sim, 0),
                    buy_month=self.buy_month, expected=round(self.expected_demand, 0),
                    q10=round(self.q10, 0), q50=round(self.q50, 0),
                    q90=round(self.q90, 0), recommended=round(self.recommended_qty, 0),
                    installed=round(self.installed, 0),
                    last_retirement=round(self.last_retirement, 1),
                    target=self.target)


def _months_until_last_retirement(rec: Records, part: int,
                                  rows) -> Tuple[int, float]:
    ps = rec.plan_servers
    pi = part
    code = cfg.PARTS[part]["code"]
    models = [i for i, m in enumerate(cfg.SERVER_MODELS) if code in m["base_qty"]]
    sel = np.isin(ps["model"], models)
    if not np.any(sel):
        return 1, float(cfg.DEFAULT.as_of_year)
    last_day = float(ps["retire"][sel].max())
    last = last_day / DAYS_PER_YEAR + cfg.DEFAULT.start_year
    months = int(np.ceil((last_day - rec.as_of_day) / MONTH_DAYS))
    return max(months, 1), last


def plan_last_buys(rec: Records, lives: Dict[int, LifeFit],
                   mult: Multipliers, fc: Forecast,
                   stock_on_hand: Optional[np.ndarray] = None,
                   target_quantile: float = 0.90) -> List[LastBuy]:
    """Last-time-buy quantity for every discontinued part."""
    rows = installed_plan_rows(rec)
    out: List[LastBuy] = []
    m_now = int(np.floor(rec.as_of_day / MONTH_DAYS))
    for code, eol_year in rec.discontinued.items():
        pi = rec.part_index[code]
        life = lives[pi]
        months, last_year = _months_until_last_retirement(rec, pi, rows)
        months = min(months, 12 * 12)
        # renewal projection of demand over the tail
        mean, base, _ = project(rows, {pi: life}, mult, m_now, months,
                                part_filter=pi)
        demand_by_month = mean[:, pi, :].sum(axis=0)
        # The tail is long and the projection's level error is roughly constant
        # in relative terms, so apply the same pre-cut-off level correction the
        # demand forecast uses.  Without it the buy is ~15-25% short, which is
        # exactly the bias the backtest is there to remove.
        level = float(np.exp(np.median(np.log(np.maximum(fc.calibration, 1e-6)))))
        demand_by_month = demand_by_month * level
        total = float(demand_by_month.sum())
        sim_months = int(round(cfg.DEFAULT.sim_years * 12))
        within_sim = float(demand_by_month[:min(sim_months, months)].sum())
        # dispersion is a (small, large) pair for the interval model
        r = float(max(fc.dispersion) if fc.dispersion is not None else 20.0)
        # Two error sources, and the second dominates:
        #  * month-to-month demand noise (small, averages out over years);
        #  * uncertainty in the hazard *level* of the extrapolated tail, which
        #    is common to every month and therefore scales with the total.  A
        #    multi-year backtest puts that level error around 12-18% depending
        #    on how far past the observed age range the tail runs.
        monthly_var = float(np.sum(demand_by_month + demand_by_month ** 2 / r))
        tail_years = months / 12.0
        level_cv = float(np.clip(0.06 + 0.02 * max(0.0, tail_years - 3.0),
                                 0.06, 0.22))
        var = monthly_var + (level_cv * total) ** 2
        target = float(target_quantile)
        sd = max(var, 1e-9) ** 0.5
        z = stats.norm.ppf(target)
        q90 = float(np.ceil(total + z * sd))
        q10 = float(np.floor(max(0.0, total - z * sd)))
        on_hand = 0.0
        if stock_on_hand is not None:
            on_hand = float(stock_on_hand[pi])
        recommended = max(0.0, q90 - on_hand)
        out.append(LastBuy(
            part=code, part_idx=pi, discontinued_year=eol_year,
            buy_month=m_now, horizon_months=months, expected_demand=total,
            q10=q10, q50=total, q90=q90, recommended_qty=recommended,
            installed=float(base[:, pi, 0].sum()),
            last_retirement=last_year, target=target,
            stock_on_hand=on_hand, expected_within_sim=within_sim))
    return out


def check_against_fleet(plans: List[LastBuy], fleet, rec: Records) -> List[dict]:
    """Reconcile each last-time buy with the fleet's realised tail demand.

    Ground-truth check only (never fed back into the plan): count every actual
    failure of the part from the planning date until the last server carrying it
    is decommissioned, then report surplus or shortfall.  This is the number the
    acceptance asks for -- how much of the last order is left over, and how much
    demand it failed to cover.
    """
    out = []
    for plan in plans:
        pi = plan.part_idx
        code = plan.part
        models = [i for i, m in enumerate(cfg.SERVER_MODELS)
                  if code in m["base_qty"]]
        ps = rec.plan_servers
        sel = np.isin(ps["model"], models)
        if np.any(sel):
            end_day = float(ps["retire"][sel].max())
        else:
            end_day = rec.as_of_day
        mask = ((fleet.part_idx == pi) & (fleet.removal_kind == 1)
                & (fleet.removal_day >= rec.as_of_day)
                & (fleet.removal_day <= end_day))
        actual = float(mask.sum())
        qty = plan.recommended_qty
        out.append(dict(part=code, recommended=round(qty, 0),
                        actual_tail_demand=round(actual, 0),
                        estimate=round(plan.expected_demand, 0),
                        estimate_error=round(
                            (plan.expected_demand - actual) / max(actual, 1e-9), 3),
                        surplus=round(max(0.0, qty - actual), 0),
                        shortfall=round(max(0.0, actual - qty), 0),
                        surplus_value=round(max(0.0, qty - actual)
                                            * cfg.UNIT_COST if False else 0.0, 0),
                        tail_years=round((end_day - rec.as_of_day)
                                         / DAYS_PER_YEAR, 1)))
        # surplus units x unit cost, computed here to keep the import lean
        out[-1]["surplus_value"] = round(
            out[-1]["surplus"] * next(p["unit_cost"] for p in cfg.PARTS
                                      if p["code"] == code), 0)
    return out


def last_buy_vector(plans: List[LastBuy]) -> np.ndarray:
    v = np.zeros(N_PARTS)
    for p in plans:
        v[p.part_idx] = p.recommended_qty
    return v
