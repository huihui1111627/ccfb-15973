"""The CMMS-style record set the engine is allowed to see.

All exports are cut at the planning date (``Config.as_of_year``-01-01).  The
estimator never touches the post-cut-off portion of the fleet.

``episodes``
    One row per part *installation*: site, part, server model, purchase lot,
    install date, and the date it left the server (failure or decommission).
    Rows still in service at the cut-off end at the cut-off (right-censored).
    This is the exposure denominator and the failure record in one table, which
    is exactly how a CMMS records swaps.

``monthly_issues``
    Thirty-six months of replacement consumption per (site, part).  Input to the
    naive "order the historical average" benchmark.

``plan_servers``
    Server-level install and retirement plan, for projecting the installed base
    of the *next* few years.  Parts in servers that arrive after the cut-off are
    new, so they start at age zero.

``discontinued``
    Parts the vendor has flagged end-of-life, and the last year they may be
    bought.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import config as cfg
from .fleet import DAYS_PER_YEAR, Fleet, year_start


@dataclass
class EpisodeTable:
    site: np.ndarray
    part: np.ndarray
    model: np.ndarray
    batch: np.ndarray
    install_day: np.ndarray
    end_day: np.ndarray          # failure/decommission, or the cut-off if censored
    end_kind: np.ndarray         # 1 failure, 2 decommission, 0 censored at cut-off
    retire_day: np.ndarray       # planned retirement of the server it sits in
    server_id: np.ndarray
    slot: np.ndarray
    episode_id: np.ndarray

    def __len__(self) -> int:
        return len(self.site)

    @property
    def age_years(self) -> np.ndarray:
        return (self.end_day - self.install_day) / DAYS_PER_YEAR

    def subset(self, sel: np.ndarray) -> "EpisodeTable":
        return EpisodeTable(**{k: getattr(self, k)[sel] for k in (
            "site", "part", "model", "batch", "install_day", "end_day",
            "end_kind", "retire_day", "server_id", "slot", "episode_id")})


@dataclass
class Records:
    as_of_day: float
    episodes: EpisodeTable
    plan_servers: dict
    monthly_issues: np.ndarray      # [site, part, month]
    months: np.ndarray              # absolute month indices
    part_index: Dict[str, int]
    site_codes: List[str]
    discontinued: Dict[str, int]
    # ground truth for acceptance only:
    _fleet: Optional[Fleet] = None

    def failures(self) -> EpisodeTable:
        return self.episodes.subset(self.episodes.end_kind == 1)

    def live_at_cutoff(self) -> EpisodeTable:
        return self.episodes.subset(self.episodes.end_kind == 0)


def month_index(day, start_year: int) -> np.ndarray:
    years = start_year + np.asarray(day, float) / DAYS_PER_YEAR
    y = np.floor(years).astype(int)
    m = np.clip(np.floor((years - y) * 12.0).astype(int), 0, 11)
    return (y - start_year) * 12 + m


def month_start_day(month: int, start_year: int) -> float:
    y = start_year + int(month) // 12
    m = int(month) % 12
    return (y - start_year + m / 12.0) * DAYS_PER_YEAR


def build_records(fleet: Fleet, c: cfg.Config = cfg.DEFAULT,
                  as_of_year: Optional[int] = None) -> Records:
    as_of_year = as_of_year or c.as_of_year
    as_of = float(year_start(as_of_year))
    part_index = {p["code"]: i for i, p in enumerate(cfg.PARTS)}
    n_parts, n_sites = len(cfg.PARTS), len(cfg.SITES)

    idx = np.flatnonzero(fleet.install_day < as_of)
    removal = fleet.removal_day[idx]
    kind = fleet.removal_kind[idx].copy()
    end_day = np.where(np.isnan(removal), as_of, removal)
    past = end_day < as_of
    kind = np.where(past, kind, 0).astype(np.int8)
    end_day = np.minimum(end_day, as_of)

    episodes = EpisodeTable(
        site=fleet.site_idx[idx].astype(np.int64),
        part=fleet.part_idx[idx].astype(np.int64),
        model=fleet.server_model_idx[fleet.server_id[idx]].astype(np.int64),
        batch=fleet.batch_idx[idx].astype(np.int64),
        install_day=fleet.install_day[idx],
        end_day=end_day,
        end_kind=kind,
        retire_day=np.maximum(fleet.retire_day[idx], end_day),
        server_id=fleet.server_id[idx].astype(np.int64),
        slot=fleet.slot_idx[idx].astype(np.int64),
        episode_id=np.arange(len(idx), dtype=np.int64),
    )

    # ---- server plan -----------------------------------------------------
    plan = dict(site=fleet.server_site.astype(np.int64),
                model=fleet.server_model_idx.astype(np.int64),
                install=fleet.server_install.copy(),
                retire=fleet.server_retire.copy())
    keep = plan["install"] < as_of + 365.25 * (c.eval_years + c.sim_years + 3)
    for k in plan:
        plan[k] = plan[k][keep]

    # ---- 36 whole months of issues before the cut-off --------------------
    n_hist = 36
    m_end = int(np.floor(as_of / (DAYS_PER_YEAR / 12.0)))
    m_start = m_end - n_hist
    fail = episodes.end_kind == 1
    in_hist = fail & (episodes.end_day >= month_start_day(m_start, c.start_year)) \
        & (episodes.end_day < month_start_day(m_end, c.start_year))
    midx = month_index(episodes.end_day[in_hist], c.start_year)
    flat = (episodes.site[in_hist] * n_parts + episodes.part[in_hist]) * n_hist \
        + (midx - m_start)
    monthly = np.bincount(flat, minlength=n_sites * n_parts * n_hist)
    monthly = monthly.reshape(n_sites, n_parts, n_hist).astype(np.int64)

    # ---- discontinuations ------------------------------------------------
    discontinued = {}
    for p in cfg.PARTS:
        models_using = [m for m in cfg.SERVER_MODELS if p["code"] in m["base_qty"]]
        last_purchase = max((m["purchase_years"][1] for m in models_using),
                            default=0)
        if last_purchase <= 2017:
            discontinued[p["code"]] = 2022
        elif last_purchase <= 2020:
            discontinued[p["code"]] = 2025

    return Records(
        as_of_day=as_of, episodes=episodes, plan_servers=plan,
        monthly_issues=monthly,
        months=np.arange(m_start, m_end, dtype=np.int64),
        part_index=part_index, site_codes=[s[0] for s in cfg.SITES],
        discontinued=discontinued, _fleet=fleet,
    )
