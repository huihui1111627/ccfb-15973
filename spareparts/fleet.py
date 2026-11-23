"""Synthetic server fleet and its ground-truth failure process.

A fleet is generated once from ``Config.seed``: server install waves, purchase
lots, latent lot quality, and the *sequence of parts* each physical position in
each server holds over time.  When a part fails it is replaced by a part from
that year's purchase lot, which can itself fail, so the failure process is a
renewal process -- as in a real fleet.

The record layer (:mod:`spareparts.records`) exposes only what a CMMS would
export; the latent parameters here are used for validation and for the oracle
baseline, never by the estimator or the inventory policy.

Ground-truth structure (deliberately non-trivial):

* hazard is Weibull wear-out plus a flat random floor (bathtub curve);
* each (site, server model, part, half-year purchase wave) is a *lot* with a
  Gamma quality multiplier, so failures cluster in bad lots;
* a share of replacements are refurbished pulls with a higher hazard;
* the same part fails faster in hot workloads (GPU) and hot regions;
* servers are decommissioned on a retirement plan, censoring their parts;
* old-generation parts are discontinued, so their last purchase is finite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import config as cfg
from .life import WeibullLife, gamma_frailty

DAYS_PER_YEAR = 365.25


@dataclass
class Fleet:
    # one row per part *installation episode* (a position can appear repeatedly)
    server_id: np.ndarray
    slot_idx: np.ndarray
    part_idx: np.ndarray
    site_idx: np.ndarray
    batch_idx: np.ndarray
    install_day: np.ndarray
    retire_day: np.ndarray
    removal_day: np.ndarray        # NaN when right-censored
    removal_kind: np.ndarray       # 1 failure, 2 decommission, 0 censored
    frailty: np.ndarray
    episode_id: np.ndarray
    # one row per server
    server_model_idx: np.ndarray
    server_site: np.ndarray
    server_install: np.ndarray
    server_retire: np.ndarray
    # metadata
    n_days: int = 0
    n_batches: int = 0
    site_codes: List[str] = field(default_factory=list)
    # latent truth, for validation only
    truth: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_servers(self) -> int:
        return len(self.server_model_idx)


def _calendar(year: int) -> int:
    return int(round((year - cfg.DEFAULT.start_year) * DAYS_PER_YEAR))


def _sample_install_day(rng, model: dict, last_day: float) -> float:
    """Server install date from the model's purchase window (lumpy tranches)."""
    y0, y1 = model["purchase_years"]
    a = _calendar(y0)
    b = min(_calendar(y1 + 1), last_day + 365.25)
    w = rng.dirichlet(np.full(4, 1.4))
    tranche = int(rng.choice(4, p=w))
    u = rng.random()
    lo = a + (b - a) * tranche / 4.0
    hi = a + (b - a) * (tranche + 1) / 4.0
    return lo + (hi - lo) * u


def build_fleet(c: cfg.Config = cfg.DEFAULT) -> Fleet:
    rng = np.random.default_rng(c.seed)
    parts = cfg.PARTS
    models = cfg.SERVER_MODELS
    part_index = {p["code"]: i for i, p in enumerate(parts)}
    n_parts, n_sites, n_models = len(parts), len(cfg.SITES), len(models)

    hub_index = {h: i for i, h in enumerate(cfg.HUB_SITES)}
    hub_bias = rng.uniform(0.55, 1.45, size=len(cfg.HUB_SITES))
    model_weights = np.array([1.0, 1.15, 1.25, 1.0, 0.45, 0.8])
    site_model_mix = rng.dirichlet(model_weights * 4.0, size=n_sites) * 0.6 \
        + model_weights / model_weights.sum() * 0.4
    site_model_mix /= site_model_mix.sum(axis=1, keepdims=True)

    n_days = int(365.25 * 26)
    last_install = _calendar(c.start_year + c.warm_years + c.eval_years + 2)
    horizon = float(n_days - 60)

    # ---- servers ---------------------------------------------------------
    sm, ss, si, sr = [], [], [], []
    for s_i, (code, prov, hub, n_serv) in enumerate(cfg.SITES):
        keep_p = min(1.0, 0.35 + hub_bias[hub_index[hub]] * 0.5)
        for _ in range(n_serv):
            m = int(rng.choice(n_models, p=site_model_mix[s_i]))
            inst = _sample_install_day(rng, models[m], last_install)
            if inst > last_install + 365.25:
                continue
            if rng.random() > keep_p:
                continue
            refresh = rng.random() < 0.22
            life = models[m]["retire_year"] * (1.35 if refresh else 1.0)
            sm.append(m)
            ss.append(s_i)
            si.append(inst)
            sr.append(inst + life * 365.25)
    server_model = np.asarray(sm, np.int16)
    server_site = np.asarray(ss, np.int16)
    server_install = np.asarray(si, np.float64)
    server_retire = np.asarray(sr, np.float64)
    n_servers = len(server_model)

    # ---- expand servers into physical positions -------------------------
    pos_server, pos_slot, pos_part = [], [], []
    for srv in range(n_servers):
        mo = models[int(server_model[srv])]
        k = 0
        for pcode, qty in mo["base_qty"].items():
            pi = part_index[pcode]
            q = qty
            if rng.random() < 0.04:          # a few boxes are under-populated
                q = max(0, q - int(rng.integers(0, 3)))
            for _ in range(q):
                pos_server.append(srv)
                pos_slot.append(k)
                pos_part.append(pi)
                k += 1
    pos_server = np.asarray(pos_server, np.int32)
    pos_slot = np.asarray(pos_slot, np.int16)
    pos_part = np.asarray(pos_part, np.int16)
    pos_site = server_site[pos_server]
    pos_model = server_model[pos_server]
    pos_retire = server_retire[pos_server]

    # latent effects
    site_quality = rng.lognormal(0.0, c.site_quality_sigma, n_sites)
    site_mult = np.array([cfg.REGION_COLD[h] for _, _, h, _ in cfg.SITES]) \
        * site_quality
    part_mult = rng.uniform(0.88, 1.12, n_parts)
    model_load = np.array([m["rho_scale"] for m in models])
    base_mult = model_load[pos_model] * site_mult[pos_site] * part_mult[pos_part]

    law = [WeibullLife(shape=p["weibull_shape"], mean_life=p["lifespan"],
                       floor_hazard=p["base_failure"], batch_cv=p["batch_scale"])
           for p in parts]

    # ---- purchase lots ---------------------------------------------------
    lot_cache: Dict[tuple, tuple] = {}
    lot_meta: List[tuple] = []
    lot_quality: List[float] = []

    def get_lot(site: int, model: int, part: int, day: float):
        wave = int(np.floor(day / (365.25 / 2.0)))
        key = (site, model, part, wave)
        rec = lot_cache.get(key)
        if rec is None:
            cv = parts[part]["batch_scale"] * rng.uniform(0.6, 1.2)
            q = float(gamma_frailty(rng, 1, cv)[0])
            rec = (len(lot_meta), q)
            lot_cache[key] = rec
            lot_meta.append(key)
            lot_quality.append(q)
        return rec

    # ---- renewal process -------------------------------------------------
    n_pos = len(pos_part)
    out_server, out_slot, out_part, out_site, out_batch = [], [], [], [], []
    out_install, out_retire, out_removal, out_kind = [], [], [], []
    out_frailty, del_frailty = [], []
    cur_install = server_install[pos_server].copy()
    alive = cur_install < horizon
    batch = np.full(n_pos, -1, np.int64)
    frailty = np.ones(n_pos)
    for i in np.flatnonzero(alive):
        batch[i], frailty[i] = get_lot(int(pos_site[i]), int(pos_model[i]),
                                       int(pos_part[i]), float(cur_install[i]))

    gen = 0
    while np.any(alive) and gen <= c.max_renewals:
        idx = np.flatnonzero(alive)
        p = pos_part[idx]
        mult = base_mult[idx]
        u1 = rng.random(len(idx))
        u2 = rng.random(len(idx))
        t = np.empty(len(idx))
        for pp in np.unique(p):
            s = p == pp
            t[s] = law[int(pp)].sample(u1[s], u2[s], frailty[idx][s])
        t_days = t * 365.25 / mult
        refurb = rng.random(len(idx)) < c.refurbish_share
        t_days = np.where(refurb, t_days / c.refurbish_hazard, t_days)
        fail_day = cur_install[idx] + t_days
        dec_day = pos_retire[idx] + rng.uniform(0, 45, size=len(idx))
        is_fail = fail_day < dec_day
        event_day = np.where(is_fail, fail_day, dec_day)

        beyond = (event_day >= horizon) | ~np.isfinite(event_day)
        rec_day = np.where(beyond, np.nan, event_day)
        rec_kind = np.where(beyond, 0, np.where(is_fail, 1, 2)).astype(np.int8)

        out_server.append(pos_server[idx])
        out_slot.append(pos_slot[idx])
        out_part.append(p)
        out_site.append(pos_site[idx])
        out_batch.append(batch[idx])
        out_install.append(cur_install[idx])
        out_retire.append(pos_retire[idx])
        out_removal.append(rec_day)
        out_kind.append(rec_kind)
        out_frailty.append(frailty[idx].copy())

        # A failed part is replaced from that year's lot and keeps running; a
        # decommissioned or still-running part ends its participation.
        cont = is_fail & ~beyond
        new_alive = np.zeros(n_pos, bool)
        cont_idx = idx[cont]
        new_alive[cont_idx] = True
        cur_install[cont_idx] = fail_day[cont]
        alive = new_alive
        for i in cont_idx:
            batch[i], frailty[i] = get_lot(int(pos_site[i]), int(pos_model[i]),
                                           int(pos_part[i]), float(cur_install[i]))
        gen += 1

    server_id = np.concatenate(out_server).astype(np.int32)
    slot_idx = np.concatenate(out_slot).astype(np.int16)
    part_idx = np.concatenate(out_part).astype(np.int16)
    site_idx = np.concatenate(out_site).astype(np.int16)
    batch_idx = np.concatenate(out_batch).astype(np.int32)
    install_day = np.concatenate(out_install)
    retire_day = np.concatenate(out_retire)
    removal_day = np.concatenate(out_removal)
    removal_kind = np.concatenate(out_kind).astype(np.int8)
    frailty_all = np.concatenate(out_frailty)
    order = np.argsort(install_day, kind="stable")
    return Fleet(
        server_id=server_id[order], slot_idx=slot_idx[order],
        part_idx=part_idx[order], site_idx=site_idx[order],
        batch_idx=batch_idx[order], install_day=install_day[order],
        retire_day=retire_day[order], removal_day=removal_day[order],
        removal_kind=removal_kind[order], frailty=frailty_all[order],
        episode_id=np.arange(len(server_id), dtype=np.int32),
        server_model_idx=server_model, server_site=server_site,
        server_install=server_install, server_retire=server_retire,
        n_days=n_days, n_batches=len(lot_meta),
        site_codes=[s[0] for s in cfg.SITES],
        truth=dict(site_mult=site_mult, part_mult=part_mult,
                   model_load=model_load, lot_quality=np.asarray(lot_quality),
                   lot_site=np.asarray([k[0] for k in lot_meta], np.int64),
                   lot_model=np.asarray([k[1] for k in lot_meta], np.int64),
                   lot_part=np.asarray([k[2] for k in lot_meta], np.int64),
                   lot_wave=np.asarray([k[3] for k in lot_meta], np.int64)),
    )


# ---------------------------------------------------------------------------
def year_start(year: int) -> int:
    return int(round((year - cfg.DEFAULT.start_year) * DAYS_PER_YEAR))
