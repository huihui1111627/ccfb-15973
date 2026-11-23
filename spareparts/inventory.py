"""Inventory policies and the end-to-end spare-parts simulator.

Three policies are run on the same fleet, the same failure stream (common random
numbers) and the same lead times, so the comparison is fair:

``fixed``
    Every site holds a fixed quota per part (a share of installed volume),
    topped up on a fixed cycle from the vendor.  Nothing is shared.
``hist``
    "Order the historical average": order-up-to = mean demand over lead time +
    review, plus a normal-approximation safety factor.  Nothing is shared.
``ours``
    Two-echelon network: vendor -> regional hub -> site forward stock.  Site
    levels come from the forecast *distribution* of lead-time-plus-review
    demand, so they follow the age of the installed base and known-bad lots.
    On a shortfall a site pulls from its region hub; if that is empty it tries a
    neighbouring hub, then a lateral transfer from a sister site.  Hub levels
    are sized by a newsvendor rule at the cheaper central holding rate; the
    site/hub risk split is chosen by simulation among a few candidates.

Mechanics
---------
The failure process is a renewal process keyed on *physical slot*: when a part
fails the next part in its slot is drawn from the same purchase-lot chain and
only starts its clock when the spare is actually installed.  A server that has
to wait therefore (a) accumulates downtime and (b) has its next failure pushed
back by exactly that wait.  The demand *rate* is identical across policies; only
the incidence timing shifts, which is the honest effect of a stockout.

Costs: holding (capital + obsolescence) per unit-day, ordering per purchase
order, expedited freight for transfers, and a per-day cost for each server
sitting offline.  The last-time buy for discontinued parts is evaluated
separately (units bought vs demand until retirement, and shelf leftover).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from . import config as cfg
from .estimation import DAYS_PER_YEAR, MONTH_DAYS, Forecast

PART_INDEX = {p["code"]: i for i, p in enumerate(cfg.PARTS)}
N_PARTS = len(cfg.PARTS)
N_SITES = len(cfg.SITES)
UNIT_COST = np.array([p["unit_cost"] for p in cfg.PARTS], float)
LEAD_DAYS = np.array([p["lead_days"] for p in cfg.PARTS], float)
CATEGORY = [p["category"] for p in cfg.PARTS]
PART_CODES = [p["code"] for p in cfg.PARTS]
SITE_NAMES = [s[0] for s in cfg.SITES]
HUB_LIST = list(cfg.HUB_SITES)
HUB_NAMES = [cfg.HUB_NAMES[h] for h in HUB_LIST]
N_HUBS = len(HUB_LIST)
HUB_OF_SITE = np.array([HUB_LIST.index(s[2]) for s in cfg.SITES])
SITE_TO_HUB_DAYS = np.array([cfg.HUB_TO_SITE_DAYS[s[0]] for s in cfg.SITES])
SHELF_LIFE = np.array([p["shelf_life"] or 1e12 for p in cfg.PARTS], float)

HUB_TRANSIT = np.full((N_HUBS, N_HUBS), np.inf)
for h, nbrs in cfg.HUB_NEIGHBOURS.items():
    for nh, hours in nbrs:
        i, j = HUB_LIST.index(h), HUB_LIST.index(nh)
        d = max(1.0, hours / 24.0)
        HUB_TRANSIT[i, j] = min(HUB_TRANSIT[i, j], d)
        HUB_TRANSIT[j, i] = min(HUB_TRANSIT[j, i], d)
SITE_TO_SITE_DAYS = np.full((N_SITES, N_SITES), np.inf)
for i in range(N_SITES):
    for j in range(N_SITES):
        if i == j:
            continue
        if HUB_OF_SITE[i] == HUB_OF_SITE[j]:
            SITE_TO_SITE_DAYS[i, j] = max(1.0, SITE_TO_HUB_DAYS[i]
                                          + SITE_TO_HUB_DAYS[j])
        else:
            SITE_TO_SITE_DAYS[i, j] = HUB_TRANSIT[HUB_OF_SITE[i],
                                                  HUB_OF_SITE[j]] + 2.0

COMPAT: List[np.ndarray] = [
    np.array([PART_INDEX[c] for c in cfg.compatible(p["code"])], np.int64)
    for p in cfg.PARTS]
CSL_TARGET = np.array([cfg.DEFAULT.criticality(p["code"]) for p in cfg.PARTS])
DOWNTIME_WEIGHT = np.array([cfg.DOWNTIME_WEIGHT[p["category"]] for p in cfg.PARTS])


# ---------------------------------------------------------------------------
# Policy objects and builders
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    kind: str
    label: str
    S: np.ndarray                    # [site, part] order-up-to
    rop: np.ndarray                  # [site, part] reorder point
    hub_S: np.ndarray                # [hub, part]
    hub_rop: np.ndarray              # [hub, part]
    review_days: int = 7
    use_hub: bool = False
    use_transfers: bool = False
    last_buy: Optional[np.ndarray] = None
    target: np.ndarray = field(default_factory=lambda: CSL_TARGET.copy())
    meta: dict = field(default_factory=dict)

    def total_units(self) -> float:
        n = self.S.sum() + self.hub_S.sum()
        if self.last_buy is not None:
            n += self.last_buy.sum()
        return float(n)


def _nb_ppf(mu, r, q) -> np.ndarray:
    mu = np.maximum(np.asarray(mu, float), 0.0)
    out = np.where(mu > 1e-9,
                   stats.nbinom.ppf(q, np.maximum(r, 1e-6),
                                    np.maximum(r, 1e-6)
                                    / (np.maximum(r, 1e-6) + mu)), 0.0)
    out = np.where(np.isfinite(out), out, mu)
    return np.ceil(np.maximum(out, 0.0))


def build_forecast_policy(fc: Forecast, cfgp: cfg.Config, risk_share: float = 0.25,
                          horizon: int = 12, review: int = 7,
                          buffer: float = 1.0) -> Policy:
    """Two-echelon (R,S) sizing.

    A site is a short lead time from its hub, so it only needs to cover
    hub->site transit plus one review period; the hub carries the vendor lead
    time for all its sites.  Holding risk centrally is cheaper (``central_holding
    _factor``) and pools the sites' independent demand, which is the whole point
    of the structure.  ``risk_share`` moves residual risk to the hub; a site's
    own level is sized for ``1 - (1-target)*(1-risk_share)``.
    """
    S = np.zeros((N_SITES, N_PARTS))
    rop = np.zeros((N_SITES, N_PARTS))
    hub_S = np.zeros((N_HUBS, N_PARTS))
    hub_rop = np.zeros((N_HUBS, N_PARTS))
    r_small, r_large = float(fc.dispersion[0]), float(fc.dispersion[1])
    for pi in range(N_PARTS):
        mu_d = fc.mean[:, pi, :horizon].mean(axis=1) / MONTH_DAYS   # per day
        r = r_small if mu_d.mean() * 30 < 20 else r_large
        mu_site = mu_d * (SITE_TO_HUB_DAYS + review)
        q_site = 1.0 - (1.0 - CSL_TARGET[pi]) * (1.0 - risk_share)
        # ``buffer`` scales the *safety* part of the level, not the cycle stock;
        # the optimiser uses it to buy exactly as much protection as the
        # service target needs and no more
        S[:, pi] = np.ceil(mu_site + buffer * (_nb_ppf(mu_site, r, q_site)
                                               - mu_site))
        rop[:, pi] = np.ceil(np.maximum(0.0, mu_site * 0.55 + buffer * (
            _nb_ppf(mu_site * 0.55, r, q_site) - mu_site * 0.55)))
        for h in range(N_HUBS):
            sites = np.flatnonzero(HUB_OF_SITE == h)
            # hub covers the vendor lead time for the whole region; sites pull
            # on a weekly cycle, which is cheap enough to absorb locally
            mu_h = mu_d[sites].sum() * (LEAD_DAYS[pi] + review)
            hub_S[h, pi] = np.ceil(mu_h + buffer * (_nb_ppf(
                mu_h, r, CSL_TARGET[pi]) - mu_h))
            hub_rop[h, pi] = np.ceil(np.maximum(0.0, mu_h * 0.55 + buffer * (
                _nb_ppf(mu_h * 0.55, r, CSL_TARGET[pi]) - mu_h * 0.55)))
    return Policy(kind="ours", label="两级网络+调拨", S=S, rop=rop, hub_S=hub_S,
                  hub_rop=hub_rop, review_days=review, use_hub=True,
                  use_transfers=True,
                  meta=dict(risk_share=risk_share, buffer=buffer))


def build_hist_policy(fc: Forecast, hist_rate: np.ndarray, hist_std: np.ndarray,
                      review: int = 30) -> Policy:
    """Order the historical average, the way a spreadsheet would.

    Reorder point = mean demand over vendor lead time + a textbook normal
    safety factor built from the *historical monthly standard deviation*.  That
    is a defensible policy -- it is what most teams do -- but it uses a flat
    average and a flat sigma, so it cannot see the installed base ageing, known
    bad lots, or the site-to-site differences the forecast does.  The comparison
    is therefore about forecast information, not about a strawman with no
    safety stock at all.
    """
    S = np.zeros((N_SITES, N_PARTS))
    rop = np.zeros((N_SITES, N_PARTS))
    for pi, p in enumerate(cfg.PARTS):
        mu_d = hist_rate[:, pi] / MONTH_DAYS
        sd_d = np.maximum(hist_std[:, pi], 0.0) / MONTH_DAYS
        mu_L = mu_d * LEAD_DAYS[pi]
        sd_L = sd_d * np.sqrt(LEAD_DAYS[pi])
        mu_LR = mu_d * (LEAD_DAYS[pi] + review)
        sd_LR = sd_d * np.sqrt(LEAD_DAYS[pi] + review)
        z = stats.norm.ppf(CSL_TARGET[pi])
        rop[:, pi] = np.ceil(np.maximum(0.0, mu_L + z * sd_L))
        S[:, pi] = np.ceil(np.maximum(0.0, mu_LR + z * sd_LR))
    return Policy(kind="hist", label="历史均值补货", S=S, rop=rop,
                  hub_S=np.zeros((N_HUBS, N_PARTS)),
                  hub_rop=np.zeros((N_HUBS, N_PARTS)), review_days=review)


def build_fixed_policy(fc: Forecast, review: int = 14) -> Policy:
    """The same fixed quota at every site, one number per part.

    Field practice: a single quota set centrally from peak site demand and
    applied to every machine room.  It over-stocks the small sites and starves
    the large ones -- the exact failure mode the user describes.
    """
    S = np.zeros((N_SITES, N_PARTS))
    for pi in range(N_PARTS):
        mu_d = fc.mean[:, pi, :12].mean(axis=1) / MONTH_DAYS
        peak = float(np.percentile(mu_d, 90)) * 60.0
        S[:, pi] = max(2.0, np.ceil(peak))
    rop = np.maximum(1.0, np.round(S * 0.5))
    return Policy(kind="fixed", label="固定数量备件", S=S, rop=rop,
                  hub_S=np.zeros((N_HUBS, N_PARTS)),
                  hub_rop=np.zeros((N_HUBS, N_PARTS)), review_days=review,
                  meta=dict(quota=float(S[0].sum())))


def installed_counts(fleet, as_of: float) -> np.ndarray:
    out = np.zeros((N_SITES, N_PARTS))
    alive = (fleet.install_day <= as_of) & (fleet.retire_day > as_of)
    alive &= np.isnan(fleet.removal_day) | (fleet.removal_day > as_of)
    np.add.at(out, (fleet.site_idx[alive], fleet.part_idx[alive]), 1)
    return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    kind: str
    label: str
    params: dict
    holding_cost: float
    ordering_cost: float
    transfer_cost: float
    last_buy_cost: float
    shortage_cost: float
    total_cost: float
    opening_capital: float
    mean_inventory_value: float
    purchases: int
    order_units: float
    demand_units: float
    stockout_events: int
    unfilled_events: int
    transfers: int
    wait_mean: float
    wait_p95: float
    server_days_offline: float
    availability: float
    csl_site: float
    csl_network: float
    last_buy_units: float
    last_buy_leftover: float
    last_buy_short: float
    last_buy_carry: float
    operating_cost: float
    series: dict
    events: dict
    last_buy_consumed: float = 0.0
    last_buy_short_units: float = 0.0

    def row(self) -> dict:
        return dict(policy=self.label, kind=self.kind,
                    total_cost=round(self.total_cost, 0),
                    operating_cost=round(self.operating_cost, 0),
                    last_buy_carry=round(self.last_buy_carry, 0),
                    holding=round(self.holding_cost, 0),
                    ordering=round(self.ordering_cost, 0),
                    transfer=round(self.transfer_cost, 0),
                    last_buy=round(self.last_buy_cost, 0),
                    shortage=round(self.shortage_cost, 0),
                    stockout_events=self.stockout_events,
                    unfilled=self.unfilled_events,
                    wait_mean=round(self.wait_mean, 2),
                    wait_p95=round(self.wait_p95, 1),
                    availability=round(self.availability, 4),
                    csl_site=round(self.csl_site, 4),
                    csl_network=round(self.csl_network, 4),
                    transfers=self.transfers,
                    purchases=self.purchases,
                    mean_inv_value=round(self.mean_inventory_value, 0),
                    last_buy_leftover=round(self.last_buy_leftover, 0),
                    last_buy_short=round(self.last_buy_short, 0))


_CHAIN_CACHE = {}


def _slot_chains(fleet):
    """Episode chains per physical slot, sorted by install date (cached).

    The lexsort over 1M episodes costs ~1 s; the acceptance run calls this once
    per policy, so caching it keeps the nine simulations cheap.
    """
    key = id(fleet)
    hit = _CHAIN_CACHE.get(key)
    if hit is not None:
        return hit[0]
    order = np.lexsort((fleet.install_day, fleet.slot_idx, fleet.server_id))
    skey = fleet.server_id[order].astype(np.int64) * 1000 \
        + fleet.slot_idx[order].astype(np.int64)
    starts = np.searchsorted(skey, skey, side="left")
    ends = np.searchsorted(skey, skey, side="right")
    out = (order, skey, starts, ends)
    if len(_CHAIN_CACHE) > 3:
        _CHAIN_CACHE.clear()
    # keep a reference to the fleet so its id() cannot be reused by a new object
    _CHAIN_CACHE[key] = (out, fleet)
    return out


def simulate(policy: Policy, fleet, rec, cfgp: cfg.Config, sim_days: int = 1095,
             seed: int = 11, log_limit: int = 3000) -> SimResult:
    t0 = rec.as_of_day
    order, skey, starts, ends = _slot_chains(fleet)
    srv = fleet.server_id[order]
    part_of = fleet.part_idx[order]
    site_of = fleet.site_idx[order]
    install = fleet.install_day[order]
    removal = fleet.removal_day[order]
    kind_of = fleet.removal_kind[order]
    retire = fleet.retire_day[order]

    # one "slot task" per physical slot, alive at t0
    live = np.flatnonzero((install <= t0)
                          & (np.isnan(removal) | (removal > t0))
                          & (retire > t0))
    n_live = len(live)
    # Vectorised: for each live slot, find the index (within its own episode
    # list) of the episode in service at t0.
    ep_ids = [order[starts[i]:ends[i]] for i in live]
    lens = np.fromiter((len(x) for x in ep_ids), np.int64, count=n_live)
    flat_ids = np.concatenate(ep_ids) if len(ep_ids) else np.empty(0, np.int64)
    in_service = (install[flat_ids] <= t0) & (
        np.isnan(removal[flat_ids]) | (removal[flat_ids] > t0))
    owner = np.repeat(np.arange(n_live), lens)
    offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64) \
        if n_live else np.empty(0, np.int64)
    ptr = np.full(n_live, -1, np.int64)
    hit = np.flatnonzero(in_service)
    if len(hit):
        # first in-service episode per slot wins (episodes are install-sorted)
        first = np.ones(len(hit), bool)
        first[1:] = owner[hit][1:] != owner[hit][:-1]
        sel = hit[first]
        ptr[owner[sel]] = sel - offsets[owner[sel]]

    # schedule chain: day index -> list of slot tasks failing that day
    fail_at: Dict[int, List[int]] = {}
    for j in range(n_live):
        e = ep_ids[j][ptr[j]] if ptr[j] >= 0 else None
        if e is not None and kind_of[e] == 1:
            d = int(np.floor(removal[e] - t0))
            if 0 <= d < sim_days:
                fail_at.setdefault(d, []).append(j)

    on_hand = np.array(policy.S, float, copy=True)
    hub_stock = np.array(policy.hub_S, float, copy=True)
    opening_value = float(np.sum(on_hand * UNIT_COST)
                          + np.sum(hub_stock * UNIT_COST))
    lb_stock = np.zeros(N_PARTS)
    if policy.last_buy is not None and np.sum(policy.last_buy) > 0:
        # The last-time buy is a one-off strategic reserve, not policy stock.
        # It is held as a single central pool and issued before normal stock;
        # its carrying cost is reported separately so it does not drown the
        # operating comparison (it is the same purchase under every policy).
        q = np.asarray(policy.last_buy, float)
        lb_stock = q.copy()
        opening_value += float(np.sum(q * UNIT_COST))

    hub_pull: Dict[int, List[Tuple[int, int, int]]] = {}    # hub -> site, routine
    vendor: Dict[int, List[Tuple[int, int, int]]] = {}      # to site
    vendor_hub: Dict[int, List[Tuple[int, int, int]]] = {}  # to hub
    transit: Dict[int, List[Tuple[int, int]]] = {}          # (site, part)
    on_order = np.zeros((N_SITES, N_PARTS))
    on_order_hub = np.zeros((N_HUBS, N_PARTS))

    backorders: List[dict] = []
    ev_stockouts: List[dict] = []
    ev_transfers: List[dict] = []

    acc = dict(holding=0.0, ordering=0.0, transfer=0.0, shortage=0.0,
               purchases=0, order_units=0.0, repl_order_units=0.0,
               stockouts=0, local_short=0,
               unfilled=0, transfers=0, waits=[], offline=0.0, demand=0.0,
               lb_demand=np.zeros(N_PARTS), lb_filled=np.zeros(N_PARTS),
               lb_consumed=np.zeros(N_PARTS), lb_carry=0.0, lb_used=0.0)
    h_rate = cfgp.f_ratio / DAYS_PER_YEAR
    hub_rate = cfgp.f_ratio * cfgp.central_holding_factor / DAYS_PER_YEAR
    order_cost = 180.0
    # vendor order arrival is rounded to the next Monday truck/receiving slot
    def receive_day(d: int) -> int:
        return int(d + ((7 - d % 7) % 7))
    # expedited courier between sites/hubs: fixed handling + per-day freight,
    # roughly independent of part value
    expedite = 55.0
    freight_per_day = 45.0

    def take_site(si, pi):
        for c in COMPAT[pi]:
            if on_hand[si, c] >= 1:
                on_hand[si, c] -= 1
                return int(c)
        return -1

    def take_lb(pi):
        for c in COMPAT[pi]:
            if lb_stock[c] >= 1:
                lb_stock[c] -= 1
                return int(c)
        return -1

    def take_hub(h, pi):
        for c in COMPAT[pi]:
            if hub_stock[h, c] >= 1:
                hub_stock[h, c] -= 1
                return int(c)
        return -1

    def source(si, pi, day, allow_net=True):
        """Return (code, arrival_day_or_None).  Arrival None = immediate."""
        c = take_site(si, pi)
        if c >= 0:
            return c, day
        c = take_lb(pi)
        if c >= 0:
            return c, day
        if not allow_net:
            return -1, None
        if policy.use_transfers:
            acc["local_short"] += 1
        hub = HUB_OF_SITE[si]
        c = take_hub(hub, pi)
        if c >= 0:
            d = day + int(np.ceil(SITE_TO_HUB_DAYS[si]))
            transit.setdefault(d, []).append((si, c))
            acc["transfer"] += expedite + freight_per_day * SITE_TO_HUB_DAYS[si]
            acc["transfers"] += 1
            ev_transfers.append(dict(day=int(day), part=PART_CODES[c], qty=1,
                                     kind="中心仓", source=HUB_NAMES[hub],
                                     target=SITE_NAMES[si],
                                     transit=float(SITE_TO_HUB_DAYS[si])))
            return c, d
        best = None
        for h2 in range(N_HUBS):
            if h2 == hub or not np.isfinite(HUB_TRANSIT[hub, h2]):
                continue
            for c2 in COMPAT[pi]:
                if hub_stock[h2, c2] >= 1:
                    best = (h2, c2)
                    break
            if best:
                break
        if best:
            h2, c2 = best
            hub_stock[h2, c2] -= 1
            delay = HUB_TRANSIT[hub, h2] + SITE_TO_HUB_DAYS[si]
            d = day + int(np.ceil(delay))
            transit.setdefault(d, []).append((si, c2))
            acc["transfer"] += expedite + freight_per_day * delay
            acc["transfers"] += 1
            ev_transfers.append(dict(day=int(day), part=PART_CODES[c2], qty=1,
                                     kind="邻区中心仓", source=HUB_NAMES[h2],
                                     target=SITE_NAMES[si], transit=float(delay)))
            return c2, d
        for s2 in range(N_SITES):
            if s2 == si or HUB_OF_SITE[s2] != hub:
                continue
            for c2 in COMPAT[pi]:
                if on_hand[s2, c2] >= 2:
                    on_hand[s2, c2] -= 1
                    delay = SITE_TO_SITE_DAYS[si, s2]
                    d = day + int(np.ceil(delay))
                    transit.setdefault(d, []).append((si, c2))
                    acc["transfer"] += expedite + freight_per_day * delay
                    acc["transfers"] += 1
                    ev_transfers.append(dict(day=int(day), part=PART_CODES[c2],
                                             qty=1, kind="横向调拨",
                                             source=SITE_NAMES[s2],
                                             target=SITE_NAMES[si],
                                             transit=float(delay)))
                    return c2, d
        return -1, None

    def advance(j, day):
        ptr[j] += 1
        if ptr[j] >= len(ep_ids[j]):
            return
        e = ep_ids[j][ptr[j]]
        if kind_of[e] != 1 or np.isnan(removal[e]):
            return
        # the replacement clock starts at the actual install day, so a wait
        # pushes this failure back one-for-one
        orig_install = install[e]
        shift = (day + t0) - orig_install
        nf = removal[e] + shift
        d = int(np.floor(nf - t0))
        if 0 <= d < sim_days:
            fail_at.setdefault(d, []).append(j)

    n_bins = sim_days // 7 + 1
    onhand_series = np.zeros((N_SITES, N_PARTS, n_bins))
    hub_series = np.zeros((N_HUBS, N_PARTS, n_bins))
    value_series = np.zeros(n_bins)

    for day in range(sim_days):
        for si, pi, qty in vendor.pop(day, []):
            on_hand[si, pi] += qty
            on_order[si, pi] -= qty
        for h, pi, qty in vendor_hub.pop(day, []):
            hub_stock[h, pi] += qty
            on_order_hub[h, pi] -= qty
        for si, pi, qty in hub_pull.pop(day, []):
            h = HUB_OF_SITE[si]
            take = min(qty, int(hub_stock[h, pi]))
            hub_stock[h, pi] -= take
            got = take
            if got > 0:
                on_hand[si, pi] += got
                # routine truck freight, ~2% of value per unit
                acc["transfer"] += 0.02 * UNIT_COST[pi] * got
            short = qty - take
            if short > 0:
                # hub is out: the site waits for the hub's own vendor order,
                # not for a second vendor lead time
                acc["transfer"] += 0.02 * UNIT_COST[pi] * short
                on_hand[si, pi] += 0  # stays on order
                acc["hub_backlog"] = acc.get("hub_backlog", 0.0) + short
            on_order[si, pi] -= qty
        for si, c in transit.pop(day, []):
            on_hand[si, c] += 1
            pending = [b for b in backorders if b["site"] == si
                       and b["arrival"] == day and b["done"] is False]
            if pending:
                b = pending[0]
                b["done"] = True
                waited = float(day - b["since"])
                acc["waits"].append(waited)
                acc["offline"] += waited * DOWNTIME_WEIGHT[b["part"]]

        for j in fail_at.pop(day, []):
            e = ep_ids[j][ptr[j]]
            pi, si = int(part_of[e]), int(site_of[e])
            acc["demand"] += 1
            lb_pi = pi if (policy.last_buy is not None
                           and policy.last_buy[pi] > 0) else -1
            if lb_pi >= 0:
                acc["lb_demand"][lb_pi] += 1
            c, arrival = source(si, pi, day)
            if arrival is None:
                acc["unfilled"] += 1
                acc["stockouts"] += 1
                backorders.append(dict(site=si, part=pi, since=day, j=j,
                                       arrival=None, done=False))
                ev_stockouts.append(dict(day=int(day), site=SITE_NAMES[si],
                                         part=PART_CODES[pi],
                                         server=int(srv[e]), kind="缺货未满足"))
            elif arrival == day:
                if lb_pi >= 0:
                    acc["lb_filled"][lb_pi] += 1
                    acc["lb_consumed"][lb_pi] += 1
                advance(j, day)
            else:
                backorders.append(dict(site=si, part=pi, since=day, j=j,
                                       arrival=arrival, done=False))

        for b in list(backorders):
            if b["done"]:
                backorders.remove(b)
                continue
            if b["arrival"] is not None:
                continue
            c = take_site(b["site"], b["part"])
            if c >= 0:
                b["done"] = True
                backorders.remove(b)
                waited = float(day - b["since"])
                acc["waits"].append(waited)
                acc["offline"] += waited * DOWNTIME_WEIGHT[b["part"]]
                advance(b["j"], day)

        if day % policy.review_days == 0:
            pos_before = on_order.copy()
            for si in range(N_SITES):
                for pi in range(N_PARTS):
                    if policy.last_buy is not None and policy.last_buy[pi] > 0:
                        # end-of-life part: no vendor channel any more, the
                        # strategic reserve is all there is
                        continue
                    pos = on_hand[si, pi] + on_order[si, pi]
                    if pos <= policy.rop[si, pi] and policy.S[si, pi] > 0:
                        qty = int(np.ceil(policy.S[si, pi] - pos))
                        if qty <= 0:
                            continue
                        acc["order_units"] += qty
                        if policy.use_hub:
                            # regular replenishment: a scheduled hub->site
                            # truck run, charged the flat replenishment freight
                            h = HUB_OF_SITE[si]
                            on_order[si, pi] += qty
                            d = day + int(np.ceil(SITE_TO_HUB_DAYS[si]))
                            hub_pull.setdefault(d, []).append((si, pi, qty))
                            acc["repl_order_units"] += qty
                        else:
                            d = day + int(np.ceil(LEAD_DAYS[pi]))
                            vendor.setdefault(receive_day(d), []).append((si, pi, qty))
                            on_order[si, pi] += qty
            hub_before = on_order_hub.copy()
            if policy.use_hub:
                for h in range(N_HUBS):
                    for pi in range(N_PARTS):
                        pos = hub_stock[h, pi] + on_order_hub[h, pi]
                        if pos <= policy.hub_rop[h, pi] and policy.hub_S[h, pi] > 0:
                            qty = int(np.ceil(policy.hub_S[h, pi] - pos))
                            if qty <= 0:
                                continue
                            acc["order_units"] += qty
                            d = day + int(np.ceil(LEAD_DAYS[pi]))
                            vendor_hub.setdefault(receive_day(d), []).append((h, pi, qty))
                            on_order_hub[h, pi] += qty

        # purchase orders are charged per shipment, not per SKU line
        if day % policy.review_days == 0:
            po_sites = np.any(on_order > pos_before + 1e-9, axis=1)
            acc["purchases"] += int(po_sites.sum())
            acc["ordering"] += int(po_sites.sum()) * order_cost
            if policy.use_hub:
                po_hubs = np.any(on_order_hub > hub_before + 1e-9, axis=1)
                acc["purchases"] += int(po_hubs.sum())
                acc["ordering"] += int(po_hubs.sum()) * order_cost
        acc["holding"] += float(np.sum(on_hand * UNIT_COST)) * h_rate
        acc["holding"] += float(np.sum(hub_stock * UNIT_COST)) * hub_rate
        acc["lb_carry"] += float(np.sum(lb_stock * UNIT_COST)) * h_rate * 0.5
        open_bo = 0.0
        for b in backorders:
            if not b["done"]:
                open_bo += DOWNTIME_WEIGHT[b["part"]]
        acc["shortage"] += open_bo * cfgp.stockout_penalty_per_day
        acc["offline"] += open_bo
        if day % 7 == 0:
            k = day // 7
            onhand_series[:, :, k] = on_hand
            hub_series[:, :, k] = hub_stock
            value_series[k] = float(np.sum(on_hand * UNIT_COST)
                                    + np.sum(hub_stock * UNIT_COST))

    # anything still waiting at the horizon counts as a wait to the end
    for b in backorders:
        if not b["done"]:
            waited = float(sim_days - b["since"])
            acc["waits"].append(waited)
            acc["offline"] += waited * DOWNTIME_WEIGHT[b["part"]]

    waits = np.asarray(acc["waits"]) if acc["waits"] else np.zeros(1)
    demand = max(acc["demand"], 1.0)
    last_buy_cost = 0.0
    last_buy_units = 0.0
    last_buy_left = 0.0
    last_buy_short = 0.0
    lb_carry = float(acc["lb_carry"])
    if policy.last_buy is not None:
        q = np.asarray(policy.last_buy, float)
        last_buy_units = float(q.sum())
        last_buy_cost = float(np.sum(q * UNIT_COST))
        # units bought but never issued, and demand never met, per part
        for pi in range(N_PARTS):
            if q[pi] <= 0:
                continue
            last_buy_left += float(lb_stock[pi])
            last_buy_short += max(0.0, float(acc["lb_demand"][pi]
                                             - acc["lb_filled"][pi]))
    # operating cost = running the spares network.  The last-time buy is a
    # shared, one-off procurement decision: its half-rate carrying cost is its
    # own line, not folded into the comparison, otherwise a constant multi-
    # thousand unit reserve would dwarf every difference between policies.
    operating = (acc["holding"] + acc["ordering"] + acc["transfer"]
                 + acc["shortage"])
    # the comparison is on operating cost; the one-off reserve is the same
    # purchase under every policy and is shown on its own line
    total = operating
    n_servers = int(np.sum((fleet.server_install <= t0)
                           & (fleet.server_retire > t0)))
    server_days = max(float(n_servers) * sim_days, 1.0)
    mean_value = float(value_series[:max(1, sim_days // 7)].mean())
    return SimResult(
        kind=policy.kind, label=policy.label, params=policy.meta,
        holding_cost=acc["holding"], ordering_cost=acc["ordering"],
        transfer_cost=acc["transfer"], last_buy_cost=last_buy_cost,
        last_buy_carry=lb_carry, operating_cost=operating,
        shortage_cost=acc["shortage"], total_cost=total,
        opening_capital=opening_value, mean_inventory_value=mean_value,
        purchases=int(acc["purchases"]), order_units=float(acc["order_units"]),
        demand_units=float(demand), stockout_events=int(acc["stockouts"]),
        unfilled_events=int(acc["unfilled"]),
        transfers=int(acc["transfers"]), wait_mean=float(waits.mean()),
        wait_p95=float(np.percentile(waits, 95)),
        server_days_offline=float(acc["offline"]),
        availability=float(1.0 - acc["offline"] / server_days),
        csl_site=float(1.0 - max(acc["stockouts"], acc["local_short"]) / demand),
        csl_network=float(1.0 - acc["unfilled"] / demand),
        last_buy_units=last_buy_units, last_buy_leftover=last_buy_left,
        last_buy_short=last_buy_short,
        last_buy_consumed=float(acc["lb_consumed"].sum()),
        last_buy_short_units=float(np.sum(np.maximum(
            acc["lb_demand"] - acc["lb_filled"], 0.0))),
        series=dict(weeks=np.arange(n_bins).tolist(),
                    value=value_series.tolist(),
                    on_hand=onhand_series.tolist(),
                    hub=hub_series.tolist()),
        events=dict(stockouts=ev_stockouts[:log_limit],
                    transfers=ev_transfers[:log_limit]))
