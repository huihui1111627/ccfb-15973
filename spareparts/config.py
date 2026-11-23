"""Fleet topology, part catalogue, and cost assumptions.

The numbers here are deliberately plausible-but-not-real; they are the knobs the
user would replace with data from their asset system.  Everything downstream
(forecasting, policy optimisation, the simulator, the dashboard) reads from a
single :class:`Config` so a scenario is fully reproducible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parts.  Costs in CNY, lead times in days, lifespans in years.
# ``downgrade_to`` lists older compatible part codes (new HDD fits older tray).
# ``substitute`` lists codes with identical specs where the vendor may ship
# either one (same cost band); it is mutual by construction.
# ---------------------------------------------------------------------------
PARTS = [
    dict(code="HDD-1T-SATA", category="HDD",   unit_cost=650,   lead_days=30,
         lifespan=12.0, weibull_shape=2.6, base_failure=0.01, batch_scale=0.55,
         shelf_life=None, downgrade_to=[]),
    dict(code="HDD-2T-SATA", category="HDD",   unit_cost=950,   lead_days=30,
         lifespan=12.0, weibull_shape=2.6, base_failure=0.012, batch_scale=0.55,
         shelf_life=None, downgrade_to=["HDD-1T-SATA"]),
    dict(code="HDD-4T-SAS",  category="HDD",   unit_cost=1750,  lead_days=45,
         lifespan=11.0, weibull_shape=2.4, base_failure=0.016, batch_scale=0.45,
         shelf_life=None, downgrade_to=["HDD-2T-SATA", "HDD-1T-SATA"]),
    dict(code="SSD-480G",    category="SSD",   unit_cost=880,   lead_days=40,
         lifespan=20.0, weibull_shape=2.8, base_failure=0.004, batch_scale=0.35,
         shelf_life=None, downgrade_to=[]),
    dict(code="SSD-960G",    category="SSD",   unit_cost=1650,  lead_days=45,
         lifespan=20.0, weibull_shape=2.8, base_failure=0.005, batch_scale=0.35,
         shelf_life=None, downgrade_to=["SSD-480G"]),
    dict(code="MEM-16G-DDR4", category="MEM",  unit_cost=700,   lead_days=35,
         lifespan=30.0, weibull_shape=3.2, base_failure=0.0025, batch_scale=0.60,
         shelf_life=None, downgrade_to=[], substitute=["MEM-16G-DDR4-ALT"]),
    dict(code="MEM-16G-DDR4-ALT", category="MEM", unit_cost=720, lead_days=32,
         lifespan=30.0, weibull_shape=3.2, base_failure=0.0025, batch_scale=0.60,
         shelf_life=None, downgrade_to=[], substitute=["MEM-16G-DDR4"]),
    dict(code="MEM-32G-DDR4", category="MEM",  unit_cost=1400,  lead_days=40,
         lifespan=30.0, weibull_shape=3.2, base_failure=0.003, batch_scale=0.60,
         shelf_life=None, downgrade_to=["MEM-16G-DDR4"],
         substitute=["MEM-32G-DDR4-ALT"]),
    dict(code="MEM-32G-DDR4-ALT", category="MEM", unit_cost=1450, lead_days=38,
         lifespan=30.0, weibull_shape=3.2, base_failure=0.003, batch_scale=0.60,
         shelf_life=None, downgrade_to=["MEM-16G-DDR4"],
         substitute=["MEM-32G-DDR4"]),
    dict(code="PSU-800W",    category="PSU",   unit_cost=1150,  lead_days=50,
         lifespan=15.0, weibull_shape=2.2, base_failure=0.011, batch_scale=0.70,
         shelf_life=6.0, downgrade_to=[]),
    dict(code="PSU-1200W",   category="PSU",   unit_cost=1750,  lead_days=55,
         lifespan=15.0, weibull_shape=2.2, base_failure=0.01, batch_scale=0.70,
         shelf_life=6.0, downgrade_to=["PSU-800W"]),
    dict(code="NIC-10G",     category="NIC",   unit_cost=520,   lead_days=25,
         lifespan=25.0, weibull_shape=3.0, base_failure=0.002, batch_scale=0.40,
         shelf_life=None, downgrade_to=[]),
    dict(code="NIC-25G",     category="NIC",   unit_cost=1450,  lead_days=35,
         lifespan=25.0, weibull_shape=3.0, base_failure=0.0025, batch_scale=0.40,
         shelf_life=None, downgrade_to=["NIC-10G"]),
    dict(code="FAN-40MM",    category="FAN",   unit_cost=85,    lead_days=20,
         lifespan=10.0, weibull_shape=2.0, base_failure=0.02, batch_scale=0.90,
         shelf_life=None, downgrade_to=[]),
    dict(code="FAN-80MM",    category="FAN",   unit_cost=120,   lead_days=20,
         lifespan=10.0, weibull_shape=2.0, base_failure=0.018, batch_scale=0.90,
         shelf_life=None, downgrade_to=["FAN-40MM"]),
]

# qty_per_server: how many of each part a given server model carries.
# rho_scale: multiplier on failure rate due to workload (GPU boxes run hotter).
SERVER_MODELS = [
    dict(code="R520-2U", generation="old",  launch_year=2009, retire_year=8,
         purchase_years=(2014, 2016), purchase_share=0.45,
         base_qty=dict(**{"HDD-1T-SATA": 8, "SSD-480G": 1, "MEM-16G-DDR4": 8,
                          "PSU-800W": 2, "NIC-10G": 1, "FAN-40MM": 4}),
         rho_scale=1.00),
    dict(code="R730-2U", generation="old",  launch_year=2011, retire_year=9,
         purchase_years=(2014, 2020), purchase_share=0.55,
         base_qty=dict(**{"HDD-2T-SATA": 8, "SSD-480G": 1, "MEM-16G-DDR4": 12,
                          "PSU-800W": 2, "NIC-10G": 1, "FAN-40MM": 4}),
         rho_scale=1.00),
    dict(code="R740-2U", generation="mid",  launch_year=2013, retire_year=7,
         purchase_years=(2014, 2024), purchase_share=0.65,
         base_qty=dict(**{"HDD-2T-SATA": 8, "SSD-960G": 1, "MEM-32G-DDR4": 12,
                          "PSU-800W": 2, "NIC-25G": 1, "FAN-40MM": 4}),
         rho_scale=1.05),
    dict(code="R750-2U", generation="mid",  launch_year=2015, retire_year=7,
         purchase_years=(2015, 2026), purchase_share=0.75,
         base_qty=dict(**{"HDD-4T-SAS": 8, "SSD-960G": 1, "MEM-32G-DDR4": 16,
                          "PSU-1200W": 2, "NIC-25G": 1, "FAN-80MM": 4}),
         rho_scale=1.10),
    dict(code="GPU-4U",  generation="new",  launch_year=2016, retire_year=5,
         purchase_years=(2016, 2027), purchase_share=0.30,
         base_qty=dict(**{"SSD-960G": 4, "MEM-32G-DDR4": 16, "PSU-1200W": 4,
                          "NIC-25G": 2, "FAN-80MM": 8}),
         rho_scale=1.55),
    dict(code="R760-2U", generation="new",  launch_year=2016, retire_year=7,
         purchase_years=(2016, 2027), purchase_share=0.50,
         base_qty=dict(**{"HDD-4T-SAS": 12, "SSD-960G": 1, "MEM-32G-DDR4-ALT": 16,
                          "PSU-1200W": 2, "NIC-25G": 1, "FAN-80MM": 4}),
         rho_scale=1.05),
]

# How much of the server's service a failed part actually takes down.  A disk in
# a RAID set degrades the array but keeps serving; a PSU in a 2N box is
# redundant; a fan subtends one of several.  Memory or a non-redundant NIC can
# take the whole box offline.  Used to weight downtime, and reported separately
# from the raw waiting time.
DOWNTIME_WEIGHT = {
    "HDD": 0.25, "SSD": 0.5, "MEM": 1.0, "PSU": 0.35, "NIC": 0.8, "FAN": 0.15,
}

# (code, province, region hub, base server count).  Big fleet in the east,
# small edge sites in the west; region hubs are 1, 2, 6, 7, 9, 10, 13, 15, 16.
SITES = [
    ("BJ-01",  "北京",  "BJ", 6800), ("BJ-02",  "北京",  "BJ", 2100),
    ("SH-01",  "上海",  "SH", 6100), ("SH-02",  "上海",  "SH", 1600),
    ("GZ-01",  "广东",  "GZ", 5800), ("GZ-02",  "广东",  "GZ", 2400),
    ("SZ-01",  "广东",  "GZ", 2600),
    ("CD-01",  "四川",  "CD", 1450), ("CD-02",  "四川",  "CD", 480),
    ("WH-01",  "湖北",  "WH", 1300), ("WH-02",  "湖北",  "WH", 400),
    ("XA-01",  "陕西",  "XA", 880),  ("XA-02",  "陕西",  "XA", 320),
    ("HZ-01",  "浙江",  "HZ", 2250), ("HZ-02",  "浙江",  "HZ", 670),
    ("NB-01",  "浙江",  "HZ", 240),  ("JN-01",  "山东",  "JN", 420),
]

# Region hub -> list of (site, truck hours).  The hub's own sites are 0h.
# Nearest-neighbour transfers between hubs are allowed with inter-hub time.
HUB_SITES: Dict[str, List[str]] = {
    "BJ": ["BJ-01", "BJ-02"],
    "SH": ["SH-01", "SH-02"],
    "GZ": ["GZ-01", "GZ-02", "SZ-01"],
    "CD": ["CD-01", "CD-02"],
    "WH": ["WH-01", "WH-02"],
    "XA": ["XA-01", "XA-02"],
    "HZ": ["HZ-01", "HZ-02", "NB-01"],
    "JN": ["JN-01"],
}

HUB_NAMES = {
    "BJ": "北京中心仓", "SH": "上海中心仓", "GZ": "广州中心仓",
    "CD": "成都中心仓", "WH": "武汉中心仓", "XA": "西安中心仓",
    "HZ": "杭州中心仓", "JN": "济南中心仓",
}

# Site -> nearest hubs for inter-hub transfers, (hours, extra cost per unit).
HUB_NEIGHBOURS: Dict[str, List[Tuple[str, float]]] = {
    "BJ": [("JN", 15), ("XA", 24), ("SH", 28), ("WH", 32)],
    "SH": [("HZ", 9), ("JN", 20), ("BJ", 28), ("GZ", 40)],
    "GZ": [("HZ", 30), ("WH", 24), ("CD", 52), ("SH", 40)],
    "CD": [("XA", 18), ("WH", 26), ("GZ", 52)],
    "WH": [("CD", 26), ("GZ", 24), ("XA", 22), ("SH", 32)],
    "XA": [("CD", 18), ("WH", 22), ("BJ", 24)],
    "HZ": [("SH", 9), ("GZ", 30)],
    "JN": [("BJ", 15), ("SH", 20)],
}

# Intra-hub (central -> edge site) transport hours.
HUB_TO_SITE_HOURS = {"BJ-02": 5, "SH-02": 4, "GZ-02": 3, "SZ-01": 4,
                     "CD-02": 7, "WH-02": 4, "XA-02": 8, "HZ-02": 3,
                     "NB-01": 5, "JN-01": 0}
# Hub cities are 1-3 truck-days from their own sites.  A same-city site is a
# few hours; a different city in the same province is a day; the far edge sites
# are two or three days.  HUB_TO_SITE_HOURS above is the local shuttle.
HUB_TO_SITE_DAYS = {
    "BJ-01": 0.5, "BJ-02": 0.75, "SH-01": 0.5, "SH-02": 0.75, "GZ-01": 0.5,
    "GZ-02": 0.75, "SZ-01": 1.25, "CD-01": 0.5, "CD-02": 1.5, "WH-01": 0.5,
    "WH-02": 1.0, "XA-01": 0.5, "XA-02": 1.75, "HZ-01": 0.5, "HZ-02": 1.0,
    "NB-01": 1.25, "JN-01": 0.5,
}

REGION_COLD = {"BJ": 1.00, "SH": 1.02, "GZ": 1.10, "CD": 0.98, "WH": 1.03,
               "XA": 0.95, "HZ": 0.99, "JN": 0.97}


@dataclass
class Config:
    seed: int = 20241124
    start_year: int = 2014
    # fleet generation horizon: the planning date is ``as_of_year``-01-01, and
    # purchases continue for ``eval_years`` beyond it so the installed base
    # keeps growing during the simulated period
    warm_years: int = 8            # history before the cut-off
    eval_years: int = 3            # years of purchases after the cut-off
    forecast_horizon: int = 24     # months of demand forecast to publish
    sim_years: int = 3             # acceptance simulator length
    as_of_year: int = 2022         # planning cut-off: data up to 2021-12-31
    # Common random numbers: same failure/life draws across policies.
    target_criticality: float = 0.95   # CSL target for critical (HDD/MEM/PSU)
    target_low: float = 0.92           # CSL target for FAN/NIC
    f_ratio: float = 0.30              # annual holding cost / unit cost
    central_holding_factor: float = 0.85
    stockout_penalty_per_day: float = 600.0   # server offline penalisation
    # Startup capex (warm-start buy) is counted separately from operating stock.
    # Ground-truth site heterogeneity (procurement quality + cooling):
    # log-normal sd of the site failure multiplier, before region effects.
    site_quality_sigma: float = 0.30
    # Replacement parts: fraction that are refurbished pulls (higher hazard).
    refurbish_share: float = 0.10
    refurbish_hazard: float = 1.8
    # Maximum number of part lifecycles generated per slot (renewals).
    max_renewals: int = 18
    # Run policy optimiser and inventory simulation unless disabled.

    @property
    def total_servers(self) -> int:
        return sum(s[3] for s in SITES)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_servers"] = self.total_servers
        return d

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True)
                            .encode()).hexdigest()[:10]

    def site_index(self) -> Dict[str, int]:
        return {s[0]: i for i, s in enumerate(SITES)}

    def criticality(self, code: str) -> float:
        cat = next(p["category"] for p in PARTS if p["code"] == code)
        return self.target_criticality if cat in ("HDD", "MEM", "PSU") else self.target_low


DEFAULT = Config()


def part_table() -> Dict[str, dict]:
    return {p["code"]: p for p in PARTS}


def _compat_matrix() -> Dict[str, List[str]]:
    """code -> list of codes that can stand in for it (including itself).

    Downgrade chains are transitive; substitutes are mutual and also inherit the
    downgrade options of the code they substitute (a 32G-ALT can be swapped for a
    32G and then for a 16G in a pinch).
    """
    by = part_table()
    out: Dict[str, List[str]] = {}

    def downgrades(code: str, seen: set) -> List[str]:
        res = []
        for c in by[code].get("downgrade_to", []):
            if c not in seen:
                seen.add(c)
                res.append(c)
                res.extend(downgrades(c, seen))
        return res

    for code in by:
        opts = [code]
        opts += downgrades(code, {code})
        for alt in by[code].get("substitute", []):
            if alt not in opts:
                opts.append(alt)
                opts += downgrades(alt, set(opts))
        # de-dup, preserve order
        seen, uniq = set(), []
        for o in opts:
            if o not in seen:
                seen.add(o)
                uniq.append(o)
        out[code] = uniq
    return out


COMPAT = _compat_matrix()


def compatible(code: str) -> List[str]:
    """Codes that may satisfy a demand for ``code`` (self first)."""
    return COMPAT.get(code, [code])


def party_info() -> dict:
    """Static identifiers exposed to the front end."""
    site_rows = []
    for code, prov, hub, n in SITES:
        site_rows.append(dict(code=code, province=prov, hub=hub,
                              servers=n,
                              hub_name=HUB_NAMES[hub]))
    return dict(
        sites=site_rows,
        hubs=[dict(code=h, name=HUB_NAMES[h],
                   sites=HUB_SITES[h]) for h in HUB_SITES],
        hub_neighbours={h: [dict(hub=n, hours=hr)
                            for n, hr in HUB_NEIGHBOURS[h]] for h in HUB_NEIGHBOURS},
        parts=[dict(p) for p in PARTS],
        server_models=[dict(m, base_qty=dict(m["base_qty"]))
                       for m in SERVER_MODELS],
        compat=COMPAT,
    )
