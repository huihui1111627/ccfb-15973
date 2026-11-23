"""Acceptance run: build the fleet once, fit once, run all policies.

Produces the full result bundle consumed by the web dashboard and the CLI: the
life fits, the demand forecast, the three inventory policies with their
simulated outcomes, the last-time-buy plans, and the forecast calibration check.
Everything is cached in one JSON file so the dashboard starts instantly.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config as cfg
from .estimation import (DAYS_PER_YEAR, MONTH_DAYS, coverage_report,
                         fit_lives, fit_multipliers, forecast_demand)
from .fleet import build_fleet
from .inventory import (N_HUBS, N_PARTS, N_SITES, PART_CODES, SITE_NAMES,
                        UNIT_COST,
                        build_fixed_policy, build_forecast_policy,
                        build_hist_policy, simulate)
from .lastbuy import check_against_fleet, last_buy_vector, plan_last_buys
from .records import build_records

CACHE = Path(__file__).resolve().parent.parent / "data" / "results.json"


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _monthly_actual(fleet, as_of: float, horizon: int) -> np.ndarray:
    """Observed failures per (site, part, month) after the cut-off."""
    m0 = int(np.floor(as_of / MONTH_DAYS))
    fail = (fleet.removal_kind == 1) & (fleet.removal_day >= as_of)
    k = np.floor(fleet.removal_day[fail] / MONTH_DAYS).astype(int) - m0
    ok = (k >= 0) & (k < horizon)
    site = fleet.site_idx[fail][ok].astype(np.int64)
    part = fleet.part_idx[fail][ok].astype(np.int64)
    flat = (site * N_PARTS + part) * horizon + k[ok]
    out = np.bincount(flat, minlength=N_SITES * N_PARTS * horizon)
    return out.reshape(N_SITES, N_PARTS, horizon)


def _truncate(fc, n: int):
    from .estimation import Forecast
    return Forecast(months=fc.months[:n], labels=fc.labels[:n],
                    mean=fc.mean[:, :, :n], q05=fc.q05[:, :, :n],
                    q25=fc.q25[:, :, :n], q75=fc.q75[:, :, :n],
                    q95=fc.q95[:, :, :n], dispersion=fc.dispersion,
                    installed=fc.installed[:, :, :n], age_mix=fc.age_mix[:, :n],
                    calibration=fc.calibration)


def _calibration_check(rec, lives, mult, fc, fleet) -> dict:
    """Compare the forecast to what really happened, for the first 12 months."""
    actual = _monthly_actual(fleet, rec.as_of_day, 12)
    cov = coverage_report(_truncate(fc, 12), actual)
    by_part = []
    for pi, p in enumerate(cfg.PARTS):
        pred = float(fc.mean[:, pi, :12].sum())
        act = float(actual[:, pi, :12].sum())
        if pred + act < 1:
            continue
        by_part.append(dict(part=p["code"], predicted=round(pred, 0),
                            actual=round(act, 0),
                            ratio=round(pred / max(act, 1e-9), 3)))
    by_site = []
    for si, s in enumerate(cfg.SITES):
        pred = float(fc.mean[si, :, :12].sum())
        act = float(actual[si, :, :12].sum())
        if pred + act < 1:
            continue
        by_site.append(dict(site=s[0], predicted=round(pred, 0),
                            actual=round(act, 0),
                            ratio=round(pred / max(act, 1e-9), 3)))
    # interval coverage by horizon bucket, to show it does not degrade
    by_horizon = []
    for lo, hi in ((0, 3), (3, 6), (6, 12)):
        from .estimation import Forecast
        sub = Forecast(months=fc.months[lo:hi], labels=fc.labels[lo:hi],
                       mean=fc.mean[:, :, lo:hi], q05=fc.q05[:, :, lo:hi],
                       q25=fc.q25[:, :, lo:hi], q75=fc.q75[:, :, lo:hi],
                       q95=fc.q95[:, :, lo:hi], dispersion=fc.dispersion,
                       installed=fc.installed[:, :, lo:hi],
                       age_mix=fc.age_mix[:, lo:hi],
                       calibration=fc.calibration)
        cov_i = coverage_report(sub, actual[:, :, lo:hi])
        by_horizon.append(dict(months=f"{lo}-{hi}", **cov_i))
    return dict(overall=cov, by_part=by_part, by_site=by_site,
                by_horizon=by_horizon,
                monthly_pred=fc.mean.sum(axis=(0, 1))[:12].round(1).tolist(),
                monthly_actual=actual.sum(axis=(0, 1))[:12].tolist())


def run(cfgp: cfg.Config = cfg.DEFAULT, quick: bool = False,
        sim_days: Optional[int] = None, verbose: bool = True) -> dict:
    t_start = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t_start:6.1f}s] {msg}", flush=True)

    log("building fleet")
    fleet = build_fleet(cfgp)
    log(f"fleet: {fleet.n_servers} servers, {len(fleet.server_id)} part episodes, "
        f"{fleet.n_batches} lots")
    rec = build_records(fleet, cfgp)
    log(f"records: {len(rec.episodes)} episodes before "
        f"{cfgp.as_of_year}-01-01, {(rec.episodes.end_kind == 1).sum()} failures")

    log("fitting lifetime distributions")
    lives = fit_lives(rec)
    log("fitting model / site / lot effects")
    mult = fit_multipliers(rec, lives)
    log("forecasting demand")
    fc = forecast_demand(rec, lives, mult, cfgp.forecast_horizon)

    log("checking forecast calibration")
    calib = _calibration_check(rec, lives, mult, fc, fleet)

    sim_days = sim_days or int(cfgp.sim_years * DAYS_PER_YEAR)
    hist_rate = rec.monthly_issues.mean(axis=2)
    hist_std = rec.monthly_issues.std(axis=2)
    policies = [
        build_fixed_policy(fc),
        build_hist_policy(fc, hist_rate, hist_std),
        build_forecast_policy(fc, cfgp),
    ]

    log("planning last-time buys")
    lb_plans = plan_last_buys(rec, lives, mult, fc)
    lb_check = check_against_fleet(lb_plans, fleet, rec)

    # last-time buy is a one-off decision common to all policies; it is
    # reconciled against projected tail demand so a 2-year simulation does not
    # look like a 12-year over-buy
    lb_vec = last_buy_vector(lb_plans)
    last_buy_targets = []
    lb_remaining = np.zeros(N_PARTS)
    for plan in lb_plans:
        lb_remaining[plan.part_idx] = max(
            0.0, plan.expected_demand - plan.expected_within_sim)

    # last-time-buy quantity vs service target (same model, different quantile)
    for q_target in (0.75, 0.90, 0.975):
        alt = plan_last_buys(rec, lives, mult, fc, target_quantile=q_target)
        chk = check_against_fleet(alt, fleet, rec)
        bundle_target = dict(
            target=q_target,
            units=float(sum(p.recommended_qty for p in alt)),
            cost=float(sum(p.recommended_qty * UNIT_COST[p.part_idx] for p in alt)),
            surplus=float(sum(c["surplus"] for c in chk)),
            shortfall=float(sum(c["shortfall"] for c in chk)),
            surplus_value=float(sum(c["surplus_value"] for c in chk)),
        )
        last_buy_targets.append(bundle_target)
    log("last-buy target sweep: " + ", ".join(
        f"q{t['target']:.2f}→{t['surplus']:.0f} surplus / {t['shortfall']:.0f} short"
        for t in last_buy_targets))

    # ── choose the two-echelon split and its buffer ────────────────────
    # "how much at each level" and "how much safety" are decisions, so search
    # them: run a small grid on a one-year simulation and keep the cheapest
    # configuration that still meets the service target.  The whole grid is
    # reported, so the choice can be audited rather than taken on faith.
    # the stockout target is a *constraint*: pick the cheapest configuration
    # whose network fill rate still reaches it
    service_target = 0.97
    sweep = []
    best_ours = None
    for rs in (0.2, 0.5):
        for buf in (1.0, 0.6, 0.3):
            cand = build_forecast_policy(fc, cfgp, risk_share=rs, buffer=buf)
            cand.last_buy = last_buy_vector(lb_plans)
            rr = simulate(cand, fleet, rec, cfgp, sim_days=min(sim_days, 365))
            row = rr.row()
            row["risk_share"] = rs
            row["buffer"] = buf
            row["site_units"] = float(cand.S.sum())
            row["hub_units"] = float(cand.hub_S.sum())
            row["meets_target"] = bool(rr.csl_network >= service_target)
            sweep.append(row)
            if row["meets_target"] and (best_ours is None
                                        or rr.total_cost < best_ours[0]):
                best_ours = (rr.total_cost, rs, buf, cand)
    log("site/hub split sweep: " + ", ".join(
        f"risk {r['risk_share']:.1f}/buf {r['buffer']:.1f}→{r['total_cost']:,.0f}"
        f"{'*' if r['meets_target'] else ''}" for r in sweep))
    if best_ours is None:
        # nothing met the target: keep the highest-service configuration
        idx = int(np.argmax([r["csl_network"] for r in sweep]))
        rs = sweep[idx]["risk_share"]
        buf = sweep[idx]["buffer"]
        cand = build_forecast_policy(fc, cfgp, risk_share=rs, buffer=buf)
        best_ours = (sweep[idx]["total_cost"], rs, buf, cand)
    policies[-1] = best_ours[3]
    chosen_risk, chosen_buffer = best_ours[1], best_ours[2]

    results = []
    for pol in policies:
        log(f"simulating {pol.kind} for {sim_days} days")
        pol.last_buy = lb_vec
        r = simulate(pol, fleet, rec, cfgp, sim_days=sim_days)
        # reconcile the buy: units bought vs (consumed + projected remaining)
        consumed = r.last_buy_consumed
        projected = float(np.sum(lb_remaining[lb_vec > 0]))
        r.last_buy_leftover = max(0.0, r.last_buy_units - consumed - projected)
        r.last_buy_short = max(
            0.0, (consumed + projected) - r.last_buy_units) + r.last_buy_short_units
        log(f"  {pol.kind}: total {r.total_cost:,.0f} CNY, "
            f"stockouts {r.stockout_events}, wait {r.wait_mean:.2f}d, "
            f"availability {r.availability:.4f}")
        results.append(r)

    # ---- comparison table ------------------------------------------------
    rows = [r.row() for r in results]
    for row, r in zip(rows, results):
        row["last_buy_leftover"] = round(r.last_buy_leftover, 0)
        row["last_buy_short"] = round(r.last_buy_short, 0)
        row["last_buy_units"] = round(r.last_buy_units, 0)
    best = min(rows, key=lambda d: d["total_cost"])

    # ---- per-site / per-part policy parameters (the deliverable) --------
    ours_pol = policies[-1]
    policy_params = dict(
        sites=SITE_NAMES, parts=PART_CODES, hubs=list(cfg.HUB_SITES),
        S=ours_pol.S.round(1).tolist(),
        rop=ours_pol.rop.round(1).tolist(),
        safety=(ours_pol.S - ours_pol.rop).clip(0).round(1).tolist(),
        hub_S=ours_pol.hub_S.round(1).tolist(),
        hub_rop=ours_pol.hub_rop.round(1).tolist(),
        review_days=ours_pol.review_days,
        risk_share=chosen_risk,
        buffer=chosen_buffer,
        targets={p["code"]: cfgp.criticality(p["code"]) for p in cfg.PARTS},
    )
    # the two baseline policies too, so the page can show what changed
    baseline_params = {}
    for pol in policies[:-1]:
        baseline_params[pol.kind] = dict(
            S=pol.S.round(1).tolist(), rop=pol.rop.round(1).tolist(),
            safety=(pol.S - pol.rop).clip(0).round(1).tolist(),
            review_days=pol.review_days)

    # ---- per-site / per-part detail for the forecast policy --------------
    detail = {}
    ours = results[-1]
    inv = np.asarray(ours.series["on_hand"])          # [site, part, weeks]
    hub_inv = np.asarray(ours.series["hub"])
    detail["inventory_end"] = {
        "site": inv[:, :, -1].round(1).tolist(),
        "hub": hub_inv[:, :, -1].round(1).tolist(),
        "sites": SITE_NAMES, "parts": PART_CODES,
        "hubs": list(cfg.HUB_SITES),
    }
    # last-buy reconciliation is shared, so attach it to the bundle once
    detail["last_buy_check"] = lb_check
    detail["inventory_weeks"] = {
        "weeks": ours.series["weeks"],
        "value": [round(v, 0) for v in ours.series["value"]],
        "site_total": (inv.sum(axis=1).round(1).tolist()),
    }

    # ---- life curves -----------------------------------------------------
    curves = {}
    for pi, p in enumerate(cfg.PARTS):
        curves[p["code"]] = dict(
            curve=lives[pi].report_curve(),
            shape=round(lives[pi].shape, 3),
            mean_life=round(lives[pi].mean_life, 2),
            floor_hazard=round(lives[pi].floor_hazard, 5),
            n_failures=lives[pi].n_failures,
            exposure=round(lives[pi].exposure_years, 0),
            true_shape=p["weibull_shape"], true_life=p["lifespan"],
            true_floor=p["base_failure"],
        )

    # ---- effects ---------------------------------------------------------
    effects = {
        "models": [m["code"] for m in cfg.SERVER_MODELS],
        "sites": SITE_NAMES,
        "model_mult": mult.model.round(3).tolist(),
        "site_mult": mult.site.round(3).tolist(),
        "lot_cv": round(mult.batch_cv, 3),
        "lot_count": int(len(mult.batch)),
        "lot_mult_sample": np.sort(mult.batch)[::max(1, len(mult.batch) // 200)]
        .round(3).tolist(),
    }

    # ---- forecast payload ------------------------------------------------
    forecast = {
        "months": fc.labels,
        "mean": fc.mean.round(2).tolist(),
        "q05": fc.q05.round(2).tolist(),
        "q95": fc.q95.round(2).tolist(),
        "installed": fc.installed.round(1).tolist(),
        "age_mix": fc.age_mix.round(2).tolist(),
        "dispersion": [round(float(v), 1) for v in fc.dispersion],
        "calibration": fc.calibration.round(3).tolist(),
    }

    # ---- stockout / transfer logs ----------------------------------------
    logs = {}
    for r in results:
        logs[r.kind] = dict(
            stockouts=r.events["stockouts"][:600],
            transfers=r.events["transfers"][:600],
            n_stockouts=r.stockout_events,
            n_transfers=r.transfers,
        )

    bundle = dict(
        meta=dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                  fingerprint=cfgp.fingerprint(),
                  as_of_year=cfgp.as_of_year, sim_days=sim_days,
                  sim_years=sim_days / DAYS_PER_YEAR,
                  forecast_horizon=cfgp.forecast_horizon,
                  servers=fleet.n_servers, episodes=len(fleet.server_id),
                  lots=fleet.n_batches,
                  sites=N_SITES, hubs=N_HUBS, parts=N_PARTS,
                  total_units_installed=int(sum(s[3] for s in cfg.SITES)),
                  target_criticality=cfgp.target_criticality,
                  target_low=cfgp.target_low,
                  f_ratio=cfgp.f_ratio,
                  stockout_penalty=cfgp.stockout_penalty_per_day,
                  lead_days={p["code"]: p["lead_days"] for p in cfg.PARTS},
                  unit_cost={p["code"]: p["unit_cost"] for p in cfg.PARTS},
                  downtime_weight=cfg.DOWNTIME_WEIGHT,
                  discontinued=rec.discontinued),
        site_info=cfg.party_info(),
        results=rows,
        best=best["policy"],
        last_buy=[p.row() for p in lb_plans],
        last_buy_check=lb_check,
        # what different service targets would cost, at fleet level: the
        # quantile is applied to the same model, so this shows the trade-off
        # between ending up short and scrapping a reserve
        last_buy_targets=last_buy_targets,
        forecast=forecast,
        calibration=calib,
        curves=curves,
        effects=effects,
        detail=detail,
        policy_params=policy_params,
        baseline_params=baseline_params,
        logs=logs,
        series={r.kind: r.series["value"] for r in results},
        split_sweep=sweep,
        chosen_risk_share=chosen_risk,
        chosen_buffer=chosen_buffer,
        run_seconds=round(time.time() - t_start, 1),
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(bundle), fh, ensure_ascii=False)
    log(f"wrote {CACHE}")
    return bundle


def load_or_run(run_if_missing: bool = True) -> dict:
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    if not run_if_missing:
        raise FileNotFoundError(CACHE)
    return run()


if __name__ == "__main__":
    run()
