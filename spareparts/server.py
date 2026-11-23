"""A small standard-library HTTP server for the spare-parts dashboard.

No third-party web framework is available in this environment, so this module
ships a tiny router: static files from ``web/`` plus a handful of JSON
endpoints backed by the cached acceptance bundle.  It is deliberately
read-only -- the dashboard displays results, it does not mutate the model.
"""
from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = ROOT / "data" / "results.json"


class State:
    bundle = None
    lock = threading.Lock()


def _load():
    with State.lock:
        if State.bundle is None:
            with open(DATA, encoding="utf-8") as fh:
                State.bundle = json.load(fh)
        return State.bundle


def _filter(params):
    b = _load()
    site = params.get("site", [None])[0]
    part = params.get("part", [None])[0]
    site_idx = None
    part_idx = None
    if site and site in [s["code"] for s in b["site_info"]["sites"]]:
        site_idx = [s["code"] for s in b["site_info"]["sites"]].index(site)
    parts = [p["code"] for p in b["site_info"]["parts"]]
    if part and part in parts:
        part_idx = parts.index(part)
    return b, site_idx, part_idx


def api_overview(params):
    b = _load()
    return dict(
        meta=b["meta"], run_seconds=b.get("run_seconds"),
        results=b["results"], best=b["best"],
        calibration=b["calibration"], last_buy=b["last_buy"],
        site_info=b["site_info"], curves=b["curves"], effects=b["effects"],
        forecast=b["forecast"], last_buy_check=b.get("last_buy_check", []),
        split_sweep=b.get("split_sweep", []),
        chosen_risk_share=b.get("chosen_risk_share"),
        chosen_buffer=b.get("chosen_buffer"),
        policy_params=b.get("policy_params"),
        last_buy_targets=b.get("last_buy_targets", []),
    )


def api_forecast(params):
    b, si, pi = _filter(params)
    fc = b["forecast"]
    out = dict(months=fc["months"], labels=fc["months"])
    if si is None and pi is None:
        out["mean"] = _sum(fc["mean"], axis=(0, 1))
        out["q05"] = _sum(fc["q05"], axis=(0, 1))
        out["q95"] = _sum(fc["q95"], axis=(0, 1))
        out["scope"] = "全部机房 / 全部零件"
    elif si is not None and pi is None:
        out["mean"] = _sum(fc["mean"], axis=(1,), index=si)
        out["q05"] = _sum(fc["q05"], axis=(1,), index=si)
        out["q95"] = _sum(fc["q95"], axis=(1,), index=si)
        out["scope"] = b["site_info"]["sites"][si]["code"]
    elif si is None and pi is not None:
        out["mean"] = _sum(fc["mean"], axis=(0,), index=pi)
        out["q05"] = _sum(fc["q05"], axis=(0,), index=pi)
        out["q95"] = _sum(fc["q95"], axis=(0,), index=pi)
        out["scope"] = b["site_info"]["parts"][pi]["code"]
    else:
        out["mean"] = _column(fc["mean"], si, pi)
        out["q05"] = _column(fc["q05"], si, pi)
        out["q95"] = _column(fc["q95"], si, pi)
        out["scope"] = (b["site_info"]["sites"][si]["code"] + " / "
                        + b["site_info"]["parts"][pi]["code"])
    return out


def _sum(arr, axis, index=None):
    import numpy as _np
    a = _np.asarray(arr, float)
    if index is not None:
        a = a.take(index, axis=axis[0])
    return a.sum(axis=axis).round(2).tolist()


def _column(arr, si, pi):
    import numpy as _np
    a = _np.asarray(arr, float)
    return a[si, pi].round(2).tolist()


def api_inventory(params):
    b, si, pi = _filter(params)
    import numpy as _np
    det = b["detail"]
    inv = _np.asarray(det["inventory_end"]["site"], float)   # [site, part]
    hub = _np.asarray(det["inventory_end"]["hub"], float)
    weeks = b["detail"]["inventory_weeks"]
    out = dict(weeks=weeks["weeks"], value=weeks["value"])
    if si is not None and pi is not None:
        out["end"] = float(inv[si, pi])
        hub_code = b["site_info"]["sites"][si]["hub"]
        hub_i = det["inventory_end"]["hubs"].index(hub_code)
        out["hub_end"] = float(hub[hub_i, pi])
        out["scope"] = det["inventory_end"]["sites"][si] + " / " + det["inventory_end"]["parts"][pi]
    else:
        out["end"] = inv.round(1).tolist()
        out["hub_end"] = hub.round(1).tolist()
        out["scope"] = "全部"
    out["sites"] = det["inventory_end"]["sites"]
    out["parts"] = det["inventory_end"]["parts"]
    return out


def api_events(params):
    b = _load()
    kind = params.get("policy", ["ours"])[0]
    logs = b["logs"].get(kind, {})
    return dict(stockouts=logs.get("stockouts", []),
                transfers=logs.get("transfers", []),
                n_stockouts=logs.get("n_stockouts", 0),
                n_transfers=logs.get("n_transfers", 0))


def api_life(params):
    b = _load()
    return dict(curves=b["curves"], effects=b["effects"], target=b["meta"])


def api_lastbuy(params):
    b = _load()
    return dict(plans=b["last_buy"], meta=b["meta"])


def api_detail(params):
    b = _load()
    return b["detail"]


ROUTES: dict[str, Callable] = {
    "/api/overview": api_overview,
    "/api/forecast": api_forecast,
    "/api/inventory": api_inventory,
    "/api/events": api_events,
    "/api/life": api_life,
    "/api/lastbuy": api_lastbuy,
    "/api/detail": api_detail,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SpareParts/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ROUTES:
                data = ROUTES[path](parse_qs(parsed.query))
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            rel = path.lstrip("/") or "index.html"
            if ".." in rel:
                self._send(403, b"forbidden", "text/plain")
                return
            f = WEB / rel
            if not f.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",
                                                      "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, f.read_bytes(), ctype)
        except FileNotFoundError:
            self._send(503, b'{"error":"results not built; run python3 -m spareparts.acceptance"}',
                       "application/json")
        except Exception as exc:                     # pragma: no cover
            self._send(500, json.dumps(dict(error=str(exc))).encode(),
                       "application/json")


def serve(port: int = 8000, host: str = "127.0.0.1"):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"dashboard on http://{host}:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
