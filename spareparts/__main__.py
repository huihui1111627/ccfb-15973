"""Command line entry point.

    python3 -m spareparts run [--years 3] [--quick]
    python3 -m spareparts serve [PORT]
    python3 -m spareparts test
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spareparts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="build the acceptance results")
    run.add_argument("--years", type=float, default=3.0,
                     help="simulation horizon in years (default 3)")
    run.add_argument("--quick", action="store_true",
                     help="1-year simulation, for a fast smoke test")

    serve = sub.add_parser("serve", help="serve the dashboard")
    serve.add_argument("port", nargs="?", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")

    sub.add_parser("test", help="run the statistical acceptance tests")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        from . import acceptance
        from . import config as cfg
        days = int((1 if args.quick else args.years) * 365.25)
        acceptance.run(cfg.DEFAULT, sim_days=days)
        return 0
    if args.cmd == "serve":
        from . import server
        server.serve(args.port, args.host)
        return 0
    if args.cmd == "test":
        return subprocess.call([sys.executable, "-m", "unittest", "discover",
                                "-s", "tests", "-v"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
