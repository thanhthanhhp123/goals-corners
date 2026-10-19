"""Command line entry point: `python -m predictor <command>`."""
from __future__ import annotations

import argparse
import json
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m predictor", description="Goals & corners predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_update = sub.add_parser("update", help="download fresh data, refit, and rebuild web/data")
    p_update.add_argument("--offline", action="store_true", help="use cached data only (no download)")

    p_serve = sub.add_parser("serve", help="serve the web app with a refresh endpoint")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--refresh-hours", type=float, default=6.0, help="auto-refresh when data is older than this (0 = off)")

    p_export = sub.add_parser("export", help="write a single self-contained HTML file with the current predictions")
    p_export.add_argument("--out", default="dist/goals-corners-predictor.html")
    p_export.add_argument("--artifact", action="store_true", help="write a page fragment (no <html>/<head>) for hosts that add their own")

    p_bt = sub.add_parser("backtest", help="print out-of-sample accuracy for a date range")
    p_bt.add_argument("--start", default="2026-07-01")
    p_bt.add_argument("--end", default=None)
    p_bt.add_argument("--step", type=int, default=7)

    p_tune = sub.add_parser("tune", help="grid-search hyperparameters on the validation season")
    p_tune.add_argument("--kind", choices=["goals", "corners"], required=True)
    p_tune.add_argument("--start", default="2025-08-01")
    p_tune.add_argument("--end", default="2026-06-15")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "update":
        from .build import build

        meta = build(force_fetch=not args.offline)
        print(f"built {meta['fixtures']} fixture predictions from {meta['matches_played']} matches")
    elif args.cmd == "export":
        from pathlib import Path

        from .export import export_html

        path = export_html(Path(args.out), artifact=args.artifact)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    elif args.cmd == "serve":
        from .server import serve

        serve(args.host, args.port, args.refresh_hours)
    elif args.cmd in ("backtest", "tune"):
        import pandas as pd

        from . import backtest, config, espn, market

        matches, teams = espn.load_dataset(force=False)
        matches = market.attach(matches, market.market_table(matches, teams, rho=config.MARKET_RHO))
        if args.cmd == "backtest":
            from .build import market_metrics

            end = args.end or pd.Timestamp.now(tz="UTC")
            preds = backtest.rolling(matches, args.start, end, step_days=args.step)
            for kind in ("goals", "corners"):
                s = backtest.summarise(preds, kind)
                for line in s.get("lines", {}).values():
                    line.pop("calibration", None)
                print(kind, json.dumps(s, indent=1, default=float))
            print("vs bookmakers", json.dumps(market_metrics(preds), indent=1, default=float))
        else:
            res = backtest.tune(matches, args.kind, args.start, args.end)
            print(res.to_string(index=False))


if __name__ == "__main__":
    main()
