"""Command line interface.

Usage:
    python -m bess_degradation fetch                          # ACEN CSVs + selected cell data (about 0.46 GB)
    python -m bess_degradation fetch --nem                    # also the AEMO NEM cache (about 1.9 GB download)
    python -m bess_degradation run --nem-data-dir ../nem-battery-behaviour/data --jobs 8
    python -m bess_degradation cells                          # only the cell analysis and ageing fit
    python -m bess_degradation run --help

Data live in ./data (``--nem-data-dir`` or BESS_NEM_DATA_DIR points to an existing NEM cache), results go to
./results, charts to ./docs/img and the interactive dashboard to ./docs/dashboard.html.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from bess_degradation import acen, aging, cells, nem, pipeline

log = logging.getLogger("bess_degradation")


def _date(s: str) -> date:
    return date.fromisoformat(s)


def cmd_fetch(args):
    acen.fetch(args.acen_dir)
    cells.fetch(args.cells_dir)
    if args.nem:
        nem.fetch(args.start, args.end, args.nem_data_dir)


def _model(args) -> aging.AgingModel:
    path = args.results_dir / "aging_fit.json"
    if not path.exists():
        return cmd_cells(args)["model"]
    m = aging.AgingModel(**json.loads(path.read_text())["model"])
    return m.with_(beta=args.beta)


def cmd_cells(args):
    res = pipeline.cell_analysis(args.cells_dir, args.results_dir, beta=args.beta)
    log.info("ageing model: %s", asdict(res["model"]))
    return res


def cmd_soc(args):
    return pipeline.soc_validation(args.nem_data_dir, args.acen_dir, args.results_dir)


def cmd_fleet(args):
    monthly = pipeline.fleet_reported_soc(args.nem_data_dir, args.results_dir, args.start, args.end)
    summary = pipeline.fleet_summary(monthly)
    summary.to_csv(args.results_dir / "fleet_reported_soc_summary.csv", index=False, float_format="%.4g")
    return monthly, summary


def cmd_strategies(args, model=None):
    model = model or _model(args)
    return pipeline.strategies(args.nem_data_dir, args.results_dir, model, args.start, args.end, args.units,
                               tuple(args.replacement_costs), args.headline_cost, temp_c=args.temp, jobs=args.jobs)


def cmd_run(args):
    from bess_degradation import plots
    soc_res = cmd_soc(args)
    fleet_monthly, fleet = cmd_fleet(args)
    cell_res = cmd_cells(args)
    strat = cmd_strategies(args, cell_res["model"])
    fan = pipeline.soh_fan(strat["evals"], cell_res["model"], cell_res["model_power_law"])
    fan.to_csv(args.results_dir / "soh_fan.csv", index=False, float_format="%.5g")
    plots.make_all(soc_res, fleet_monthly, fleet, cell_res, strat, fan, args.img_dir, args.dashboard,
                   headline=f"pf_aware_{int(args.headline_cost)}")
    (args.results_dir / "run.json").write_text(json.dumps({
        "strategy_period": [str(args.start), str(args.end)], "units": args.units, "temperature_c": args.temp,
        "replacement_cost_aud_per_kwh": args.replacement_costs, "headline_cost_aud_per_kwh": args.headline_cost,
        "beta": args.beta, "model": asdict(cell_res["model"]),
    }, indent=2, default=float))
    log.info("results in %s, charts in %s, dashboard %s", args.results_dir, args.img_dir, args.dashboard)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bess-degradation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "soc", "fleet", "cells", "strategies", "run"):
        p = sub.add_parser(name)
        p.add_argument("--nem-data-dir", type=Path, default=Path(os.environ.get("BESS_NEM_DATA_DIR", "data/nem")))
        p.add_argument("--acen-dir", type=Path, default=Path("data/acen"))
        p.add_argument("--cells-dir", type=Path, default=Path("data/cells"))
        p.add_argument("--results-dir", type=Path, default=Path("results"))
        p.add_argument("--start", type=_date, default=date(2025, 10, 1))
        p.add_argument("--end", type=_date, default=date(2026, 9, 30), help="inclusive")
        p.add_argument("--beta", type=float, default=1.0,
                       help="depth-of-cycle exponent of the ageing model (1 = throughput only)")
        if name == "fetch":
            p.add_argument("--nem", action="store_true", help="also build the AEMO NEM cache")
        if name in ("strategies", "run"):
            p.add_argument("--units", nargs="+", default=pipeline.STRATEGY_UNITS, help="DUIDs")
            p.add_argument("--replacement-costs", nargs="+", type=float, default=[125, 250, 500],
                           help="AUD per kWh of capacity for the degradation-aware strategy")
            p.add_argument("--headline-cost", type=float, default=250.0)
            p.add_argument("--temp", type=float, default=25.0, help="cell temperature, degC")
            p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        if name == "run":
            p.add_argument("--img-dir", type=Path, default=Path("docs/img"))
            p.add_argument("--dashboard", type=Path, default=Path("docs/dashboard.html"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    {"fetch": cmd_fetch, "soc": cmd_soc, "fleet": cmd_fleet, "cells": cmd_cells, "strategies": cmd_strategies,
     "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
