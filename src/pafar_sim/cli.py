"""Command-line interface for reproducible simulations and saved-output reporting."""
from __future__ import annotations

import argparse
from pathlib import Path

from .aggregation import write_aggregates
from .config import apply_condition, effective_config_checksum, load_config, project_root
from .exp1.oracle import build_oracle, oracle_filename
from .exp1.runner import run_experiment1
from .exp2.runner import run_experiment2
from .plotting import make_tables_and_figures


def _common_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicate-end", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--output-root", default="outputs/production")
    parser.add_argument("--resume", action="store_true")


def main(argv: list[str] | None = None) -> int:
    """Dispatch simulation, oracle, aggregation, and plotting commands."""
    parser = argparse.ArgumentParser(prog="pafar-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    exp1 = sub.add_parser("run-exp1"); _common_run(exp1)
    exp2 = sub.add_parser("run-exp2"); _common_run(exp2)
    oracle = sub.add_parser("build-oracle")
    oracle.add_argument("--config", required=True); oracle.add_argument("--scenario", required=True)
    oracle.add_argument("--site", default="A"); oracle.add_argument("--output-root", default="outputs/oracle")
    aggregate = sub.add_parser("aggregate"); aggregate.add_argument("--raw-root", required=True); aggregate.add_argument("--output-root", required=True)
    figures = sub.add_parser("figures"); figures.add_argument("--results", required=True); figures.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        write_aggregates(args.raw_root, args.output_root); return 0
    if args.command == "figures":
        make_tables_and_figures(args.results, args.output_dir); return 0
    loaded = load_config(args.config)
    config = apply_condition(loaded.data, getattr(args, "condition", None))
    if getattr(args, "master_seed", None) is not None:
        config["master_seed"] = args.master_seed
    checksum = effective_config_checksum(config)
    if args.command == "build-oracle":
        exp = config["experiment1"]
        target = Path(args.output_root) / oracle_filename(
            args.scenario, args.site, int(exp["oracle_nref"]), int(config["master_seed"]),
            hmax=int(config["hmax"]), tmin=int(config["tmin"]), smooth_length=int(config["smooth_length"]),
        )
        build_oracle(target, args.scenario, args.site, int(exp["oracle_nref"]), int(config["master_seed"]),
                     hmax=int(config["hmax"]), tmin=int(config["tmin"]), smooth_length=int(config["smooth_length"]),
                     chunk_size=int(exp.get("oracle_chunk_size", 25000)))
        return 0
    default_scenarios = config["experiment1" if args.command == "run-exp1" else "experiment2"]["scenarios"]
    scenarios = args.scenarios or list(default_scenarios)
    runner = run_experiment1 if args.command == "run-exp1" else run_experiment2
    runner(config, checksum, scenarios, args.replicate_start, args.replicate_end,
           args.output_root, args.n_jobs, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
