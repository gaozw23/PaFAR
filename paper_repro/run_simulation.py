"""Run frozen PaFAR primary simulations through the maintained runners."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "RUN_PAFAR_PRIMARY_SIMULATION"
PRIMARY = {
    "exp1": {"config": "configs/exp1_primary.yaml", "script": "scripts/run_exp1.py"},
    "exp2": {"config": "configs/exp2_primary.yaml", "script": "scripts/run_exp2.py"},
}


def _groups(experiment: str, scenarios: list[str] | None, replicate_end: int | None) -> list[tuple[list[str], int]]:
    if experiment == "exp1":
        selected = scenarios or ["S1", "S2", "S3", "S4"]
        if any(item not in {"S1", "S2", "S3", "S4"} for item in selected):
            raise ValueError("Experiment I scenarios must be S1, S2, S3, or S4")
        return [(selected, 499 if replicate_end is None else replicate_end)]

    selected = scenarios or ["E1", "E2", "E3"]
    if any(item not in {"E1", "E2", "E3"} for item in selected):
        raise ValueError("Experiment II scenarios must be E1, E2, or E3")
    if replicate_end is not None:
        return [(selected, replicate_end)]
    groups: list[tuple[list[str], int]] = []
    e12 = [item for item in selected if item in {"E1", "E2"}]
    if e12:
        groups.append((e12, 99))
    if "E3" in selected:
        groups.append((["E3"], 49))
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(PRIMARY), required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicate-end", type=int, help="override the frozen primary end index")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output-root", default="outputs/production")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.confirm != CONFIRMATION:
        parser.error(f"production execution requires --confirm {CONFIRMATION}")
    if args.replicate_start < 0 or (args.replicate_end is not None and args.replicate_end < args.replicate_start):
        parser.error("replicate indices must define a nonnegative inclusive range")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1")

    output_arg = Path(args.output_root)
    if output_arg.is_absolute() or ".." in output_arg.parts:
        parser.error("--output-root must be a project-relative path")
    output = ROOT / output_arg
    if output.exists() and not args.resume:
        parser.error(f"{output_arg} already exists; use --resume to protect completed checkpoints")

    try:
        groups = _groups(args.experiment, args.scenarios, args.replicate_end)
    except ValueError as error:
        parser.error(str(error))
    selected = PRIMARY[args.experiment]
    for scenarios, end in groups:
        command = [
            sys.executable,
            str(ROOT / selected["script"]),
            "--config", selected["config"],
            "--replicate-start", str(args.replicate_start),
            "--replicate-end", str(end),
            "--n-jobs", str(args.n_jobs),
            "--output-root", output_arg.as_posix(),
        ]
        for scenario in scenarios:
            command.extend(["--scenario", scenario])
        if args.resume:
            command.append("--resume")
        print("Running:", shlex.join(command))
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
