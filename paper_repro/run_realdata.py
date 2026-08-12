"""Run the maintained PaFAR PhysioNet pipeline without downloading data."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "RUN_PAFAR_REALDATA_PRIMARY"
STAGES = ("all", "raw", "cohort", "features", "internal", "transfer", "bootstrap", "robustness", "manuscript")


def _raw_directories(config_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_spec = Path(config.get("project_root", "."))
    project = (config_path.parent.parent / project_spec).resolve()
    return project / config["raw"]["A"], project / config["raw"]["B"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/realdata_primary.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    config_arg = Path(args.config)
    if config_arg.is_absolute() or ".." in config_arg.parts:
        parser.error("--config must be a project-relative path")
    config_path = ROOT / config_arg
    if not config_path.is_file():
        parser.error(f"configuration not found: {config_arg}")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1")
    if not args.check_only and args.confirm != CONFIRMATION:
        parser.error(f"analysis execution requires --confirm {CONFIRMATION}")

    raw_a, raw_b = _raw_directories(config_path)
    missing = [path for path in (raw_a, raw_b) if not path.is_dir()]
    if missing:
        print("PhysioNet/CinC 2019 raw data are not included in this repository.", file=sys.stderr)
        print("Place authorized patient files in:", file=sys.stderr)
        print(f"  {raw_a.relative_to(ROOT)}", file=sys.stderr)
        print(f"  {raw_b.relative_to(ROOT)}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        str(ROOT / "scripts/run_realdata_pipeline.py"),
        "--config", config_arg.as_posix(),
        "--stage", args.stage,
        "--n-jobs", str(args.n_jobs),
    ]
    if args.resume:
        command.append("--resume")
    if args.check_only:
        command.append("--check-only")
    else:
        command.extend(["--confirm", CONFIRMATION])
    print("Running:", shlex.join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
