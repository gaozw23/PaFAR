"""Build manuscript-ready tables and figures from existing saved summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "BUILD_PAFAR_PAPER_OUTPUTS"
SIMULATION_SOURCE = ROOT / "outputs/production/all_replicate_results.csv.gz"
REALDATA_SOURCES = (
    ROOT / "outputs/realdata/internal_primary/internal_results_summary.csv",
    ROOT / "outputs/realdata/transfer_primary/transfer_results_summary.csv",
)


def _run(command: list[str]) -> int:
    print("Running:", shlex.join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("all", "simulation", "realdata"), default="all")
    parser.add_argument("--config", default="configs/realdata_primary.yaml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    config_arg = Path(args.config)
    if config_arg.is_absolute() or ".." in config_arg.parts:
        parser.error("--config must be a project-relative path")
    if not (ROOT / config_arg).is_file():
        parser.error(f"configuration not found: {config_arg}")
    if not args.check_only and args.confirm != CONFIRMATION:
        parser.error(f"output generation requires --confirm {CONFIRMATION}")

    need_simulation = args.component in {"all", "simulation"}
    need_realdata = args.component in {"all", "realdata"}
    missing: list[Path] = []
    if need_simulation and not SIMULATION_SOURCE.is_file():
        missing.append(SIMULATION_SOURCE)
    if need_realdata:
        missing.extend(path for path in REALDATA_SOURCES if not path.is_file())
    if missing:
        print("Required saved summaries are missing; analyses will not be rerun automatically.", file=sys.stderr)
        for path in missing:
            print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
        return 2

    if args.check_only:
        if need_simulation:
            return _run([sys.executable, str(ROOT / "scripts/build_primary_manuscript_outputs.py"), "--check-only"])
        print("All requested saved summaries are present; no outputs were written.")
        return 0

    if need_simulation:
        code = _run([sys.executable, str(ROOT / "scripts/build_primary_manuscript_outputs.py")])
        if code:
            return code
    if need_realdata:
        code = _run([
            sys.executable,
            str(ROOT / "scripts/build_realdata_manuscript_outputs.py"),
            "--config", config_arg.as_posix(),
        ])
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
