"""Report whether the local environment is ready for PaFAR reproduction."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


MINIMUM_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-import",
        action="store_true",
        help="also check whether the installed pafar_sim package is importable",
    )
    args = parser.parse_args()

    pyproject = ROOT / "pyproject.toml"
    requirements = ROOT / "requirements.txt"
    checks = {
        "python_3_10_or_newer": sys.version_info >= MINIMUM_PYTHON,
        "running_from_repository_root": Path.cwd().resolve() == ROOT,
        "pyproject_toml_present": pyproject.is_file(),
        "requirements_txt_present": requirements.is_file(),
    }
    if args.check_import:
        checks["pafar_sim_importable"] = importlib.util.find_spec("pafar_sim") is not None

    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Current directory: {Path.cwd().resolve()}")
    print(f"Repository root: {ROOT}")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    print("\nRecommended installation commands:")
    print("  python -m venv .venv")
    print('  python -m pip install --upgrade pip')
    print('  python -m pip install -e ".[test]"')
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
