"""Run the unit-test suite with the active Python interpreter."""
from __future__ import annotations
import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))

