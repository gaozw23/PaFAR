"""Print and validate the execution environment."""
from __future__ import annotations
import importlib
import platform
import sys

MINIMUM = (3, 10)
PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "joblib", "yaml", "pytest")

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
if sys.version_info < MINIMUM:
    raise SystemExit("Python 3.10 or newer is required")
for name in PACKAGES:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")

