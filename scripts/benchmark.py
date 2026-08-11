"""Small reproducible benchmarks for the three prescribed hotspots."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from pafar_sim.exp1.dgp import generate_exp1
from pafar_sim.exp1.oracle import build_oracle
from pafar_sim.exp2.dgp import generate_exp2
from pafar_sim.exp2.features import build_raw_features

timings = {}
start = time.perf_counter(); batch = generate_exp1(np.random.default_rng(1), 1000, "S2", False); timings["exp1_1000_trajectories_seconds"] = time.perf_counter() - start
start = time.perf_counter(); ehr = generate_exp2(np.random.default_rng(2), 100, "E2"); build_raw_features(ehr); timings["feature_kernel_100_patients_seconds"] = time.perf_counter() - start
target = Path("outputs/benchmarks/oracle_2000.npy")
start = time.perf_counter(); build_oracle(target, "S2", "A", 2000, 99, chunk_size=500); timings["oracle_2000_seconds"] = time.perf_counter() - start
target.parent.mkdir(parents=True, exist_ok=True)
(target.parent / "benchmark.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
print(json.dumps(timings, indent=2))

