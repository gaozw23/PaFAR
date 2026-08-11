"""Preflight and explicitly guarded production orchestration.

This script is safe by default: without --execute it only validates effective
configurations and prints the intended work. Oracle construction is serialized
before any parallel Experiment I replicates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import psutil

from pafar_sim.config import effective_config_checksum, load_config
from pafar_sim.exp1.oracle import build_oracle, load_oracle, oracle_filename
from pafar_sim.exp1.runner import run_experiment1
from pafar_sim.exp2.runner import run_experiment2
from pafar_sim.production import finalize_primary_production, load_and_validate_lock, run_primary_gate


class PeakRSS:
    def __init__(self) -> None:
        self.process = psutil.Process(); self.peak = self.process.memory_info().rss
        self.stop = threading.Event(); self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self.stop.wait(.1):
            rss = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try: rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied): pass
            self.peak = max(self.peak, rss)

    def __enter__(self): self.thread.start(); return self
    def __exit__(self, *_): self.stop.set(); self.thread.join()


def _prepare_oracles(config: dict, output: Path) -> None:
    exp = config["experiment1"]
    for scenario in exp["scenarios"]:
        site = "B" if scenario == "S4" else "A"
        kwargs = dict(
            hmax=int(config["hmax"]), tmin=int(config["tmin"]),
            smooth_length=int(config["smooth_length"]),
        )
        path = output / "oracle" / oracle_filename(
            scenario, site, int(exp["oracle_nref"]), int(config["master_seed"]), **kwargs,
        )
        if path.exists():
            load_oracle(path, scenario, site, int(exp["oracle_nref"]), int(config["master_seed"]), **kwargs)
        else:
            build_oracle(
                path, scenario, site, int(exp["oracle_nref"]), int(config["master_seed"]),
                chunk_size=int(exp.get("oracle_chunk_size", 25_000)), **kwargs,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/production")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-production", default="")
    args = parser.parse_args()
    exp1 = load_config("configs/exp1_primary.yaml").data
    exp2 = load_config("configs/exp2_primary.yaml").data
    print(f"Experiment I effective checksum: {effective_config_checksum(exp1)}")
    print(f"Experiment II effective checksum: {effective_config_checksum(exp2)}")
    print("Planned counts: S1-S4=500 each; E1=100; E2=100; E3=50")
    print(f"Planned parallelism: n_jobs={args.n_jobs}; oracles are built/validated serially first")
    if not args.execute:
        print("CHECK ONLY: no oracle or production simulation was run")
        return 0
    if args.confirm_production != "RUN_PRODUCTION":
        raise SystemExit("Refusing production execution without --confirm-production RUN_PRODUCTION")
    output = Path(args.output_root)
    lock = load_and_validate_lock(output, exp1, exp2)
    started = time.perf_counter()
    with PeakRSS() as memory:
        print("Stage 1/4: serial oracle preparation and validation")
        _prepare_oracles(exp1, output)
        print("Stage 2/4: final-result gate prefixes")
        run_experiment1(exp1, effective_config_checksum(exp1), ["S1", "S2", "S3", "S4"], 0, 19, output, args.n_jobs, True)
        run_experiment2(exp2, effective_config_checksum(exp2), ["E1", "E2", "E3"], 0, 9, output, args.n_jobs, True)
        gate = run_primary_gate(output, exp1, exp2, lock)
        print(f"PRIMARY gate passed: {gate}")
        print("Stage 3/4: remaining PRIMARY replicates with --resume semantics")
        run_experiment1(exp1, effective_config_checksum(exp1), ["S1", "S2", "S3", "S4"], 20, 499, output, args.n_jobs, True)
        run_experiment2(exp2, effective_config_checksum(exp2), ["E1", "E2"], 10, 99, output, args.n_jobs, True)
        run_experiment2(exp2, effective_config_checksum(exp2), ["E3"], 10, 49, output, args.n_jobs, True)
        print("Stage 4/4: aggregate, tables, figures, assertions, report, review bundle")
        runtime = {"wall_seconds": time.perf_counter() - started, "peak_rss_bytes": memory.peak, "peak_rss_gib": memory.peak / 1024**3, "n_jobs": args.n_jobs}
        finalize_primary_production(output, exp1, exp2, lock, runtime)
    print(f"PRIMARY production complete in {time.perf_counter()-started:.1f} seconds; peak RSS {memory.peak/1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
