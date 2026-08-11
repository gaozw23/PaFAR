"""Run exactly one production-size E1/E2/E3 diagnostic pilot, never production repeats."""
from __future__ import annotations

import gc
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import psutil

from pafar_sim.calibration import hc_threshold, marginal_threshold
from pafar_sim.config import load_config
from pafar_sim.diagnostics import distribution_summary, reconstruct_smoke_exp2
from pafar_sim.io_utils import atomic_write_csv, atomic_write_json
from pafar_sim.metrics import evaluate_metrics


class PeakRSS:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self.stop_event.wait(.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRSS":
        self.thread.start(); return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set(); self.thread.join(); self.peak = max(self.peak, self.process.memory_info().rss)


def main() -> int:
    config = load_config("configs/exp2_primary.yaml").data
    output = Path("outputs/production_pilot"); output.mkdir(parents=True, exist_ok=True)
    timing_rows, result_rows, diagnostic_rows = [], [], []
    for scenario in ("E1", "E2", "E3"):
        scenario_started = time.perf_counter()
        with PeakRSS() as memory:
            try:
                reconstruction = reconstruct_smoke_exp2(config, scenario, 0)
                calibration_started = time.perf_counter()
                calibration = reconstruction.splits["source_calibration"]
                test = reconstruction.splits["test"]
                non_cal = ~calibration.batch.event
                f_values, t_values = calibration.maximum_fixed[non_cal], calibration.maximum_time[non_cal]
                f = marginal_threshold(f_values, float(config["alpha"]))
                t = marginal_threshold(t_values, float(config["alpha"]))
                hc = hc_threshold(f_values, float(config["alpha"]), float(config["delta"]))
                specifications = [
                    ("PaFAR-F", test.logit, f), ("PaFAR-T", test.standardized, t), ("PaFAR-HC", test.logit, hc),
                ]
                if scenario == "E3":
                    reservoir = reconstruction.splits["target_calibration_reservoir"]
                    local_f_values, local_t_values = reservoir.maximum_fixed[:500], reservoir.maximum_time[:500]
                    specifications = [
                        ("Direct source transfer", test.logit, f), ("Source PaFAR-T", test.standardized, t),
                        ("Source PaFAR-HC", test.logit, hc),
                        ("Local PaFAR-F", test.logit, marginal_threshold(local_f_values, float(config["alpha"]))),
                        ("Local PaFAR-T", test.standardized, marginal_threshold(local_t_values, float(config["alpha"]))),
                        ("Local PaFAR-HC", test.logit, hc_threshold(local_f_values, float(config["alpha"]), float(config["delta"]))),
                    ]
                calibration_seconds = reconstruction.timings["calibration_seconds"] + (time.perf_counter() - calibration_started)
                metric_started = time.perf_counter()
                for method, score, threshold in specifications:
                    metric = evaluate_metrics(
                        score, test.batch.eligible, threshold.threshold, test.batch.event,
                        test.batch.onset, test.batch.horizon, test.batch.patient_ids,
                    )
                    result_rows.append({
                        "scenario": scenario, "replicate": 0, "site": "B" if scenario == "E3" else "A",
                        "method": method, "threshold": threshold.threshold, "threshold_index": threshold.index,
                        "m0": threshold.m0, "alpha_m0": threshold.alpha_m0,
                        "pfa": metric.pfa, "sens3": metric.sens3, "sens0": metric.sens0,
                        "learner_failure": False, "best_iteration": reconstruction.learner.best_iteration,
                    })
                metric_seconds = time.perf_counter() - metric_started
                for split_name, split in (("source_calibration_non_events", calibration), ("test_non_events", test)):
                    patient_mask = ~split.batch.event
                    for stage, values in (
                        ("raw_probability", split.risk[patient_mask][split.batch.eligible[patient_mask]]),
                        ("causal_moving_average", split.smoothed[patient_mask][split.batch.eligible[patient_mask]]),
                        ("patient_maximum_fixed", split.maximum_fixed[patient_mask]),
                        ("patient_maximum_time", split.maximum_time[patient_mask]),
                    ):
                        diagnostic_rows.append({"scenario": scenario, "split": split_name, "score_stage": stage, **distribution_summary(values)})
                failure, reason = False, ""
            except Exception as exc:
                failure, reason = True, f"{type(exc).__name__}: {exc}"
                calibration_seconds = metric_seconds = np.nan
                result_rows.append({"scenario": scenario, "replicate": 0, "site": "B" if scenario == "E3" else "A", "method": "learner_failure", "learner_failure": True, "failure_reason": reason})
                reconstruction = None
        wall = time.perf_counter() - scenario_started
        timings = reconstruction.timings if reconstruction is not None else {}
        timing_rows.append({
            "scenario": scenario, "replicate": 0, "dgp_seconds": timings.get("dgp_seconds", np.nan),
            "feature_construction_seconds": timings.get("feature_seconds", np.nan),
            "learner_fit_seconds": timings.get("learner_fit_seconds", np.nan),
            "calibration_seconds": calibration_seconds, "metric_seconds": metric_seconds,
            "aggregation_write_seconds": np.nan, "total_wall_seconds": wall,
            "peak_rss_bytes": memory.peak, "peak_rss_gib": memory.peak / 1024**3,
            "best_iteration": reconstruction.learner.best_iteration if reconstruction is not None else np.nan,
            "learner_failure": failure, "failure_reason": reason,
        })
        del reconstruction; gc.collect()
    write_started = time.perf_counter()
    atomic_write_csv(pd.DataFrame(result_rows), output / "production_one_rep_results_after_fix.csv.gz")
    atomic_write_csv(pd.DataFrame(diagnostic_rows), output / "production_one_rep_score_diagnostics_after_fix.csv")
    write_seconds = time.perf_counter() - write_started
    for row in timing_rows:
        row["aggregation_write_seconds"] = write_seconds / len(timing_rows)
        row["total_wall_seconds"] += row["aggregation_write_seconds"]
    atomic_write_csv(pd.DataFrame(timing_rows), output / "production_one_rep_timing_after_fix.csv")
    atomic_write_json({
        "purpose": "one-replicate production-size diagnostic only", "production_replicates_run": 0,
        "pilot_replicates": {"E1": 1, "E2": 1, "E3": 1}, "n_jobs": 1,
        "config": "configs/exp2_primary.yaml", "manuscript_tables_include_pilot": False,
    }, output / "pilot_manifest.json")
    print(pd.DataFrame(timing_rows).to_string(index=False))
    return 1 if any(row["learner_failure"] for row in timing_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
