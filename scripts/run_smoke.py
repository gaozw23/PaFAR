"""Run unit tests, both smoke experiments, validation, aggregation, plots, and report."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import tracemalloc
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from pafar_sim.aggregation import aggregate_results, collect_checkpoints
from pafar_sim.config import effective_config_checksum, load_config, project_root
from pafar_sim.exp1.runner import run_experiment1
from pafar_sim.exp2.runner import run_experiment2
from pafar_sim.io_utils import atomic_write_csv, atomic_write_json, environment_manifest
from pafar_sim.plotting import make_tables_and_figures


def validate(frame: pd.DataFrame, config: dict, resume_unchanged: bool) -> list[tuple[str, bool, str]]:
    """Apply the sixteen post-smoke checks with evidence messages."""
    checks: list[tuple[str, bool, str]] = []
    expected_scenarios = set(config["experiment1"]["scenarios"] + config["experiment2"]["scenarios"])
    checks.append(("expected scenarios", expected_scenarios <= set(frame.scenario), str(sorted(set(frame.scenario)))))
    base1 = {"Pointwise-alpha", "Binwise Bonferroni", "Naive maximum", "PaFAR-F", "PaFAR-T", "PaFAR-HC", "Oracle-F"}
    base2 = {"Fixed 0.5", "Youden", "Pointwise-alpha", "Binwise Bonferroni", "Naive maximum", "PaFAR-F", "PaFAR-T", "PaFAR-HC"}
    method_ok = all(base1 <= set(frame.loc[frame.scenario == s, "method"]) for s in ("S1", "S2", "S3"))
    method_ok &= all(base2 <= set(frame.loc[frame.scenario == s, "method"]) for s in ("E1", "E2"))
    checks.append(("expected methods", method_ok, f"{frame.method.nunique()} distinct methods"))
    marginal = frame[frame.method.isin(["PaFAR-F", "PaFAR-T", "Local PaFAR-F", "Local PaFAR-T"])]
    k_ok = True
    for row in marginal.itertuples():
        m0 = int(row.m0); expected_k = ceil((m0 + 1) * (1 - row.alpha))
        k_ok &= int(row.threshold_index) == expected_k and np.isclose(row.alpha_m0, (m0 + 1 - expected_k) / (m0 + 1))
    checks.append(("PaFAR index and alpha_m0", bool(k_ok), f"checked {len(marginal)} rows"))
    checks.append(("strict crossing", True, "test_alerting strict-tie and max-equivalence tests passed"))
    checks.append(("PFA range", bool(frame.pfa.dropna().between(0, 1).all()), "all defined PFA"))
    sens_ppv = frame[["sens3", "sens0", "ppv3_standardized", "ppv0_standardized"]].stack().dropna()
    checks.append(("sensitivity/PPV range", bool(sens_ppv.between(0, 1).all()), "all defined values"))
    checks.append(("nonnegative burden", bool((frame.alert_burden_100d.dropna() >= 0).all()), "all defined burdens"))
    target = frame[frame.method.str.startswith("Local")]
    prefix_ok = list(config["experiment1"]["target_prefixes"]) == sorted(config["experiment1"]["target_prefixes"])
    prefix_ok &= list(config["experiment2"]["target_prefixes"]) == sorted(config["experiment2"]["target_prefixes"])
    prefix_ok &= target.groupby(["scenario", "replicate"]).seed_record.nunique().le(1).all()
    checks.append(("nested target prefixes", bool(prefix_ok), "one seed-fixed reservoir per scenario/replicate"))
    t_rows = frame[frame.method.str.contains("PaFAR-T")]
    checks.append(("template uses validation non-events", bool((t_rows.template_fit_source == "validation_non_events").all()), f"{len(t_rows)} rows"))
    calibration_rows = frame[~frame.method.isin(["Fixed 0.5", "Youden", "Oracle-F"])]
    expected_source = np.where(calibration_rows.method.str.startswith("Local"), "target_calibration_non_events", "calibration_non_events")
    checks.append(("threshold uses correct non-events", bool(np.array_equal(calibration_rows.threshold_fit_source.to_numpy(), expected_source)), f"{len(calibration_rows)} rows"))
    exp2 = frame[frame.experiment == "Experiment II"]
    source_rows = exp2[~exp2.method.str.startswith("Local")]
    local_rows = exp2[exp2.method.str.startswith("Local")]
    m0_ok = (source_rows.m0 == source_rows.realized_calibration_non_events).all()
    m0_ok &= (local_rows.m0 == local_rows.target_m0).all()
    checks.append(("Experiment II realized m0", bool(m0_ok), "m0 equals D=0 calibration count"))
    iteration_ok = (exp2.prediction_iteration_end == exp2.best_iteration + 1).all()
    checks.append(("best-iteration prediction range", bool(iteration_ok), "end=best_iteration+1"))
    checks.append(("causal features", True, "future-perturbation and (t-6,t] tests passed"))
    failure_ok = "learner_failure" in frame and not frame.learner_failure.isna().any()
    checks.append(("failures retained", bool(failure_ok), f"{int(frame.learner_failure.sum())} failure rows"))
    checks.append(("manifest and seeds", bool(frame.seed_record.notna().all()), "every row has seed record"))
    checks.append(("resume", resume_unchanged, "completed checkpoint mtimes unchanged"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--fresh", action="store_true", help="recompute complete smoke checkpoints with identical seeds")
    args = parser.parse_args()
    root = project_root(); output = root / "outputs" / "smoke"
    prior_manifest = None
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and not args.fresh:
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_manifest = None
    loaded = load_config(root / args.config)
    config = loaded.data
    config_checksum = effective_config_checksum(config)
    test_started = time.perf_counter()
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, text=True, capture_output=True)
    test_seconds = time.perf_counter() - test_started
    match = re.search(r"(\d+) passed", tests.stdout)
    passed = int(match.group(1)) if match else 0
    if tests.returncode:
        print(tests.stdout); print(tests.stderr, file=sys.stderr); return tests.returncode
    tracemalloc.start()
    smoke_started = time.perf_counter()
    run_experiment1(config, config_checksum, list(config["experiment1"]["scenarios"]), 0,
                    int(config["experiment1"]["replicates"]) - 1, output, args.n_jobs, not args.fresh)
    run_experiment2(config, config_checksum, list(config["experiment2"]["scenarios"]), 0,
                    int(config["experiment2"]["replicates"]) - 1, output, args.n_jobs, not args.fresh)
    frame = collect_checkpoints(output / "raw")
    learner_files = sorted((output / "learner_metrics").rglob("rep_*.csv.gz"))
    learner_metrics = pd.concat(
        (pd.read_csv(path, float_precision="round_trip") for path in learner_files), ignore_index=True
    ) if learner_files else pd.DataFrame()
    mtimes = {path: path.stat().st_mtime_ns for path in (output / "raw").rglob("rep_*.csv.gz")}
    run_experiment1(config, config_checksum, list(config["experiment1"]["scenarios"]), 0,
                    int(config["experiment1"]["replicates"]) - 1, output, args.n_jobs, True)
    run_experiment2(config, config_checksum, list(config["experiment2"]["scenarios"]), 0,
                    int(config["experiment2"]["replicates"]) - 1, output, args.n_jobs, True)
    resume_unchanged = all(path.stat().st_mtime_ns == stamp for path, stamp in mtimes.items())
    elapsed = time.perf_counter() - smoke_started
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    if prior_manifest and prior_manifest.get("smoke_seconds", 0) > elapsed:
        elapsed = float(prior_manifest["smoke_seconds"])
        peak = max(peak, int(prior_manifest.get("peak_tracemalloc_bytes", 0)))
    summary = aggregate_results(frame)
    failure = frame[frame.learner_failure.fillna(False)]
    timing_cols = [c for c in frame if c.endswith("_seconds")]
    timing = frame.groupby(["experiment", "scenario"], as_index=False)[timing_cols].mean()
    atomic_write_csv(frame, output / "smoke_results_long.csv.gz")
    atomic_write_csv(learner_metrics, output / "learner_metrics.csv")
    atomic_write_csv(summary, output / "smoke_summary.csv")
    atomic_write_csv(failure, output / "failure_summary.csv")
    atomic_write_csv(timing, output / "timing_summary.csv")
    checks = validate(frame, config, resume_unchanged)
    manifest = environment_manifest(root, config_checksum, int(config["master_seed"]))
    manifest.update({
        "config": str(loaded.path), "smoke_seconds": elapsed, "test_seconds": test_seconds,
        "unit_tests_passed": passed, "unit_tests_failed": 0,
        "checks": [{"name": name, "passed": ok, "evidence": evidence} for name, ok, evidence in checks],
        "peak_tracemalloc_bytes": peak, "production_simulation_run": False,
    })
    atomic_write_json(manifest, output / "run_manifest.json")
    make_tables_and_figures(output / "smoke_results_long.csv.gz", output)
    scenario_status = frame.groupby("scenario").agg(rows=("method", "size"), failures=("learner_failure", "sum")).reset_index()
    method_summary = frame.groupby(["scenario", "method", "target_m0"], dropna=False).agg(
        m0=("m0", "first"), k=("threshold_index", "first"), threshold=("threshold", "first"),
        pfa=("pfa", "mean"), sens3=("sens3", "mean")
    ).reset_index()
    undefined = int(frame[["pfa", "sens3", "sens0", "ppv3_standardized", "median_lead"]].isna().sum().sum())
    infinite = int(frame.infinite_threshold.fillna(False).sum())
    tree = "\n".join(str(path.relative_to(root)) for path in sorted(output.rglob("*")) if path.is_file())
    check_lines = "\n".join(f"- {'PASS' if ok else 'FAIL'}: {name} ({evidence})" for name, ok, evidence in checks)
    production_replicates = 500 * 4 + 100 * 2 + 50
    smoke_replicates = int(config["experiment1"]["replicates"]) * 4 + int(config["experiment2"]["replicates"]) * 3
    extrapolated_hours = elapsed * production_replicates / smoke_replicates / 3600
    report = f"""# Smoke test report

## Outcome

- Unit tests: {passed} passed, 0 failed in {test_seconds:.2f} seconds.
- Smoke simulation: {elapsed:.2f} seconds, {len(frame)} method rows.
- Automated checks: {sum(ok for _, ok, _ in checks)}/{len(checks)} passed.
- Learner failure rows: {int(frame.learner_failure.sum())}; infinite thresholds: {infinite}; selected undefined metrics: {undefined}.
- Peak Python allocation observed by `tracemalloc`: {peak / 1024**2:.1f} MiB (does not include all native-library allocations).
- Production simulation was not run.

## Environment

```json
{json.dumps(manifest['packages'], indent=2)}
```

Python: `{manifest['python'].splitlines()[0]}`. RNG: `{manifest['bit_generator']}`.

## Scenario status

```text
{scenario_status.to_string(index=False)}
```

## Method operating points

```text
{method_summary.to_string(index=False)}
```

## Automated validation

{check_lines}

## Actual output tree

```text
{tree}
```

## Runtime interpretation

The measured smoke runtime is {elapsed:.2f} seconds. A purely replicate-count linear extrapolation to the primary counts is approximately {extrapolated_hours:.1f} serial hours. This is a coarse extrapolation, not a measured production runtime; oracle size, data sizes, parallel efficiency, storage, and Experiment II model-fit cost make direct scaling unreliable.

## Unresolved issues

Any failed automated check above is unresolved. Monte Carlo smoke estimates are deliberately noisy and are not manuscript results. Peak memory excludes native XGBoost/NumPy allocations. No PhysioNet data or Section 7 real-data analysis was touched.
"""
    (output / "SMOKE_TEST_REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("## Environment")[0])
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
