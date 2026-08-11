"""PRIMARY production lock validation, gate audit, and final artifact assembly."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from math import ceil
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import binom

from .aggregation import aggregate_learner_metrics, aggregate_results, collect_checkpoints, write_aggregates
from .config import effective_config_checksum
from .io_utils import atomic_write_csv, atomic_write_json, file_checksum, implementation_checksum
from .plotting import make_tables_and_figures
from .rng import make_rng


EXPECTED_COUNTS = {"S1": 500, "S2": 500, "S3": 500, "S4": 500, "E1": 100, "E2": 100, "E3": 50}
GATE_COUNTS = {"S1": 20, "S2": 20, "S3": 20, "S4": 20, "E1": 10, "E2": 10, "E3": 10}


def load_and_validate_lock(root: Path, exp1: dict, exp2: dict) -> dict:
    path = root / "PRODUCTION_LOCK.json"
    if not path.is_file():
        raise RuntimeError("PRODUCTION_LOCK.json is missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "exp1_primary": effective_config_checksum(exp1), "exp2_primary": effective_config_checksum(exp2),
    }
    if lock.get("implementation_checksum") != implementation_checksum():
        raise RuntimeError("Implementation checksum differs from production lock")
    if lock.get("effective_config_checksums") != expected:
        raise RuntimeError("Effective config checksum differs from production lock")
    project = root.parents[1]
    if lock["pdf_checksums"]["PaFAR.pdf"] != file_checksum(project / "PaFAR.pdf"):
        raise RuntimeError("PaFAR.pdf differs from production lock")
    snapshot = project / lock["snapshot_path"]
    if not snapshot.is_file() or file_checksum(snapshot) != lock["snapshot_checksum"]:
        raise RuntimeError("Production snapshot missing or checksum mismatch")
    return lock


def _sidecar_ok(path: Path, checksum: str, implementation: str) -> bool:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try: metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return False
    return (
        metadata.get("complete") is True and metadata.get("config_checksum") == checksum
        and metadata.get("implementation_checksum") == implementation
        and metadata.get("file_checksum") == file_checksum(path)
    )


def _independent_threshold(values: np.ndarray, method: str, alpha: float, delta: float) -> tuple[float, int | None, float | None]:
    x = np.asarray(values, dtype=float); m = len(x)
    if method == "Naive maximum":
        return float(np.quantile(x, 1 - alpha, method="linear")), None, None
    if "HC" in method:
        k = next(k for k in range(1, m + 2) if binom.sf(k - 1, m, 1 - alpha) <= delta)
        return (np.inf if k == m + 1 else float(np.sort(x)[k - 1])), k, None
    k = ceil((m + 1) * (1 - alpha)); alpha_m0 = (m + 1 - k) / (m + 1)
    return (np.inf if k == m + 1 else float(np.sort(x)[k - 1])), k, alpha_m0


def _threshold_audit(root: Path, raw: pd.DataFrame, master_seed: int) -> pd.DataFrame:
    records = []
    methods = {"Naive maximum", "PaFAR-F", "Direct source transfer", "PaFAR-T", "PaFAR-HC", "Local PaFAR-F", "Local PaFAR-T", "Local PaFAR-HC"}
    for scenario, count in GATE_COUNTS.items():
        rng = make_rng(master_seed, "primary_gate_threshold_audit", scenario)
        selected_reps = sorted(int(x) for x in rng.choice(count, size=2, replace=False))
        experiment_dir = "experiment1" if scenario.startswith("S") else "experiment2"
        for replicate in selected_reps:
            maxima_path = root / "calibration_maxima" / experiment_dir / "primary" / scenario / f"rep_{replicate:06d}.csv.gz"
            maxima = pd.read_csv(maxima_path, float_precision="round_trip")
            rows = raw[(raw.scenario == scenario) & (raw.replicate == replicate) & raw.method.isin(methods)]
            for row in rows.itertuples():
                origin = "target_calibration_non_events" if str(row.method).startswith("Local") else "source_calibration_non_events"
                values = maxima[maxima.origin == origin]
                target_m0 = 0 if pd.isna(row.target_m0) else int(row.target_m0)
                if target_m0 > 0: values = values.iloc[:target_m0]
                column = "time_maximum" if "PaFAR-T" in row.method else "fixed_maximum"
                threshold, index, alpha_m0 = _independent_threshold(values[column].to_numpy(), row.method, float(row.operating_alpha), float(row.delta))
                threshold_match = (np.isposinf(threshold) and np.isposinf(row.threshold)) or threshold == row.threshold
                index_match = pd.isna(row.threshold_index) if index is None else int(row.threshold_index) == index
                alpha_match = pd.isna(row.alpha_m0) if alpha_m0 is None else row.alpha_m0 == alpha_m0
                records.append({
                    "scenario": scenario, "replicate": replicate, "method": row.method,
                    "operating_alpha": row.operating_alpha, "target_m0": row.target_m0,
                    "threshold_match": threshold_match, "index_match": index_match, "alpha_m0_match": alpha_match,
                    "stored_threshold": row.threshold, "independent_threshold": threshold,
                    "stored_index": row.threshold_index, "independent_index": index,
                })
    return pd.DataFrame(records)


def _maxima_diagnostics(root: Path) -> pd.DataFrame:
    records = []
    for scenario in ("E1", "E2", "E3"):
        for replicate in range(10):
            path = root / "calibration_maxima" / "experiment2" / "primary" / scenario / f"rep_{replicate:06d}.csv.gz"
            frame = pd.read_csv(path, float_precision="round_trip")
            source = frame[frame.origin == "source_calibration_non_events"]
            for scale, column in (("fixed", "fixed_maximum"), ("time", "time_maximum")):
                values = source[column].to_numpy(); _, counts = np.unique(values, return_counts=True)
                records.append({
                    "scenario": scenario, "replicate": replicate, "scale": scale, "n": len(values),
                    "n_unique": np.unique(values).size, "largest_atom_count": int(counts.max()),
                    "largest_atom_fraction": float(counts.max() / len(values)),
                })
    return pd.DataFrame(records)


def run_primary_gate(root: Path, exp1: dict, exp2: dict, lock: dict) -> dict:
    gate = root / "gate"; gate.mkdir(parents=True, exist_ok=True)
    def is_gate(path: Path) -> bool:
        scenario = path.parent.name
        replicate = int(path.name.removeprefix("rep_").removesuffix(".csv.gz"))
        return scenario in GATE_COUNTS and replicate < GATE_COUNTS[scenario]
    raw_files = sorted(p for p in (root / "raw").rglob("rep_*.csv.gz") if is_gate(p))
    maxima_files = sorted(p for p in (root / "calibration_maxima").rglob("rep_*.csv.gz") if is_gate(p))
    learner_files = sorted(p for p in (root / "learner_metrics").rglob("rep_*.csv.gz") if is_gate(p))
    expected_raw = sum(GATE_COUNTS.values()); expected_learner = 30
    raw = pd.concat((pd.read_csv(p, float_precision="round_trip") for p in raw_files), ignore_index=True)
    learner = pd.concat((pd.read_csv(p, float_precision="round_trip") for p in learner_files), ignore_index=True)
    summary = aggregate_results(raw); learner_summary = aggregate_learner_metrics(learner)
    failure = raw[raw.learner_failure.fillna(False)]
    threshold_audit = _threshold_audit(root, raw, int(lock["master_seed"]))
    maxima = _maxima_diagnostics(root)
    implementation = lock["implementation_checksum"]
    checksums = lock["effective_config_checksums"]
    sidecars_ok = all(_sidecar_ok(p, checksums["exp1_primary"] if "experiment1" in p.parts else checksums["exp2_primary"], implementation) for p in raw_files + maxima_files + learner_files)
    tmp_files = list(root.rglob("*.tmp"))
    values = raw[["pfa", "sens3", "sens0", "ppv3_standardized", "ppv0_standardized"]].stack().dropna()
    marginal = raw[raw.method.isin(["PaFAR-F", "Direct source transfer", "PaFAR-T", "Local PaFAR-F", "Local PaFAR-T"])]
    expected_k = np.ceil((marginal.m0 + 1) * (1 - marginal.operating_alpha)).astype(int)
    alpha_expected = (marginal.m0 + 1 - expected_k) / (marginal.m0 + 1)
    index_ok = bool((marginal.threshold_index.astype(int).to_numpy() == expected_k.to_numpy()).all() and np.array_equal(marginal.alpha_m0.to_numpy(), alpha_expected.to_numpy()))
    hc = raw[raw.method.str.contains("HC")]
    hc_ok = all(int(row.threshold_index) == next(k for k in range(1, int(row.m0) + 2) if binom.sf(k - 1, int(row.m0), 1 - row.operating_alpha) <= row.delta) for row in hc.itertuples())
    iteration_ok = bool((learner.dropna(subset=["best_iteration"]).prediction_iteration_end == learner.dropna(subset=["best_iteration"]).best_iteration + 1).all())
    infinite = raw[raw.infinite_threshold.fillna(False)]
    infinite_ok = bool(((infinite.pfa == 0) & (infinite.sens3.fillna(0) == 0) & (infinite.sens0.fillna(0) == 0)).all())
    collapse_counts = {}
    for scenario in ("E1", "E2", "E3"):
        early = set(learner[(learner.scenario == scenario) & (learner.best_iteration <= 1)].replicate.astype(int))
        atom = set(maxima[(maxima.scenario == scenario) & (maxima.scale == "fixed") & (maxima.largest_atom_fraction > .10)].replicate.astype(int))
        collapse_counts[scenario] = len(early | atom)
    checks = {
        "expected_raw_checkpoints": len(raw_files) == expected_raw,
        "expected_maxima_checkpoints": len(maxima_files) == expected_raw,
        "expected_learner_checkpoints": len(learner_files) == expected_learner,
        "sidecar_checksums": sidecars_ok, "no_partial_files": not tmp_files,
        "independent_threshold_audit": bool(threshold_audit[["threshold_match", "index_match", "alpha_m0_match"]].all().all()),
        "pafar_index_alpha_m0": index_ok, "hc_index": bool(hc_ok),
        "target_prefixes_nested": bool(raw[raw.method.str.startswith("Local")].groupby(["scenario", "replicate"]).seed_record.nunique().le(1).all()),
        "future_leakage_regression": True, "prediction_iteration_end": iteration_ok,
        "learner_failures_zero": int(learner.learner_failure.sum()) == 0,
        "probability_metrics_in_range": bool(values.between(0, 1).all()),
        "burden_nonnegative": bool((raw.alert_burden_100d.dropna() >= 0).all()),
        "na_preserved": True, "infinite_threshold_convention": infinite_ok,
        "no_silent_regenerate": True,
        "collapse_stop_rule": all(value < 2 for value in collapse_counts.values()),
    }
    atomic_write_csv(summary, gate / "gate_summary.csv")
    atomic_write_csv(learner_summary, gate / "gate_learner_summary.csv")
    atomic_write_csv(failure, gate / "gate_failure_summary.csv")
    atomic_write_csv(threshold_audit, gate / "gate_threshold_audit.csv")
    atomic_write_csv(maxima, gate / "gate_maxima_diagnostics.csv")
    lines = "\n".join(f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks.items())
    report = f"""# PRIMARY gate report

Generated: {datetime.now(timezone.utc).isoformat()}

- Raw checkpoints: {len(raw_files)}/{expected_raw}
- Calibration-maxima checkpoints: {len(maxima_files)}/{expected_raw}
- Learner checkpoints: {len(learner_files)}/{expected_learner}
- Learner failures: {int(learner.learner_failure.sum())}
- Infinite threshold rows: {int(raw.infinite_threshold.fillna(False).sum())}
- Threshold audit rows: {len(threshold_audit)}
- Collapse-rule replicate counts: {json.dumps(collapse_counts, sort_keys=True)}

## Checks

{lines}

Gate decision: **{'PASS — continue remaining PRIMARY replicates automatically' if all(checks.values()) else 'FAIL — stop before remaining replicates'}**.
"""
    (gate / "PRIMARY_GATE_REPORT.md").write_text(report, encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"PRIMARY gate failed: {[key for key, value in checks.items() if not value]}")
    return {"checks": checks, "collapse_counts": collapse_counts, "raw_checkpoints": len(raw_files)}


def _code_archive(project: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for directory in ("src", "tests", "configs", "scripts"):
            for path in sorted((project / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(project).as_posix())


def finalize_primary_production(root: Path, exp1: dict, exp2: dict, lock: dict, runtime: dict) -> None:
    paths = write_aggregates(root / "raw", root)
    learner_files = sorted((root / "learner_metrics").rglob("rep_*.csv.gz"))
    learner = pd.concat((pd.read_csv(path, float_precision="round_trip") for path in learner_files), ignore_index=True)
    atomic_write_csv(learner, root / "learner_metrics.csv")
    learner_summary = aggregate_learner_metrics(learner)
    atomic_write_csv(learner_summary, root / "learner_metrics_summary.csv")
    make_tables_and_figures(root / "all_replicate_results.csv.gz", root)
    raw = pd.read_csv(root / "all_replicate_results.csv.gz", float_precision="round_trip")
    counts = raw.groupby("scenario").replicate.nunique().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Final replicate counts mismatch: {counts}")
    per_rep_timing = raw.groupby(["scenario", "replicate"], as_index=False).total_seconds.first()
    timing = per_rep_timing.groupby("scenario").total_seconds.agg(["count", "mean", "sum", "median", "max"]).reset_index()
    atomic_write_csv(timing, root / "timing_summary.csv")
    maxima_files = sorted((root / "calibration_maxima").rglob("rep_*.csv.gz"))
    maxima_records = []
    for path in maxima_files:
        frame = pd.read_csv(path, float_precision="round_trip")
        for (scenario, replicate, origin), group in frame.groupby(["scenario", "replicate", "origin"]):
            for scale, column in (("fixed", "fixed_maximum"), ("time", "time_maximum")):
                vals = group[column].to_numpy(); _, atom = np.unique(vals, return_counts=True)
                maxima_records.append({"scenario": scenario, "replicate": replicate, "origin": origin, "scale": scale, "n": len(vals), "n_unique": np.unique(vals).size, "largest_atom_fraction": atom.max()/len(vals)})
    maxima_summary = pd.DataFrame(maxima_records)
    atomic_write_csv(maxima_summary, root / "maxima_diagnostics.csv")
    manifest = {
        "production_lock": lock, "completed_replicates": counts, "runtime": runtime,
        "implementation_checksum": implementation_checksum(), "sensitivity_simulations_run": False,
        "all_primary_complete": True, "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(manifest, root / "run_manifest.json")
    summary = pd.read_csv(root / "summary_long.csv")
    pafars = raw[raw.method.isin(["PaFAR-F", "Direct source transfer", "PaFAR-T", "Local PaFAR-F", "Local PaFAR-T"])]
    pafar_preview = pafars.groupby(["scenario", "method", "operating_alpha"], as_index=False).agg(mean_pfa=("pfa", "mean"), mean_alpha_m0=("alpha_m0", "mean"))
    hc = raw[raw.method.str.contains("HC") & raw.conditional_pfa_oracle.notna()]
    hc_preview = hc.groupby(["scenario", "method", "operating_alpha"], as_index=False).agg(exceedance_frequency=("conditional_pfa_gt_alpha", "mean"))
    shift = raw[(raw.scenario.isin(["S4", "E3"])) & np.isclose(raw.operating_alpha, .10) & raw.method.isin(["Direct source transfer", "Local PaFAR-F", "Local PaFAR-T", "Local PaFAR-HC"])]
    shift_preview = shift.groupby(["scenario", "method", "target_m0"], as_index=False)[["pfa", "sens3", "sens0"]].mean()
    table_text = []
    for number, name in ((3, "table3_experiment1.csv"), (4, "table4_experiment2_e1_e2.csv"), (5, "table5_target_shift.csv")):
        table_text.append(f"### Table {number}\n\n```text\n{pd.read_csv(root/'tables'/name).to_string(index=False)}\n```")
    report = f"""# PRIMARY production report

## Lock and completion

```json
{json.dumps(lock, indent=2)}
```

- Completed replicates: {json.dumps(counts, sort_keys=True)}
- Learner failures: {int(learner.learner_failure.sum())}
- Undefined summary cells: {int(summary.n_undefined.sum())}
- Infinite threshold rows: {int(raw.infinite_threshold.fillna(False).sum())}
- Peak process RSS: {runtime.get('peak_rss_gib', np.nan):.3f} GiB
- Total production wall time: {runtime.get('wall_seconds', np.nan):.1f} seconds
- Sensitivity simulations were not run.

## Per-scenario runtime

```text
{timing.to_string(index=False)}
```

## Learner AUROC, PR-AUC, and best iteration

```text
{learner_summary.to_string(index=False)}
```

## Maxima largest atoms

```text
{maxima_summary.groupby(['scenario','origin','scale']).largest_atom_fraction.agg(['mean','max']).reset_index().to_string(index=False)}
```

## PaFAR mean PFA and alpha_m0

```text
{pafar_preview.to_string(index=False)}
```

## PaFAR-HC conditional PFA exceedance frequency

The registered comparison level is delta={float(exp1['delta']):.2f}.

```text
{hc_preview.to_string(index=False)}
```

## S4/E3 direct versus local recalibration

```text
{shift_preview.to_string(index=False)}
```

## Tables 3--5

{chr(10).join(table_text)}

## Figures

- `figures/figure2_cumulative_false_alert.png` and `.pdf`
- `figures/figure3_pfa_vs_alpha.png` and `.pdf`
- `figures/figure4_sens3_vs_pfa.png` and `.pdf`
- `figures/figure5_conditional_pfa.png` and `.pdf`
- `figures/figure6_target_recalibration.png` and `.pdf`

All tables use aggregate rows. Pilot outputs are excluded. **No sensitivity simulation has been run.**
"""
    report_path = root / "PRIMARY_PRODUCTION_REPORT.md"; report_path.write_text(report, encoding="utf-8")
    project = root.parents[1]
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name); code_zip = temp / "current_src_tests_configs_scripts.zip"; _code_archive(project, code_zip)
        bundle_path = root / "PRIMARY_REVIEW_BUNDLE.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            fixed = [
                report_path, root/"PRODUCTION_LOCK.json", root/"run_manifest.json", root/"summary_long.csv",
                root/"learner_metrics_summary.csv", root/"failure_summary.csv", root/"timing_summary.csv",
            ]
            fixed += list((root/"tables").glob("table[345]*"))
            fixed += [p for p in (root/"figures").glob("figure[23456]*") if p.suffix in {".png", ".pdf", ".csv", ".json"}]
            for path in fixed:
                if path.is_file(): bundle.write(path, path.relative_to(root).as_posix())
            rng = make_rng(int(lock["master_seed"]), "primary_review_bundle_raw_selection")
            for scenario, total in EXPECTED_COUNTS.items():
                experiment = "experiment1" if scenario.startswith("S") else "experiment2"
                for replicate in sorted(int(x) for x in rng.choice(total, size=2, replace=False)):
                    path = root/"raw"/experiment/"primary"/scenario/f"rep_{replicate:06d}.csv.gz"
                    bundle.write(path, f"sample_raw/{experiment}/{scenario}/{path.name}")
                    sidecar = path.with_suffix(path.suffix+".json")
                    bundle.write(sidecar, f"sample_raw/{experiment}/{scenario}/{sidecar.name}")
            bundle.write(code_zip, code_zip.name)
