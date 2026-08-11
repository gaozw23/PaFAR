"""Across-replicate summaries that preserve undefined estimates and failures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv


METRIC_COLUMNS = (
    "pfa", "long_stay_pfa", "sens3", "sens0", "premature", "median_lead",
    "ppv3_standardized", "ppv0_standardized", "alert_burden_100d",
    "alert_episodes_per_patient", "conditional_pfa_oracle", "conditional_pfa_gt_alpha",
    "any_alert_rate_non_event", "any_alert_rate_event",
    "valid3_rate_all_events", "valid0_rate_all_events",
    "mean_episodes_non_event", "mean_episodes_event",
    "mean_exposure_days_non_event", "mean_exposure_days_event",
    "ppv3_standardized_pi050", "ppv0_standardized_pi050", "alert_burden_100d_pi050",
)


def aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Create long summaries with MCSE and explicit defined/undefined counts."""
    if "is_alias" in frame:
        frame = frame.loc[~frame["is_alias"].fillna(False).astype(bool)].copy()
    identifiers = [c for c in (
        "experiment", "condition", "scenario", "site", "method", "alpha", "operating_alpha",
        "operating_point", "target_m0", "calibration_m0",
    ) if c in frame]
    records: list[dict[str, object]] = []
    for keys, group in frame.groupby(identifiers, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(identifiers, key_values))
        failures = int(group.get("learner_failure", pd.Series(False, index=group.index)).fillna(False).sum())
        failure_frequency = failures / len(group) if len(group) else np.nan
        infinite_frequency = float(group.get("infinite_threshold", pd.Series(False, index=group.index)).fillna(False).astype(bool).mean())
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            defined = values.dropna()
            records.append({
                **base, "metric": metric,
                "mean": float(defined.mean()) if len(defined) else np.nan,
                "mcse": float(defined.std(ddof=1) / np.sqrt(len(defined))) if len(defined) > 1 else np.nan,
                "n_total": int(len(values)), "n_defined": int(len(defined)),
                "n_undefined": int(values.isna().sum()), "n_failures": failures,
                "learner_failure_frequency": failure_frequency,
                "infinite_threshold_frequency": infinite_frequency,
            })
        if "threshold" in group:
            thresholds = pd.to_numeric(group["threshold"], errors="coerce")
            finite = thresholds[np.isfinite(thresholds)]
            records.append({
                **base, "metric": "threshold", "mean": float(finite.mean()) if len(finite) else np.nan,
                "mcse": np.nan,
                "threshold_mean": float(finite.mean()) if len(finite) else np.nan,
                "threshold_sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "n_total": int(len(thresholds)), "n_defined": int(len(finite)),
                "n_undefined": int((~np.isfinite(thresholds)).sum()), "n_failures": failures,
                "finite_count": int(len(finite)), "infinite_frequency": float(np.isposinf(thresholds).mean()),
                "learner_failure_frequency": failure_frequency,
                "infinite_threshold_frequency": infinite_frequency,
            })
    return pd.DataFrame(records)


def aggregate_learner_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each learner replicate exactly once."""
    identifiers = [c for c in ("experiment", "condition", "scenario", "site", "score_generator") if c in frame]
    records = []
    for keys, group in frame.groupby(identifiers, dropna=False, sort=True):
        base = dict(zip(identifiers, keys if isinstance(keys, tuple) else (keys,)))
        failures = group.get("learner_failure", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        auroc = pd.to_numeric(group.get("auroc_weighted"), errors="coerce")
        pr = pd.to_numeric(group.get("aucpr_weighted"), errors="coerce")
        iteration = pd.to_numeric(group.get("best_iteration"), errors="coerce")
        def mean_mcse(values: pd.Series) -> tuple[float, float]:
            x = values.dropna()
            return (
                float(x.mean()) if len(x) else np.nan,
                float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            )
        roc_mean, roc_mcse = mean_mcse(auroc)
        pr_mean, pr_mcse = mean_mcse(pr)
        valid_iteration = iteration.dropna()
        records.append({
            **base, "n_total": len(group), "auroc_mean": roc_mean, "auroc_mcse": roc_mcse,
            "pr_auc_mean": pr_mean, "pr_auc_mcse": pr_mcse,
            "best_iteration_mean": float(valid_iteration.mean()) if len(valid_iteration) else np.nan,
            "best_iteration_sd": float(valid_iteration.std(ddof=1)) if len(valid_iteration) > 1 else np.nan,
            "best_iteration_q05": float(valid_iteration.quantile(.05)) if len(valid_iteration) else np.nan,
            "best_iteration_median": float(valid_iteration.median()) if len(valid_iteration) else np.nan,
            "best_iteration_q95": float(valid_iteration.quantile(.95)) if len(valid_iteration) else np.nan,
            "learner_failure_frequency": float(failures.mean()),
        })
    return pd.DataFrame(records)


def collect_checkpoints(raw_root: str | Path) -> pd.DataFrame:
    """Read all completed replicate CSV files below a raw output directory."""
    files = sorted(Path(raw_root).rglob("rep_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No replicate checkpoints below {raw_root}")
    return pd.concat((pd.read_csv(path, float_precision="round_trip") for path in files), ignore_index=True)


def write_aggregates(raw_root: str | Path, output_root: str | Path) -> dict[str, Path]:
    """Collect raw results and write required aggregate tables."""
    out = Path(output_root)
    frame = collect_checkpoints(raw_root)
    summary = aggregate_results(frame)
    failure = frame.loc[frame.get("learner_failure", False).fillna(False)] if "learner_failure" in frame else frame.iloc[0:0]
    timing_cols = [c for c in frame if c.endswith("_seconds")]
    timing = frame.groupby([c for c in ("experiment", "scenario") if c in frame], dropna=False)[timing_cols].mean().reset_index() if timing_cols else pd.DataFrame()
    paths = {
        "all": out / "all_replicate_results.csv.gz", "summary": out / "summary_long.csv",
        "failure": out / "failure_summary.csv", "timing": out / "timing_summary.csv",
    }
    atomic_write_csv(frame, paths["all"]); atomic_write_csv(summary, paths["summary"])
    atomic_write_csv(failure, paths["failure"]); atomic_write_csv(timing, paths["timing"])
    learner_root = Path(raw_root).parent / "learner_metrics"
    learner_files = sorted(learner_root.rglob("rep_*.csv.gz")) if learner_root.exists() else []
    if learner_files:
        learner = pd.concat((pd.read_csv(path, float_precision="round_trip") for path in learner_files), ignore_index=True)
        key = [c for c in ("scenario", "condition", "site", "replicate", "score_generator") if c in learner]
        if key and learner.duplicated(key).any():
            raise ValueError("learner metrics contain duplicate replicate keys")
        paths["learner_summary"] = out / "learner_metrics_summary.csv"
        atomic_write_csv(aggregate_learner_metrics(learner), paths["learner_summary"])
    return paths
