"""Replicate runner for Experiment I."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..alerting import first_alert, hourly_bin_thresholds
from ..calibration import (
    ThresholdResult, binwise_bonferroni_thresholds, fit_time_template, hc_threshold,
    marginal_threshold, naive_maximum_threshold, pointwise_threshold,
)
from ..io_utils import checkpoint_complete, json_value, write_checkpoint
from ..metrics import cumulative_false_alert_curve, evaluate_metrics, standardized_from_components
from ..output_schema import describe_method
from ..rng import make_rng, seed_record
from ..score import causal_moving_average, clipped_logit, trajectory_max
from .dgp import Exp1Batch, generate_exp1
from .oracle import build_oracle, conditional_pfa, load_oracle, oracle_filename, oracle_threshold


def _locked_score(batch: Exp1Batch, length: int) -> np.ndarray:
    return clipped_logit(causal_moving_average(batch.risk, length))


def _slice_batch(batch: Exp1Batch, size: int) -> Exp1Batch:
    """Take a nested patient prefix without copying unrelated trajectories."""
    return Exp1Batch(*(getattr(batch, field)[:size] for field in batch.__dataclass_fields__))


def _threshold_meta(result: ThresholdResult | None, threshold: float, default_m0: int | float = np.nan) -> dict[str, Any]:
    return {
        "threshold": threshold, "threshold_index": result.index if result else np.nan,
        "m0": result.m0 if result else default_m0,
        "alpha_m0": result.alpha_m0 if result else np.nan,
        "infinite_threshold": bool(np.isposinf(threshold)),
    }


def _evaluate_method(
    *, method: str, score_non: np.ndarray, score_event: np.ndarray,
    non: Exp1Batch, events: Exp1Batch, threshold: float | np.ndarray,
    threshold_result: ThresholdResult | None, base: dict[str, Any],
    oracle_maxima: np.ndarray, scalar_for_record: float | None = None,
    template_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = np.vstack((score_non, score_event))
    eligible = np.vstack((non.eligible, events.eligible))
    event = np.r_[non.event, events.event]
    onset = np.r_[non.onset, events.onset]
    horizon = np.r_[non.horizon, events.horizon]
    ids = np.r_[non.patient_ids, events.patient_ids]
    prevalences = [float(x) for x in base.get("standardization_prevalences", [0.10, 0.05])]
    metrics = evaluate_metrics(score, eligible, threshold, event, onset, horizon, ids, prevalence=prevalences[0])
    tau_non = first_alert(score_non, non.eligible, threshold)
    curve = cumulative_false_alert_curve(tau_non, non.horizon, score_non.shape[1])
    scalar = float(threshold) if np.ndim(threshold) == 0 else np.nan
    conditional = conditional_pfa(oracle_maxima, scalar) if method in {"PaFAR-F", "PaFAR-HC", "Local PaFAR-F", "Local PaFAR-HC", "Direct source transfer"} and np.isfinite(scalar) else (0.0 if np.isposinf(scalar) else np.nan)
    row = {
        **base, "method": method, **describe_method(method, str(base["scenario"])),
        **_threshold_meta(threshold_result, scalar if scalar_for_record is None else scalar_for_record, base.get("m0", np.nan)),
        **metrics.as_dict(), "conditional_pfa_oracle": conditional,
        "cumulative_false_alert_curve": json_value(curve),
        "conditional_pfa_gt_alpha": bool(conditional > base["alpha"]) if np.isfinite(conditional) else False,
        "threshold_vector": json_value(threshold.tolist()) if np.ndim(threshold) else "",
        "template_fit_source": "validation_non_events" if template_meta else "",
        "threshold_fit_source": (
            "target_calibration_non_events" if method.startswith("Local") else
            ("independent_oracle_reference" if method == "Oracle-F" else "calibration_non_events")
        ),
        "template_boundaries": json_value(template_meta.get("boundaries", [])) if template_meta else "",
        "template_locations": json_value(template_meta.get("locations", [])) if template_meta else "",
        "template_scales": json_value(template_meta.get("scales", [])) if template_meta else "",
        "raw_bin_boundaries": json_value(template_meta.get("raw_boundaries", [])) if template_meta else "",
        "learner_failure": False, "failure_reason": "", "aucpr_weighted": np.nan, "auroc_weighted": np.nan,
    }
    for prevalence in prevalences[1:]:
        suffix = f"pi{int(round(prevalence * 1000)):03d}"
        standardized = standardized_from_components(metrics, prevalence)
        row[f"ppv3_standardized_{suffix}"] = standardized["ppv3"]
        row[f"ppv0_standardized_{suffix}"] = standardized["ppv0"]
        row[f"alert_burden_100d_{suffix}"] = standardized["alert_burden_100d"]
    return row


def _standard_methods(
    val: Exp1Batch, cal: Exp1Batch, non: Exp1Batch, events: Exp1Batch,
    oracle_maxima: np.ndarray, base: dict[str, Any], smooth_length: int, tmin: int, hmax: int,
) -> list[dict[str, Any]]:
    alpha, delta = float(base["alpha"]), float(base["delta"])
    u_val, u_cal, u_non, u_event = (_locked_score(x, smooth_length) for x in (val, cal, non, events))
    template = fit_time_template(u_val, val.eligible, tmin, hmax)
    z_cal, z_non, z_event = (template.transform(x) for x in (u_cal, u_non, u_event))
    f_max = trajectory_max(u_cal, cal.eligible)
    t_max = trajectory_max(z_cal, cal.eligible)
    f = marginal_threshold(f_max, alpha)
    t_result = marginal_threshold(t_max, alpha)
    hc = hc_threshold(f_max, alpha, delta)
    oracle_c, oracle_k = oracle_threshold(oracle_maxima, alpha)
    point = pointwise_threshold(u_cal, cal.eligible, alpha)
    naive = naive_maximum_threshold(f_max, alpha)
    bounds, bin_c = binwise_bonferroni_thresholds(u_cal, cal.eligible, tmin, hmax, alpha)
    hourly_c = hourly_bin_thresholds(hmax, bounds, bin_c)
    template_meta = {
        "boundaries": template.boundaries, "locations": template.locations, "scales": template.scales,
        "raw_boundaries": (np.asarray(template.locations) + t_result.threshold * np.asarray(template.scales)).tolist(),
    }
    f_method = "Direct source transfer" if str(base["scenario"]) == "S4" else "PaFAR-F"
    specs = [
        ("Pointwise-alpha", u_non, u_event, point, None, None),
        ("Binwise Bonferroni", u_non, u_event, hourly_c, None, {"boundaries": bounds, "raw_boundaries": bin_c.tolist()}),
        ("Naive maximum", u_non, u_event, naive, None, None),
        (f_method, u_non, u_event, f.threshold, f, None),
        ("PaFAR-T", z_non, z_event, t_result.threshold, t_result, template_meta),
        ("PaFAR-HC", u_non, u_event, hc.threshold, hc, None),
        ("Oracle-F", u_non, u_event, oracle_c, ThresholdResult(oracle_c, oracle_k, oracle_maxima.size, alpha, np.nan), None),
    ]
    return [
        _evaluate_method(method=m, score_non=sn, score_event=se, non=non, events=events,
                         threshold=c, threshold_result=result, base=base, oracle_maxima=oracle_maxima,
                         template_meta=meta)
        for m, sn, se, c, result, meta in specs
    ]


def run_exp1_replicate(
    config: dict[str, Any], config_checksum: str, scenario: str, replicate: int,
    output_root: str | Path, resume: bool = False,
) -> Path:
    """Run one deterministic Experiment I replicate and atomically checkpoint it."""
    root = Path(output_root)
    condition = str(config.get("condition", "primary"))
    checkpoint = root / "raw" / "experiment1" / condition / scenario / f"rep_{replicate:06d}.csv.gz"
    if resume and checkpoint_complete(checkpoint, config_checksum):
        return checkpoint
    started = time.perf_counter()
    alpha, delta = float(config["alpha"]), float(config["delta"])
    alpha_grid = sorted(set(float(x) for x in config.get("alpha_grid", [alpha])) | {alpha})
    tmin, hmax, length = int(config["tmin"]), int(config["hmax"]), int(config["smooth_length"])
    exp = config["experiment1"]
    signal = float(exp.get("signal", 1.5))
    seeds = seed_record(int(config["master_seed"]), "experiment1", scenario, replicate, condition)
    site = "B" if scenario == "S4" else "A"
    oracle_name = oracle_filename(
        scenario, site, int(exp["oracle_nref"]), int(config["master_seed"]),
        hmax=hmax, tmin=tmin, smooth_length=length,
    )
    oracle_path = root / "oracle" / oracle_name
    if oracle_path.exists():
        oracle_maxima = load_oracle(
            oracle_path, scenario, site, int(exp["oracle_nref"]), int(config["master_seed"]),
            hmax=hmax, tmin=tmin, smooth_length=length,
        )
    else:
        oracle_maxima = build_oracle(
            oracle_path, scenario, site, int(exp["oracle_nref"]), int(config["master_seed"]),
            hmax=hmax, tmin=tmin, smooth_length=length, chunk_size=int(exp.get("oracle_chunk_size", 25000)),
        )
    dgp_started = time.perf_counter()
    val = generate_exp1(make_rng(seeds["validation"], "batch"), int(exp["nvalidation_non_events"]), scenario, False, site="A", hmax=hmax, tmin=tmin, signal=signal)
    calibration_sizes = [int(x) for x in exp.get("calibration_sizes", [exp["ncalibration_non_events"]])]
    calibration_n = max(calibration_sizes)
    cal = generate_exp1(make_rng(seeds["calibration"], "batch"), calibration_n, scenario, False, site="A", hmax=hmax, tmin=tmin, signal=signal)
    non = generate_exp1(make_rng(seeds["test_non_events"], "batch"), int(exp["ntest_non_events"]), scenario, False, site=site, hmax=hmax, tmin=tmin, signal=signal)
    events = generate_exp1(make_rng(seeds["test_events"], "batch"), int(exp["ntest_events"]), scenario, True, site=site, hmax=hmax, tmin=tmin, signal=signal, patient_id_start=len(non.patient_ids))
    dgp_seconds = time.perf_counter() - dgp_started
    base = {
        "experiment": "Experiment I", "condition": condition, "scenario": scenario, "site": site, "replicate": replicate,
        "alpha": alpha, "delta": delta, "signal_strength": float(exp.get("signal", 1.5)),
        "target_m0": np.nan, "config_checksum": config_checksum,
        "m0": int(exp["ncalibration_non_events"]),
        "master_seed": int(config["master_seed"]), "replicate_seed": f"u64:{seeds['replicate']}",
        "seed_record": json_value(seeds), "dgp_seconds": dgp_seconds,
        "standardization_prevalences": tuple(config.get("standardization_prevalences", [0.10, 0.05])),
    }
    calibration_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for calibration_m0 in calibration_sizes:
        for operating_alpha in alpha_grid:
            calibration_base = {
                **base, "alpha": operating_alpha, "operating_alpha": operating_alpha,
                "operating_point": "primary" if operating_alpha == alpha else "alpha_grid",
                "m0": calibration_m0, "calibration_m0": calibration_m0,
                "target_m0": 0 if scenario == "S4" else np.nan,
            }
            rows.extend(_standard_methods(val, _slice_batch(cal, calibration_m0), non, events, oracle_maxima,
                                          calibration_base, length, tmin, hmax))
    calibration_seconds = time.perf_counter() - calibration_started

    if scenario == "S4":
        reservoir_size = int(exp["target_reservoir"])
        reservoir = generate_exp1(make_rng(seeds["target_reservoir"], "batch"), reservoir_size, scenario, False, site="B", hmax=hmax, tmin=tmin, signal=signal)
        u_val = _locked_score(val, length)
        u_res, u_non, u_event = (_locked_score(x, length) for x in (reservoir, non, events))
        template = fit_time_template(u_val, val.eligible, tmin, hmax)
        z_res, z_non, z_event = (template.transform(x) for x in (u_res, u_non, u_event))
        for operating_alpha in alpha_grid:
            for prefix in exp["target_prefixes"]:
                m = int(prefix)
                fmax = trajectory_max(u_res[:m], reservoir.eligible[:m])
                tmax = trajectory_max(z_res[:m], reservoir.eligible[:m])
                for method, sn, se, result in (
                    ("Local PaFAR-F", u_non, u_event, marginal_threshold(fmax, operating_alpha)),
                    ("Local PaFAR-T", z_non, z_event, marginal_threshold(tmax, operating_alpha)),
                    ("Local PaFAR-HC", u_non, u_event, hc_threshold(fmax, operating_alpha, delta)),
                ):
                    local_base = {
                        **base, "alpha": operating_alpha, "operating_alpha": operating_alpha,
                        "operating_point": "primary" if operating_alpha == alpha else "alpha_grid",
                        "target_m0": m,
                    }
                    local_meta = None
                    if method == "Local PaFAR-T":
                        local_meta = {
                            "boundaries": template.boundaries, "locations": template.locations,
                            "scales": template.scales,
                            "raw_boundaries": (np.asarray(template.locations) + result.threshold * np.asarray(template.scales)).tolist(),
                        }
                    rows.append(_evaluate_method(method=method, score_non=sn, score_event=se, non=non, events=events,
                                threshold=result.threshold, threshold_result=result, base=local_base, oracle_maxima=oracle_maxima,
                                template_meta=local_meta))
    total_seconds = time.perf_counter() - started
    for row in rows:
        row.pop("standardization_prevalences", None)
        row.update({"calibration_seconds": calibration_seconds, "feature_seconds": 0.0,
                    "learner_fit_seconds": 0.0, "metric_seconds": np.nan, "total_seconds": total_seconds})
    source_cal = _slice_batch(cal, int(exp["ncalibration_non_events"]))
    u_val = _locked_score(val, length)
    u_source = _locked_score(source_cal, length)
    source_template = fit_time_template(u_val, val.eligible, tmin, hmax)
    source_maxima = pd.DataFrame({
        "experiment": "Experiment I", "condition": condition, "scenario": scenario,
        "replicate": replicate, "origin": "source_calibration_non_events",
        "patient_order": np.arange(len(source_cal.patient_ids)),
        "fixed_maximum": trajectory_max(u_source, source_cal.eligible),
        "time_maximum": trajectory_max(source_template.transform(u_source), source_cal.eligible),
    })
    if scenario == "S4":
        u_target = _locked_score(reservoir, length)
        target_maxima = pd.DataFrame({
            "experiment": "Experiment I", "condition": condition, "scenario": scenario,
            "replicate": replicate, "origin": "target_calibration_non_events",
            "patient_order": np.arange(len(reservoir.patient_ids)),
            "fixed_maximum": trajectory_max(u_target, reservoir.eligible),
            "time_maximum": trajectory_max(source_template.transform(u_target), reservoir.eligible),
        })
        source_maxima = pd.concat([source_maxima, target_maxima], ignore_index=True)
    maxima_checkpoint = root / "calibration_maxima" / "experiment1" / condition / scenario / f"rep_{replicate:06d}.csv.gz"
    write_checkpoint(source_maxima, maxima_checkpoint, config_checksum)
    write_checkpoint(pd.DataFrame(rows), checkpoint, config_checksum)
    return checkpoint


def run_experiment1(
    config: dict[str, Any], config_checksum: str, scenarios: list[str], replicate_start: int,
    replicate_end: int, output_root: str | Path, n_jobs: int = 1, resume: bool = False,
) -> list[Path]:
    """Run an inclusive replicate range with order-independent loky workers."""
    tasks = [(scenario, rep) for scenario in scenarios for rep in range(replicate_start, replicate_end + 1)]
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_exp1_replicate)(config, config_checksum, scenario, rep, output_root, resume)
        for scenario, rep in tasks
    )
