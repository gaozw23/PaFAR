"""Replicate runner for Experiment II."""
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
    marginal_threshold, naive_maximum_threshold, pointwise_threshold, youden_threshold,
)
from ..io_utils import checkpoint_complete, json_value, write_checkpoint
from ..metrics import cumulative_false_alert_curve, evaluate_metrics, standardized_from_components, threshold_free_metrics
from ..output_schema import describe_method
from ..rng import make_rng, seed_record
from ..score import causal_moving_average, clipped_logit, trajectory_max
from .dgp import Exp2Batch, NonEventSamplingInfo, generate_exp2, generate_exp2_non_events
from .features import FeaturePreprocessor, RawFeatureRows, build_raw_features, score_history_mask
from .learner import FittedLearner, fit_xgboost


METHODS = (
    "Fixed 0.5", "Youden", "Pointwise-alpha", "Binwise Bonferroni",
    "Naive maximum", "PaFAR-F", "PaFAR-T", "PaFAR-HC",
)


def _prediction_matrix(prediction: np.ndarray, rows: RawFeatureRows, n: int, hmax: int) -> np.ndarray:
    matrix = np.full((n, hmax), np.nan, dtype=np.float64)
    matrix[rows.patient_index, rows.hour - 1] = prediction
    return matrix


def _predict_batch(
    learner: FittedLearner, preprocessor: FeaturePreprocessor, batch: Exp2Batch,
    tmin: int, smooth_length: int,
) -> tuple[RawFeatureRows, np.ndarray, float]:
    """Predict all locked rows needed by the smoother, including pre-burn history."""
    started = time.perf_counter()
    rows = build_raw_features(batch, score_history_mask(batch, tmin, smooth_length))
    prediction = learner.predict(preprocessor.transform(rows))
    return rows, _prediction_matrix(prediction, rows, len(batch.patient_ids), batch.measurements.shape[1]), time.perf_counter() - started


def _eligible_flat(score: np.ndarray, batch: Exp2Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient, hour0 = np.nonzero(batch.eligible)
    values = score[patient, hour0]
    labels = (
        batch.event[patient]
        & ((batch.onset[patient] - (hour0 + 1)) > 0)
        & ((batch.onset[patient] - (hour0 + 1)) <= 6)
    ).astype(int)
    return values, labels, patient


def _metric_row(
    method: str, score: np.ndarray, threshold: float | np.ndarray, test: Exp2Batch,
    result: ThresholdResult | None, base: dict[str, Any],
    template_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prevalences = [float(x) for x in base.get("standardization_prevalences", [0.10, 0.05])]
    primary_prevalence = prevalences[0]
    metrics = evaluate_metrics(
        score, test.eligible, threshold, test.event, test.onset, test.horizon,
        test.patient_ids, prevalence=primary_prevalence,
    )
    non = ~test.event
    tau_non = first_alert(score[non], test.eligible[non], threshold)
    curve = cumulative_false_alert_curve(tau_non, test.horizon[non], score.shape[1])
    scalar = float(threshold) if np.ndim(threshold) == 0 else np.nan
    metadata = describe_method(method, str(base["scenario"]))
    target_m0 = 0 if method == "Direct source transfer" else base.get("target_m0", np.nan)
    threshold_source = "calibration_non_events"
    if method.startswith("Local"):
        threshold_source = "target_calibration_non_events"
    elif method == "Youden":
        threshold_source = "validation"
    elif method == "Fixed 0.5":
        threshold_source = "fixed"
    row = {
        **base, "target_m0": target_m0, "method": method, **metadata, "threshold": scalar,
        "threshold_index": result.index if result else np.nan,
        "m0": result.m0 if result else int(base.get("m0", 0)),
        "alpha_m0": result.alpha_m0 if result else np.nan,
        "infinite_threshold": bool(np.isposinf(scalar)),
        "threshold_vector": json_value(np.asarray(threshold).tolist()) if np.ndim(threshold) else "",
        "template_fit_source": "validation_non_events" if template_meta else "",
        "threshold_fit_source": threshold_source,
        "template_boundaries": json_value(template_meta.get("boundaries", [])) if template_meta else "",
        "template_locations": json_value(template_meta.get("locations", [])) if template_meta else "",
        "template_scales": json_value(template_meta.get("scales", [])) if template_meta else "",
        "raw_bin_boundaries": json_value(template_meta.get("raw_boundaries", [])) if template_meta else "",
        **metrics.as_dict(), "auroc_weighted": np.nan, "aucpr_weighted": np.nan,
        "cumulative_false_alert_curve": json_value(curve),
        "conditional_pfa_oracle": np.nan, "conditional_pfa_gt_alpha": False,
        "learner_failure": False, "failure_reason": "",
    }
    for prevalence in prevalences[1:]:
        suffix = f"pi{int(round(prevalence * 1000)):03d}"
        standardized = standardized_from_components(metrics, prevalence)
        row[f"ppv3_standardized_{suffix}"] = standardized["ppv3"]
        row[f"ppv0_standardized_{suffix}"] = standardized["ppv0"]
        row[f"alert_burden_100d_{suffix}"] = standardized["alert_burden_100d"]
    return row


def _failure_rows(base: dict[str, Any], reason: str) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        rows.append({
            **base, "operating_alpha": float(base["alpha"]), "operating_point": "primary",
            "method": method, **describe_method(method, str(base["scenario"])),
            "threshold": np.nan, "threshold_index": np.nan, "m0": np.nan,
            "alpha_m0": np.nan, "infinite_threshold": False,
            "pfa": np.nan, "long_stay_pfa": np.nan, "sens3": np.nan, "sens0": np.nan,
            "premature": np.nan, "median_lead": np.nan, "ppv3_standardized": np.nan,
            "ppv0_standardized": np.nan, "alert_burden_100d": np.nan,
            "alert_episodes_per_patient": np.nan, "auroc_weighted": np.nan, "aucpr_weighted": np.nan,
            "conditional_pfa_oracle": np.nan, "conditional_pfa_gt_alpha": False,
            "learner_failure": True, "failure_reason": reason,
        })
    return rows


def _alpha_grid(config: dict[str, Any]) -> list[float]:
    values = [float(x) for x in config.get("alpha_grid", [config["alpha"]])]
    primary = float(config["alpha"])
    return sorted(set(values + [primary]))


def run_exp2_replicate(
    config: dict[str, Any], config_checksum: str, scenario: str, replicate: int,
    output_root: str | Path, resume: bool = False,
) -> Path:
    """Run one Experiment II replicate without resampling on learner failure."""
    root = Path(output_root)
    condition = str(config.get("condition", "primary"))
    checkpoint = root / "raw" / "experiment2" / condition / scenario / f"rep_{replicate:06d}.csv.gz"
    if resume and checkpoint_complete(checkpoint, config_checksum):
        return checkpoint
    total_started = time.perf_counter()
    exp = config["experiment2"]
    primary_alpha, delta = float(config["alpha"]), float(config["delta"])
    tmin, hmax, length = int(config["tmin"]), int(config["hmax"]), int(config["smooth_length"])
    seeds = seed_record(int(config["master_seed"]), "experiment2", scenario, replicate, condition)
    base = {
        "experiment": "Experiment II", "condition": condition, "scenario": scenario,
        "site": "B" if scenario == "E3" else "A", "replicate": replicate,
        "alpha": primary_alpha, "delta": delta, "signal_strength": float(exp.get("delta_s", 1.0)),
        "target_m0": np.nan, "config_checksum": config_checksum,
        "master_seed": int(config["master_seed"]), "replicate_seed": f"u64:{seeds['replicate']}",
        "seed_record": json_value(seeds),
        "standardization_prevalences": tuple(config.get("standardization_prevalences", [0.10, 0.05])),
    }
    common_dgp = dict(hmax=hmax, tmin=tmin, delta_s=float(exp.get("delta_s", 1.0)))
    dgp_started = time.perf_counter()
    train = generate_exp2(make_rng(seeds["training"], "batch"), int(exp["ntrain"]), scenario, site="A", **common_dgp)
    validation = generate_exp2(make_rng(seeds["validation"], "batch"), int(exp["nvalidation"]), scenario, site="A", **common_dgp)
    calibration = generate_exp2(make_rng(seeds["calibration"], "batch"), int(exp["ncalibration"]), scenario, site="A", **common_dgp)
    test_site = "B" if scenario == "E3" else "A"
    test_n = int(exp.get("target_test_patients", exp["ntest"])) if scenario == "E3" else int(exp["ntest"])
    test = generate_exp2(make_rng(seeds["test_non_events"], "natural_batch"), test_n, scenario, site=test_site, **common_dgp)
    reservoir: Exp2Batch | None = None
    reservoir_info: NonEventSamplingInfo | None = None
    if scenario == "E3":
        reservoir, reservoir_info = generate_exp2_non_events(
            make_rng(seeds["target_reservoir"], "candidates"), int(exp["target_reservoir"]), scenario,
            order_rng=make_rng(seeds["target_reservoir_order"], "order"), site="B", **common_dgp,
        )
    dgp_seconds = time.perf_counter() - dgp_started

    feature_started = time.perf_counter()
    train_rows = build_raw_features(train, train.eligible)
    validation_fit_rows = build_raw_features(validation, validation.eligible)
    preprocessor = FeaturePreprocessor.fit(train_rows)
    train_x = preprocessor.transform(train_rows)
    validation_x = preprocessor.transform(validation_fit_rows)
    feature_seconds = time.perf_counter() - feature_started
    try:
        learner, pi_y_hat = fit_xgboost(
            train_x, train_rows.labels, train_rows.patient_index,
            validation_x, validation_fit_rows.labels, validation_fit_rows.patient_index,
            seed=seeds["xgboost"], num_boost_round=int(exp["num_boost_round"]),
            early_stopping_rounds=int(exp["early_stopping_rounds"]),
        )
    except (ValueError, RuntimeError) as exc:
        rows = _failure_rows({**base, "dgp_seconds": dgp_seconds, "feature_seconds": feature_seconds}, str(exc))
        elapsed = time.perf_counter() - total_started
        for row in rows:
            row.update({"learner_fit_seconds": 0.0, "calibration_seconds": 0.0, "metric_seconds": 0.0, "total_seconds": elapsed})
        write_checkpoint(pd.DataFrame(rows), checkpoint, config_checksum)
        learner_checkpoint = root / "learner_metrics" / "experiment2" / condition / scenario / f"rep_{replicate:06d}.csv.gz"
        write_checkpoint(pd.DataFrame([{
            "experiment": "Experiment II", "condition": condition, "scenario": scenario,
            "site": test_site, "replicate": replicate, "score_generator": "xgboost",
            "auroc_weighted": np.nan, "aucpr_weighted": np.nan,
            "best_iteration": np.nan, "prediction_iteration_end": np.nan, "best_score": np.nan,
            "learner_failure": True, "failure_reason": str(exc), "config_checksum": config_checksum,
            "master_seed": int(config["master_seed"]), "replicate_seed": f"u64:{seeds['replicate']}",
        }]), learner_checkpoint, config_checksum)
        return checkpoint

    validation_history_rows, validation_matrix, val_feature_time = _predict_batch(learner, preprocessor, validation, tmin, length)
    calibration_rows, calibration_matrix, cal_feature_time = _predict_batch(learner, preprocessor, calibration, tmin, length)
    test_rows, test_matrix, test_feature_time = _predict_batch(learner, preprocessor, test, tmin, length)
    feature_seconds += val_feature_time + cal_feature_time + test_feature_time
    reservoir_matrix = None
    if reservoir is not None:
        _, reservoir_matrix, reservoir_feature_time = _predict_batch(learner, preprocessor, reservoir, tmin, length)
        feature_seconds += reservoir_feature_time

    smooth_validation = causal_moving_average(validation_matrix, length)
    smooth_calibration = causal_moving_average(calibration_matrix, length)
    smooth_test = causal_moving_average(test_matrix, length)
    u_validation, u_calibration, u_test = map(clipped_logit, (smooth_validation, smooth_calibration, smooth_test))
    val_non = ~validation.event
    cal_non = ~calibration.event
    realized_m0 = int(cal_non.sum())
    template = fit_time_template(u_validation[val_non], validation.eligible[val_non], tmin, hmax)
    z_calibration = template.transform(u_calibration)
    z_test = template.transform(u_test)
    f_max = trajectory_max(u_calibration[cal_non], calibration.eligible[cal_non])
    t_max = trajectory_max(z_calibration[cal_non], calibration.eligible[cal_non])
    val_risk, val_labels, val_patient = _eligible_flat(smooth_validation, validation)
    youden = youden_threshold(val_risk, val_labels, val_patient)
    test_risk, test_labels, test_patient = _eligible_flat(smooth_test, test)
    auroc, aucpr = threshold_free_metrics(test_risk, test_labels, test_patient)
    common = {
        **base, "m0": realized_m0, "realized_calibration_non_events": realized_m0,
        "pi_y_hat": pi_y_hat, "best_iteration": learner.best_iteration,
        "prediction_iteration_end": learner.prediction_end, "best_score": learner.best_score,
        "score_history_first_hour": max(1, tmin - length + 1),
        "target_reservoir_candidates": reservoir_info.candidates if reservoir_info else np.nan,
        "target_reservoir_acceptance_rate": reservoir_info.acceptance_rate if reservoir_info else np.nan,
    }
    rows: list[dict[str, Any]] = []
    invariant_base = {**common, "alpha": primary_alpha, "operating_alpha": primary_alpha, "operating_point": "alpha_invariant"}
    rows.extend([
        _metric_row("Fixed 0.5", smooth_test, 0.5, test, None, invariant_base),
        _metric_row("Youden", smooth_test, youden, test, None, invariant_base),
    ])
    f_method = "Direct source transfer" if scenario == "E3" else "PaFAR-F"
    for operating_alpha in _alpha_grid(config):
        op_base = {
            **common, "alpha": operating_alpha, "operating_alpha": operating_alpha,
            "operating_point": "primary" if operating_alpha == primary_alpha else "alpha_grid",
        }
        f_result = marginal_threshold(f_max, operating_alpha)
        t_result = marginal_threshold(t_max, operating_alpha)
        hc_result = hc_threshold(f_max, operating_alpha, delta)
        point = pointwise_threshold(u_calibration[cal_non], calibration.eligible[cal_non], operating_alpha)
        naive = naive_maximum_threshold(f_max, operating_alpha)
        boundaries, bin_threshold = binwise_bonferroni_thresholds(
            u_calibration[cal_non], calibration.eligible[cal_non], tmin, hmax, operating_alpha,
        )
        hourly_threshold = hourly_bin_thresholds(hmax, boundaries, bin_threshold)
        template_meta = {
            "boundaries": template.boundaries, "locations": template.locations, "scales": template.scales,
            "raw_boundaries": (np.asarray(template.locations) + t_result.threshold * np.asarray(template.scales)).tolist(),
        }
        specs = [
            ("Pointwise-alpha", u_test, point, None, None),
            ("Binwise Bonferroni", u_test, hourly_threshold, None, {"boundaries": boundaries, "raw_boundaries": bin_threshold.tolist()}),
            ("Naive maximum", u_test, naive, None, None),
            (f_method, u_test, f_result.threshold, f_result, None),
            ("PaFAR-T", z_test, t_result.threshold, t_result, template_meta),
            ("PaFAR-HC", u_test, hc_result.threshold, hc_result, None),
        ]
        rows.extend(_metric_row(method, score, threshold, test, result, op_base, meta) for method, score, threshold, result, meta in specs)
        if scenario == "E3" and reservoir is not None and reservoir_matrix is not None:
            u_reservoir = clipped_logit(causal_moving_average(reservoir_matrix, length))
            z_reservoir = template.transform(u_reservoir)
            for prefix in exp["target_prefixes"]:
                m = int(prefix)
                fmax_local = trajectory_max(u_reservoir[:m], reservoir.eligible[:m])
                tmax_local = trajectory_max(z_reservoir[:m], reservoir.eligible[:m])
                local_base = {**op_base, "target_m0": m}
                for method, score, result in (
                    ("Local PaFAR-F", u_test, marginal_threshold(fmax_local, operating_alpha)),
                    ("Local PaFAR-T", z_test, marginal_threshold(tmax_local, operating_alpha)),
                    ("Local PaFAR-HC", u_test, hc_threshold(fmax_local, operating_alpha, delta)),
                ):
                    rows.append(_metric_row(method, score, result.threshold, test, result, local_base, template_meta if method.endswith("T") else None))

    calibration_seconds = time.perf_counter() - total_started - dgp_seconds - feature_seconds - learner.fitting_seconds
    total_seconds = time.perf_counter() - total_started
    for row in rows:
        row.pop("standardization_prevalences", None)
        row.update({
            "dgp_seconds": dgp_seconds, "feature_seconds": feature_seconds,
            "learner_fit_seconds": learner.fitting_seconds, "calibration_seconds": max(calibration_seconds, 0.0),
            "metric_seconds": np.nan, "total_seconds": total_seconds,
        })
    maxima_records = pd.DataFrame({
        "experiment": "Experiment II", "condition": condition, "scenario": scenario,
        "replicate": replicate, "origin": "source_calibration_non_events",
        "patient_order": np.arange(realized_m0), "fixed_maximum": f_max, "time_maximum": t_max,
    })
    if scenario == "E3" and reservoir is not None and reservoir_matrix is not None:
        u_reservoir = clipped_logit(causal_moving_average(reservoir_matrix, length))
        z_reservoir = template.transform(u_reservoir)
        local = pd.DataFrame({
            "experiment": "Experiment II", "condition": condition, "scenario": scenario,
            "replicate": replicate, "origin": "target_calibration_non_events",
            "patient_order": np.arange(len(reservoir.patient_ids)),
            "fixed_maximum": trajectory_max(u_reservoir, reservoir.eligible),
            "time_maximum": trajectory_max(z_reservoir, reservoir.eligible),
        })
        maxima_records = pd.concat([maxima_records, local], ignore_index=True)
    maxima_checkpoint = root / "calibration_maxima" / "experiment2" / condition / scenario / f"rep_{replicate:06d}.csv.gz"
    write_checkpoint(maxima_records, maxima_checkpoint, config_checksum)
    write_checkpoint(pd.DataFrame(rows), checkpoint, config_checksum)
    learner_checkpoint = root / "learner_metrics" / "experiment2" / condition / scenario / f"rep_{replicate:06d}.csv.gz"
    learner_row = pd.DataFrame([{
        "experiment": "Experiment II", "condition": condition, "scenario": scenario,
        "site": test_site, "replicate": replicate, "score_generator": "xgboost",
        "auroc_weighted": auroc, "aucpr_weighted": aucpr,
        "best_iteration": learner.best_iteration, "prediction_iteration_end": learner.prediction_end,
        "best_score": learner.best_score, "learner_failure": False, "failure_reason": "",
        "config_checksum": config_checksum, "master_seed": int(config["master_seed"]),
        "replicate_seed": f"u64:{seeds['replicate']}",
    }])
    write_checkpoint(learner_row, learner_checkpoint, config_checksum)
    return checkpoint


def run_experiment2(
    config: dict[str, Any], config_checksum: str, scenarios: list[str], replicate_start: int,
    replicate_end: int, output_root: str | Path, n_jobs: int = 1, resume: bool = False,
) -> list[Path]:
    """Run an inclusive Experiment II replicate range in independent loky workers."""
    tasks = [(scenario, rep) for scenario in scenarios for rep in range(replicate_start, replicate_end + 1)]
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_exp2_replicate)(config, config_checksum, scenario, rep, output_root, resume)
        for scenario, rep in tasks
    )
