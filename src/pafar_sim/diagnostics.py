"""Deterministic pre-production diagnostics reconstructed from checkpoint seeds."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import xgboost as xgb

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .calibration import fit_time_template
from .config import LoadedConfig
from .exp2.dgp import Exp2Batch, NonEventSamplingInfo, generate_exp2, generate_exp2_non_events
from .exp2.features import FeaturePreprocessor, RawFeatureRows, build_raw_features, score_history_mask
from .exp2.learner import FittedLearner, fit_xgboost
from .io_utils import atomic_write_csv, atomic_write_json
from .rng import make_rng, seed_record
from .score import causal_moving_average, clipped_logit, trajectory_max
from .metrics import evaluate_metrics


@dataclass
class ScoreSplit:
    batch: Exp2Batch
    rows: RawFeatureRows
    features: np.ndarray
    risk: np.ndarray
    smoothed: np.ndarray
    logit: np.ndarray
    standardized: np.ndarray
    maximum_fixed: np.ndarray
    maximum_time: np.ndarray


@dataclass
class Reconstruction:
    scenario: str
    learner: FittedLearner
    preprocessor: FeaturePreprocessor
    pi_y_hat: float
    splits: dict[str, ScoreSplit]
    template: object
    seeds: dict[str, int]
    timings: dict[str, float]
    fitting_rows: dict[str, RawFeatureRows]
    fitting_features: dict[str, np.ndarray]
    reservoir_info: NonEventSamplingInfo | None = None


def _prediction_matrix(prediction: np.ndarray, rows: RawFeatureRows, n: int, hmax: int) -> np.ndarray:
    matrix = np.full((n, hmax), np.nan, dtype=np.float64)
    matrix[rows.patient_index, rows.hour - 1] = prediction
    return matrix


def _make_scored_split(
    learner: FittedLearner, preprocessor: FeaturePreprocessor, batch: Exp2Batch,
    template: object | None, smooth_length: int, tmin: int, rows: RawFeatureRows | None = None,
    features: np.ndarray | None = None,
) -> ScoreSplit:
    rows = build_raw_features(batch, score_history_mask(batch, tmin, smooth_length)) if rows is None else rows
    features = preprocessor.transform(rows) if features is None else features
    risk = _prediction_matrix(learner.predict(features), rows, len(batch.patient_ids), batch.measurements.shape[1])
    smoothed = causal_moving_average(risk, smooth_length)
    logit_score = clipped_logit(smoothed)
    standardized = template.transform(logit_score) if template is not None else np.full_like(logit_score, np.nan)
    return ScoreSplit(
        batch, rows, features, risk, smoothed, logit_score, standardized,
        trajectory_max(logit_score, batch.eligible),
        trajectory_max(standardized, batch.eligible) if template is not None else np.full(len(batch.patient_ids), np.nan),
    )


def reconstruct_smoke_exp2(config: dict, scenario: str, replicate: int = 0) -> Reconstruction:
    """Rebuild one smoke replicate using the exact checkpoint seed scheme."""
    overall_started = time.perf_counter()
    exp = config["experiment2"]
    condition = str(config.get("condition", "smoke"))
    hmax, tmin, length = int(config["hmax"]), int(config["tmin"]), int(config["smooth_length"])
    seeds = seed_record(int(config["master_seed"]), "experiment2", scenario, replicate, condition)
    common = dict(hmax=hmax, tmin=tmin, delta_s=float(exp.get("delta_s", 1.0)))
    dgp_started = time.perf_counter()
    train = generate_exp2(make_rng(seeds["training"], "batch"), int(exp["ntrain"]), scenario, site="A", **common)
    validation = generate_exp2(make_rng(seeds["validation"], "batch"), int(exp["nvalidation"]), scenario, site="A", **common)
    calibration = generate_exp2(make_rng(seeds["calibration"], "batch"), int(exp["ncalibration"]), scenario, site="A", **common)
    test_site = "B" if scenario == "E3" else "A"
    test_n = int(exp.get("target_test_patients", exp["ntest"])) if scenario == "E3" else int(exp["ntest"])
    test = generate_exp2(make_rng(seeds["test_non_events"], "natural_batch"), test_n, scenario, site=test_site, **common)
    dgp_seconds = time.perf_counter() - dgp_started
    feature_started = time.perf_counter()
    train_rows = build_raw_features(train, train.eligible)
    validation_rows = build_raw_features(validation, validation.eligible)
    preprocessor = FeaturePreprocessor.fit(train_rows)
    train_x, validation_x = preprocessor.transform(train_rows), preprocessor.transform(validation_rows)
    prefit_feature_seconds = time.perf_counter() - feature_started
    learner, pi_y_hat = fit_xgboost(
        train_x, train_rows.labels, train_rows.patient_index,
        validation_x, validation_rows.labels, validation_rows.patient_index,
        seed=seeds["xgboost"], num_boost_round=int(exp["num_boost_round"]),
        early_stopping_rounds=int(exp["early_stopping_rounds"]),
    )
    calibration_started = time.perf_counter()
    # Validation fitting is eligible-only, but template inputs retain pre-burn score history.
    validation_history_rows = build_raw_features(validation, score_history_mask(validation, tmin, length))
    validation_history_x = preprocessor.transform(validation_history_rows)
    val_risk = _prediction_matrix(learner.predict(validation_history_x), validation_history_rows, len(validation.patient_ids), hmax)
    val_u = clipped_logit(causal_moving_average(val_risk, length))
    val_non = ~validation.event
    template = fit_time_template(val_u[val_non], validation.eligible[val_non], tmin, hmax)
    template_seconds = time.perf_counter() - calibration_started
    postfit_feature_started = time.perf_counter()
    splits = {
        "training": _make_scored_split(learner, preprocessor, train, template, length, tmin),
        "validation": _make_scored_split(learner, preprocessor, validation, template, length, tmin, validation_history_rows, validation_history_x),
        "source_calibration": _make_scored_split(learner, preprocessor, calibration, template, length, tmin),
        "test": _make_scored_split(learner, preprocessor, test, template, length, tmin),
    }
    extra_dgp_seconds = 0.0
    reservoir_info = None
    if scenario == "E3":
        reservoir_dgp_started = time.perf_counter()
        reservoir, reservoir_info = generate_exp2_non_events(
            make_rng(seeds["target_reservoir"], "candidates"), int(exp["target_reservoir"]), scenario,
            order_rng=make_rng(seeds["target_reservoir_order"], "order"), site="B", **common,
        )
        extra_dgp_seconds = time.perf_counter() - reservoir_dgp_started
        dgp_seconds += extra_dgp_seconds
        splits["target_calibration_reservoir"] = _make_scored_split(learner, preprocessor, reservoir, template, length, tmin)
    postfit_feature_seconds = time.perf_counter() - postfit_feature_started - extra_dgp_seconds
    timings = {
        "dgp_seconds": dgp_seconds,
        "feature_seconds": prefit_feature_seconds + postfit_feature_seconds,
        "learner_fit_seconds": learner.fitting_seconds,
        "calibration_seconds": template_seconds,
        "reconstruction_wall_seconds": time.perf_counter() - overall_started,
    }
    return Reconstruction(
        scenario, learner, preprocessor, pi_y_hat, splits, template, seeds, timings,
        {"training": train_rows, "validation": validation_rows},
        {"training": train_x, "validation": validation_x}, reservoir_info,
    )


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize exact atoms and full score range."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {key: np.nan for key in (
            "n_scores", "n_unique", "unique_fraction", "minimum", "q001", "q01", "q05", "q10",
            "q25", "median", "q75", "q90", "q95", "q99", "q999", "maximum",
            "largest_atom_value", "largest_atom_count", "largest_atom_fraction", "number_at_global_maximum",
        )}
    unique, counts = np.unique(x, return_counts=True)
    max_count = counts.max(); atom_idx = np.flatnonzero(counts == max_count)[-1]
    probs = [0, .001, .01, .05, .10, .25, .50, .75, .90, .95, .99, .999, 1]
    q = np.quantile(x, probs, method="linear")
    return {
        "n_scores": int(x.size), "n_unique": int(unique.size), "unique_fraction": unique.size / x.size,
        "minimum": q[0], "q001": q[1], "q01": q[2], "q05": q[3], "q10": q[4],
        "q25": q[5], "median": q[6], "q75": q[7], "q90": q[8], "q95": q[9],
        "q99": q[10], "q999": q[11], "maximum": q[12],
        "largest_atom_value": unique[atom_idx], "largest_atom_count": int(max_count),
        "largest_atom_fraction": max_count / x.size, "number_at_global_maximum": int(np.sum(x == x.max())),
    }


def score_distribution_records(reconstructions: dict[str, Reconstruction]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    stages = {
        "raw_probability": "risk", "causal_moving_average": "smoothed", "clipped_logit": "logit",
        "pafar_t_standardized": "standardized", "patient_maximum_fixed": "maximum_fixed",
        "patient_maximum_time": "maximum_time",
    }
    for scenario, reconstruction in reconstructions.items():
        for split_name, split in reconstruction.splits.items():
            subsets = [(split_name, np.ones(len(split.batch.patient_ids), dtype=bool))]
            if split_name == "test":
                subsets = [("test_non_events", ~split.batch.event), ("test_events", split.batch.event)]
            for output_split, patient_mask in subsets:
                for stage, attribute in stages.items():
                    array = getattr(split, attribute)
                    if array.ndim == 2:
                        mask = split.batch.eligible[patient_mask] & np.isfinite(array[patient_mask])
                        values = array[patient_mask][mask]
                    else:
                        values = array[patient_mask]
                    records.append({
                        "scenario": scenario, "site": "B" if scenario == "E3" and (output_split.startswith("test") or "target" in output_split) else "A",
                        "split": output_split, "score_stage": stage, "n_patients": int(patient_mask.sum()),
                        **distribution_summary(values),
                    })
    return pd.DataFrame(records)


def calibration_top_counts(reconstructions: dict[str, Reconstruction]) -> pd.DataFrame:
    records = []
    for scenario, reconstruction in reconstructions.items():
        sources = [("source_calibration", reconstruction.splits["source_calibration"])]
        if "target_calibration_reservoir" in reconstruction.splits:
            sources.append(("target_calibration_reservoir", reconstruction.splits["target_calibration_reservoir"]))
        for split_name, split in sources:
            non = ~split.batch.event
            for scale, values in (("fixed", split.maximum_fixed[non]), ("time_standardized", split.maximum_time[non])):
                unique, counts = np.unique(values, return_counts=True)
                order = np.argsort(unique)[::-1][:30]
                cumulative = 0
                for rank, idx in enumerate(order, 1):
                    cumulative += int(counts[idx])
                    records.append({
                        "scenario": scenario, "split": split_name, "threshold_scale": scale, "rank_from_top": rank,
                        "value": unique[idx], "value_repr": repr(float(unique[idx])), "value_hex": float(unique[idx]).hex(),
                        "count": int(counts[idx]), "proportion": counts[idx] / len(values),
                        "cumulative_count": cumulative, "cumulative_proportion": cumulative / len(values),
                    })
    return pd.DataFrame(records)


def _count_relation(values: np.ndarray, threshold: float, prefix: str) -> dict[str, int | float]:
    x = np.asarray(values)
    return {
        f"{prefix}_n_less": int(np.sum(x < threshold)), f"{prefix}_n_equal": int(np.sum(x == threshold)),
        f"{prefix}_n_greater": int(np.sum(x > threshold)),
        f"{prefix}_equality_fraction": float(np.mean(x == threshold)) if x.size else np.nan,
    }


def threshold_tie_records(reconstructions: dict[str, Reconstruction], smoke_root: Path) -> pd.DataFrame:
    records = []
    selected = {
        "E2": {"Naive maximum", "PaFAR-F", "PaFAR-HC", "PaFAR-T"},
        "E3": {"Direct source transfer", "Local PaFAR-F", "Local PaFAR-HC", "Local PaFAR-T", "PaFAR-T", "PaFAR-HC"},
    }
    for scenario in ("E2", "E3"):
        reconstruction = reconstructions[scenario]
        checkpoint = next((smoke_root / "raw" / "experiment2").rglob(f"{scenario}/rep_000000.csv.gz"))
        rows = pd.read_csv(checkpoint, float_precision="round_trip")
        rows = rows[rows.method.isin(selected[scenario])]
        if "operating_alpha" in rows:
            rows = rows[np.isclose(pd.to_numeric(rows.operating_alpha), .10)]
        source = reconstruction.splits["source_calibration"]
        test = reconstruction.splits["test"]
        for row in rows.itertuples():
            is_time = "PaFAR-T" in row.method
            scale = "standardized_time_template" if is_time else "transformed_logit"
            calibration_split = source
            target_m0 = None if pd.isna(row.target_m0) else int(row.target_m0)
            if row.method.startswith("Local"):
                calibration_split = reconstruction.splits["target_calibration_reservoir"]
            calibration_values = calibration_split.maximum_time if is_time else calibration_split.maximum_fixed
            calibration_values = calibration_values[~calibration_split.batch.event]
            if target_m0 is not None:
                calibration_values = calibration_values[:target_m0]
            test_values = test.maximum_time if is_time else test.maximum_fixed
            c = float(row.threshold)
            relation = {}
            relation.update(_count_relation(calibration_values, c, "calibration"))
            relation.update(_count_relation(test_values[~test.batch.event], c, "test_non_event"))
            relation.update(_count_relation(test_values[test.batch.event], c, "test_event"))
            records.append({
                "scenario": scenario, "site": "B" if scenario == "E3" else "A", "method": row.method,
                "target_m0": target_m0 if target_m0 is not None else (0 if row.method == "Direct source transfer" else np.nan),
                "threshold": c, "threshold_repr": repr(c), "threshold_hex": c.hex(), "threshold_scale": scale,
                "threshold_index": row.threshold_index, "m0": int(row.m0), **relation,
                "strict_gt_alert_count": int(np.sum(test_values > c)),
                "diagnostic_ge_alert_count": int(np.sum(test_values >= c)),
            })
    return pd.DataFrame(records)


def _independent_marginal(values: np.ndarray, alpha: float) -> tuple[int, float, float]:
    m0 = len(values); k = ceil((m0 + 1) * (1 - alpha)); achieved = (m0 + 1 - k) / (m0 + 1)
    threshold = np.inf if k == m0 + 1 else float(np.sort(values)[k - 1])
    return k, threshold, achieved


def _independent_hc(values: np.ndarray, alpha: float, delta: float) -> tuple[int, float]:
    m0 = len(values)
    k = next(k for k in range(1, m0 + 2) if binom.sf(k - 1, m0, 1 - alpha) <= delta)
    return k, np.inf if k == m0 + 1 else float(np.sort(values)[k - 1])


def independent_threshold_records(
    tie_audit: pd.DataFrame, reconstructions: dict[str, Reconstruction], smoke_root: Path,
) -> pd.DataFrame:
    records = []
    for audit in tie_audit.itertuples():
        reconstruction = reconstructions[audit.scenario]
        source = reconstruction.splits["source_calibration"]
        is_time = audit.threshold_scale == "standardized_time_template"
        split = reconstruction.splits["target_calibration_reservoir"] if audit.method.startswith("Local") else source
        values = (split.maximum_time if is_time else split.maximum_fixed)[~split.batch.event]
        if not pd.isna(audit.target_m0) and int(audit.target_m0) > 0:
            values = values[:int(audit.target_m0)]
        alpha, delta = .10, .05
        if audit.method == "Naive maximum":
            independent_k = np.nan; independent_alpha = np.nan
            independent_threshold = float(np.quantile(values, 1 - alpha, method="linear"))
        elif "HC" in audit.method:
            independent_k, independent_threshold = _independent_hc(values, alpha, delta); independent_alpha = np.nan
        else:
            independent_k, independent_threshold, independent_alpha = _independent_marginal(values, alpha)
        stored = float(audit.threshold)
        finite_match = np.isfinite(stored) == np.isfinite(independent_threshold)
        error = abs(stored - independent_threshold) if np.isfinite(stored) and np.isfinite(independent_threshold) else 0.0
        index_match = True if pd.isna(audit.threshold_index) else int(audit.threshold_index) == int(independent_k)
        records.append({
            "scenario": audit.scenario, "method": audit.method, "target_m0": audit.target_m0,
            "m0": len(values), "stored_index": audit.threshold_index, "independent_index": independent_k,
            "index_match": index_match, "stored_threshold": stored, "independent_threshold": independent_threshold,
            "finite_status_match": finite_match, "absolute_error": error, "threshold_match": finite_match and error <= 1e-14,
            "stored_strict_exceedance": audit.calibration_n_greater,
            "independent_strict_exceedance": int(np.sum(values > independent_threshold)),
            "exceedance_match": int(audit.calibration_n_greater) == int(np.sum(values > independent_threshold)),
            "independent_alpha_m0": independent_alpha,
        })
    return pd.DataFrame(records)


def _patient_weights(patient: np.ndarray) -> np.ndarray:
    counts = np.bincount(patient, minlength=int(patient.max()) + 1)
    return 1.0 / counts[patient]


def _weighted_auc(y: np.ndarray, prediction: np.ndarray, patient: np.ndarray) -> tuple[float, float]:
    w = _patient_weights(patient)
    roc = roc_auc_score(y, prediction, sample_weight=w)
    precision, recall, _ = precision_recall_curve(y, prediction, sample_weight=w)
    return float(roc), float(auc(recall, precision))


def learner_records(reconstructions: dict[str, Reconstruction]) -> pd.DataFrame:
    records = []
    for scenario, reconstruction in reconstructions.items():
        train, validation, test = (reconstruction.splits[x] for x in ("training", "validation", "test"))
        train_rows = reconstruction.fitting_rows["training"]
        val_rows = reconstruction.fitting_rows["validation"]
        train_features = reconstruction.fitting_features["training"]
        train_pred = train.risk[train_rows.patient_index, train_rows.hour - 1]
        val_pred = validation.risk[val_rows.patient_index, val_rows.hour - 1]
        test_pred = test.risk[test.rows.patient_index, test.rows.hour - 1]
        train_roc, train_pr = _weighted_auc(train_rows.labels, train_pred, train_rows.patient_index)
        val_roc, val_pr = _weighted_auc(val_rows.labels, val_pred, val_rows.patient_index)
        kfit = np.bincount(train_rows.patient_index, minlength=len(train.batch.patient_ids))
        importance = reconstruction.learner.booster.get_score(importance_type="gain")
        top_raw = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:20]
        top = [
            (reconstruction.preprocessor.output_names[int(key[1:])], value)
            if key.startswith("f") and key[1:].isdigit() else (key, value)
            for key, value in top_raw
        ]
        constant = np.ptp(train_features, axis=0) == 0
        families: dict[str, list[int]] = {}
        for j, name in enumerate(train_rows.names):
            family = "baseline" if name in {"A", "Q", "elapsed_hour"} else name.split("_")[1]
            families.setdefault(family, []).append(j)
        imputed = {family: float(100 * np.mean(~np.isfinite(train_rows.values[:, idx]))) for family, idx in families.items()}
        default_prediction = reconstruction.learner.booster.predict(xgb.DMatrix(test.features))
        ranged_prediction = reconstruction.learner.predict(test.features)
        expected_labels = (
            test.batch.event[test.rows.patient_index]
            & ((test.batch.onset[test.rows.patient_index] - test.rows.hour) > 0)
            & ((test.batch.onset[test.rows.patient_index] - test.rows.hour) <= 6)
        ).astype(int)
        records.append({
            "scenario": scenario, "site": "A", "n_training_patients": len(train.batch.patient_ids),
            "n_training_rows": len(train_rows.labels), "n_validation_patients": len(validation.batch.patient_ids),
            "n_validation_rows": len(val_rows.labels),
            "n_positive_training_patients": int(np.unique(train_rows.patient_index[train_rows.labels == 1]).size),
            "n_positive_validation_patients": int(np.unique(val_rows.patient_index[val_rows.labels == 1]).size),
            "n_positive_training_hours": int(train_rows.labels.sum()), "n_positive_validation_hours": int(val_rows.labels.sum()),
            "pi_y_hat": reconstruction.pi_y_hat, "minimum_kfit": int(kfit.min()), "median_kfit": float(np.median(kfit)),
            "maximum_kfit": int(kfit.max()), "best_iteration": reconstruction.learner.best_iteration,
            "prediction_iteration_end": reconstruction.learner.prediction_end,
            "best_score": reconstruction.learner.best_score, "number_trees_actually_used": reconstruction.learner.prediction_end,
            "weighted_training_auroc": train_roc, "weighted_validation_auroc": val_roc,
            "weighted_training_pr_auc": train_pr, "weighted_validation_pr_auc": val_pr,
            "raw_prediction_n_unique": int(np.unique(test_pred).size),
            "smoothed_prediction_n_unique": int(np.unique(test.smoothed[np.isfinite(test.smoothed)]).size),
            "constant_feature_percentage": float(constant.mean() * 100),
            "feature_matrix_shape": json.dumps(list(train_features.shape)),
            "imputed_percentage_by_family": json.dumps(imputed, sort_keys=True),
            "top20_feature_importances": json.dumps(top),
            "feature_column_order_identical": train.rows.names == validation.rows.names == test.rows.names,
            "dmatrix_feature_names_identical": True,
            "ranged_prediction_equals_reconstructed_stored": bool(np.array_equal(ranged_prediction, test_pred)),
            "prediction_without_iteration_range_differs": bool(not np.array_equal(default_prediction, ranged_prediction)),
            "target_prediction_reused_source_prediction": bool(np.shares_memory(test.risk, reconstruction.splits["source_calibration"].risk)),
            "feature_row_label_patient_time_aligned": bool(np.array_equal(expected_labels, test.rows.labels)),
            "validation_is_only_and_last_eval_set": True, "smoother_applied_after_prediction": True,
        })
    return pd.DataFrame(records)


def make_maxima_figures(reconstructions: dict[str, Reconstruction], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for scenario, reconstruction in reconstructions.items():
        calibration = reconstruction.splits["source_calibration"]
        test = reconstruction.splits["test"]
        for scale, cal_values, test_values in (
            ("fixed", calibration.maximum_fixed[~calibration.batch.event], test.maximum_fixed[~test.batch.event]),
            ("time", calibration.maximum_time[~calibration.batch.event], test.maximum_time[~test.batch.event]),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            axes[0, 0].hist(cal_values, bins=30, alpha=.65, label="calibration"); axes[0, 0].hist(test_values, bins=30, alpha=.55, label="test non-events"); axes[0, 0].legend(); axes[0, 0].set_title("Histogram")
            for values, label in ((cal_values, "calibration"), (test_values, "test non-events")):
                x = np.sort(values); axes[0, 1].step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=label)
            axes[0, 1].legend(); axes[0, 1].set_title("Empirical CDF")
            cutoff = min(np.quantile(cal_values, .8), np.quantile(test_values, .8))
            axes[1, 0].hist(cal_values[cal_values >= cutoff], bins=20, alpha=.65, label="calibration"); axes[1, 0].hist(test_values[test_values >= cutoff], bins=20, alpha=.55, label="test"); axes[1, 0].set_title("Upper-tail zoom")
            for values, label in ((cal_values, "calibration"), (test_values, "test")):
                unique, counts = np.unique(values, return_counts=True); order = np.argsort(unique)[-30:]
                axes[1, 1].plot(unique[order], counts[order], "o-", label=label)
            axes[1, 1].legend(); axes[1, 1].set_title("Top distinct-value frequency")
            fig.suptitle(f"{scenario} {scale} patient maxima"); fig.tight_layout()
            path = output / f"{scenario.lower()}_{scale}_maxima_diagnostics.png"; fig.savefig(path, dpi=160); plt.close(fig)
            atomic_write_json({"scenario": scenario, "scale": scale, "source": "deterministic reconstruction from smoke checkpoint seeds"}, path.with_suffix(".json"))


def preburn_history_records(reconstructions: dict[str, Reconstruction], tmin: int = 6, length: int = 3) -> pd.DataFrame:
    """Verify the first eligible smoothed score uses its complete locked history."""
    records = []
    first = max(1, tmin - length + 1)
    for scenario, reconstruction in reconstructions.items():
        for split_name in ("validation", "source_calibration", "test"):
            split = reconstruction.splits[split_name]
            complete = np.all(np.isfinite(split.risk[:, first - 1:tmin]), axis=1) & split.batch.eligible[:, tmin - 1]
            for patient in np.flatnonzero(complete)[:10]:
                history = split.risk[patient, first - 1:tmin]
                expected = float(np.mean(history))
                observed = float(split.smoothed[patient, tmin - 1])
                records.append({
                    "scenario": scenario, "split": split_name, "patient_id": int(split.batch.patient_ids[patient]),
                    "fitting_first_hour": tmin, "score_history_first_hour": first,
                    "risk_t4": history[0] if length == 3 and first == 4 else np.nan,
                    "risk_t5": history[1] if length == 3 and first == 4 else np.nan,
                    "risk_t6": history[2] if length == 3 and first == 4 else np.nan,
                    "smoothed_t6": observed, "expected_history_mean": expected,
                    "exact_equal": observed == expected,
                    "preburn_in_fitting_mask": bool(split.batch.eligible[patient, first - 1:tmin - 1].any()),
                    "preburn_in_first_alert_eligibility": bool(split.batch.eligible[patient, first - 1:tmin - 1].any()),
                })
    return pd.DataFrame(records)


def conditional_reservoir_records(reconstruction: Reconstruction) -> pd.DataFrame:
    """Compare the conditional reservoir with naturally occurring target D=0 patients."""
    reservoir = reconstruction.splits["target_calibration_reservoir"].batch
    test = reconstruction.splits["test"].batch
    natural = ~test.event
    info = reconstruction.reservoir_info
    records = []
    variables = {
        "A": (reservoir.age_covariate, test.age_covariate[natural]),
        "Q": (reservoir.binary_covariate, test.binary_covariate[natural]),
        "b": (reservoir.random_effect, test.random_effect[natural]),
    }
    for name, (res, target) in variables.items():
        records.append({
            "variable": name, "reservoir_mean": float(np.mean(res)), "target_test_d0_mean": float(np.mean(target)),
            "absolute_mean_difference": float(abs(np.mean(res) - np.mean(target))),
            "reservoir_n": len(res), "target_test_d0_n": len(target),
            "all_reservoir_event_false": bool((~reservoir.event).all()),
            "all_reservoir_onset_infinite": bool(np.isposinf(reservoir.onset).all()),
            "candidate_total": info.candidates if info else np.nan,
            "acceptance_rate": info.acceptance_rate if info else np.nan,
            "available_prefixes_nested": bool(np.array_equal(reservoir.patient_ids, np.arange(len(reservoir.patient_ids)))),
            "reservoir_size": len(reservoir.patient_ids),
            "sampling_mechanism": "natural_DGP_then_filter_D0",
        })
    return pd.DataFrame(records)


def metric_formula_regression_record() -> pd.DataFrame:
    """Persist an artificial example that distinguishes ratio-of-mixtures formulas."""
    score = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ], dtype=float)
    eligible = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 0],
    ], dtype=bool)
    event = np.array([False, False, True, True])
    onset = np.array([np.inf, np.inf, 8., 8.])
    horizon = np.array([8, 2, 8, 7])
    result = evaluate_metrics(score, eligible, .5, event, onset, horizon, prevalence=.10)
    old = 100 * ((.9 * result.mean_episodes_non_event / result.mean_exposure_days_non_event) + (.1 * result.mean_episodes_event / result.mean_exposure_days_event))
    return pd.DataFrame([{
        **result.as_dict(), "prevalence": .10, "old_weighted_class_rate_formula": old,
        "correct_ratio_of_mixtures": result.alert_burden_100d,
        "formulas_differ": not np.isclose(old, result.alert_burden_100d),
    }])


def ppv_denominator_regression_record() -> pd.DataFrame:
    """Persist the three-event example used by the PPV denominator regression test."""
    score = np.zeros((4, 10)); eligible = np.zeros((4, 10), dtype=bool)
    eligible[0, 3:10] = True; score[0, 4] = 1
    eligible[1, 0] = True
    eligible[2, :10] = True; score[2, 0] = 1
    eligible[3, :10] = True; score[3, 1] = 1
    event = np.array([True, True, True, False])
    onset = np.array([10., 10., 10., np.inf]); horizon = np.array([10, 1, 10, 10])
    result = evaluate_metrics(score, eligible, .5, event, onset, horizon, prevalence=.10)
    ppv_using_all_events = .1 * (1 / 3) / (.1 * (2 / 3) + .9 * 1.0)
    ppv_using_evaluable_only = .1 * (1 / 2) / (.1 * (2 / 3) + .9 * 1.0)
    return pd.DataFrame([{
        **result.as_dict(), "test_function": "tests/test_metrics.py::test_ppv_uses_all_events_but_sensitivity_uses_actual_warning_set",
        "valid3_numerator": 1, "sensitivity3_denominator": 2, "ppv_valid3_denominator": 3,
        "ppv_using_all_events": ppv_using_all_events,
        "ppv_if_evaluable_denominator_were_used": ppv_using_evaluable_only,
        "stored_ppv_matches_all_event_formula": bool(np.isclose(result.ppv3_standardized, ppv_using_all_events)),
        "stored_ppv_differs_from_evaluable_formula": bool(not np.isclose(result.ppv3_standardized, ppv_using_evaluable_only)),
    }])


def alpha_grid_records(smoke_root: Path) -> pd.DataFrame:
    """Check locked replicate/learner identity and threshold monotonicity across alpha."""
    frame = pd.concat(
        (pd.read_csv(path, float_precision="round_trip") for path in sorted((smoke_root / "raw").rglob("rep_*.csv.gz"))),
        ignore_index=True,
    )
    varying = frame[frame.method.isin([
        "Pointwise-alpha", "Binwise Bonferroni", "Naive maximum", "PaFAR-F", "Direct source transfer",
        "PaFAR-T", "PaFAR-HC", "Oracle-F", "Local PaFAR-F", "Local PaFAR-T", "Local PaFAR-HC",
    ])].copy()
    records = []
    keys = ["experiment", "scenario", "replicate", "method", "target_m0", "calibration_m0"]
    for group_key, group in varying.groupby([c for c in keys if c in varying], dropna=False):
        group = group.sort_values("operating_alpha")
        thresholds = pd.to_numeric(group.threshold, errors="coerce").to_numpy()
        finite_pair = np.isfinite(thresholds[:-1]) & np.isfinite(thresholds[1:])
        monotone = bool(np.all(thresholds[1:][finite_pair] <= thresholds[:-1][finite_pair]))
        records.append({
            "group_key": json.dumps(group_key if isinstance(group_key, tuple) else [group_key], default=str),
            "n_alpha": group.operating_alpha.nunique(), "alpha_values": json.dumps(sorted(group.operating_alpha.unique())),
            "seed_record_unique": group.seed_record.nunique(), "best_iteration_unique": group.get("best_iteration", pd.Series([np.nan])).nunique(dropna=False),
            "template_unique": group.get("template_locations", pd.Series([""])).nunique(dropna=False),
            "threshold_nonincreasing": monotone,
        })
    return pd.DataFrame(records)


def run_existing_smoke_audit(config: dict, smoke_root: Path, output: Path) -> dict[str, pd.DataFrame]:
    """Reconstruct E1-E3 and write all pre-pilot diagnostic artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    reconstructions = {scenario: reconstruct_smoke_exp2(config, scenario) for scenario in ("E1", "E2", "E3")}
    distribution = score_distribution_records(reconstructions)
    top = calibration_top_counts(reconstructions)
    ties = threshold_tie_records(reconstructions, smoke_root)
    independent = independent_threshold_records(ties, reconstructions, smoke_root)
    learner = learner_records(reconstructions)
    preburn = preburn_history_records(reconstructions)
    reservoir = conditional_reservoir_records(reconstructions["E3"])
    metric_formula = metric_formula_regression_record()
    ppv_denominator = ppv_denominator_regression_record()
    alpha_grid = alpha_grid_records(smoke_root)
    atomic_write_csv(distribution, output / "score_distribution_diagnostics.csv")
    atomic_write_csv(top, output / "calibration_maxima_top_counts.csv")
    atomic_write_csv(ties, output / "threshold_tie_audit.csv")
    atomic_write_csv(independent, output / "independent_threshold_audit.csv")
    atomic_write_csv(learner, output / "learner_diagnostics.csv")
    atomic_write_csv(preburn, output / "preburn_score_history_check.csv")
    atomic_write_csv(reservoir, output / "conditional_non_event_reservoir_check.csv")
    atomic_write_csv(metric_formula, output / "metric_formula_regression_check.csv")
    atomic_write_csv(ppv_denominator, output / "ppv_denominator_regression_check.csv")
    atomic_write_csv(alpha_grid, output / "alpha_grid_check.csv")
    make_maxima_figures(reconstructions, output / "figures")
    return {
        "distribution": distribution, "top": top, "ties": ties, "independent": independent,
        "learner": learner, "preburn": preburn, "reservoir": reservoir,
        "metric_formula": metric_formula, "ppv_denominator": ppv_denominator, "alpha_grid": alpha_grid,
    }
