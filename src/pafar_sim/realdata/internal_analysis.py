"""Pooled internal PhysioNet analysis with hospital-specific calibration."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pafar_sim.alerting import hourly_bin_thresholds
from pafar_sim.calibration import youden_threshold
from pafar_sim.io_utils import atomic_write_csv, atomic_write_json
from .bootstrap import stratified_indices
from .calibration import MethodThreshold, calibrate_hospital, exact_binomial, fit_template
from .feature_builder import fitting_mask
from .feature_cache import open_cache, patient_codes
from .imputation import FrozenPreprocessor, fit_preprocessor, save_preprocessor
from .learner import SelectedLearner, fit_grid
from .json_utils import dumps_json_numeric, encode_json_numeric, numeric_state
from .recovery import load_fixed_internal_split
from .schema import RealDataConfig
from .scoring import TrajectoryData, evaluate, score_patients, threshold_free
from .splitting import seed_registry, stratified_patient_split
from .utility import normalized_utility, utility_components


METHODS = ("Fixed 0.5", "Youden", "Pointwise-alpha", "Bonferroni", "Naive maximum", "PaFAR-F", "PaFAR-T", "PaFAR-HC")


def fit_split_preprocessor(config: RealDataConfig, split: pd.DataFrame, split_name: str) -> FrozenPreprocessor:
    index = patient_codes(config)
    training_ids = set(split.loc[split.split == "train", "patient_id"])
    training_codes = set(index.loc[index.patient_id.isin(training_ids), "patient_code"].astype(int))
    matrices, masks = [], []
    names: tuple[str, ...] | None = None
    with ExitStack() as stack:
        for hospital in ("A", "B"):
            cache = stack.enter_context(open_cache(config, hospital))
            matrices.append(cache["values"])
            mask = np.isin(cache["patient_code"], np.fromiter(training_codes, dtype=np.int32))
            mask &= fitting_mask(cache["hours"], cache["onset"], config.tmin, config.hmax)
            masks.append(mask)
            current = tuple(cache["meta"]["feature_names"])
            if names is not None and current != names:
                raise ValueError("A/B feature order mismatch")
            names = current
        assert names is not None
        fitted = fit_preprocessor(matrices, masks, names)
    save_preprocessor(fitted, config.outputs / "models" / split_name / "preprocessor")
    return fitted


def _split_selections(split: pd.DataFrame, name: str) -> dict[str, set[str]]:
    return {hospital: set(split.loc[(split.split == name) & (split.hospital_set == hospital), "patient_id"]) for hospital in ("A", "B")}


def _threshold_payload(record: MethodThreshold, hmax: int) -> float | np.ndarray:
    if record.method == "Bonferroni":
        assert record.boundaries is not None
        return hourly_bin_thresholds(hmax, record.boundaries, np.asarray(record.threshold, dtype=float))
    return float(record.threshold)


def _serialize_threshold_records(
    threshold_records: dict[float, list[MethodThreshold]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Build CSV rows and JSON diagnostic rows without changing thresholds."""
    threshold_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for _, records in threshold_records.items():
        for record in records:
            row_record = asdict(record)
            state = numeric_state(record.threshold)
            row_record["threshold"] = dumps_json_numeric(record.threshold, separators=(",", ":"))
            row_record["boundaries"] = dumps_json_numeric(record.boundaries, separators=(",", ":"))
            row_record.update(state)
            threshold_rows.append(row_record)
            if state["threshold_nonfinite_count"]:
                diagnostic_rows.append({
                    "method": record.method,
                    "hospital": record.hospital,
                    "operating_alpha": record.alpha,
                    "threshold_shape": list(np.asarray(record.threshold).shape),
                    "threshold_scale": "time_template_standardized" if record.method == "PaFAR-T" else "fixed_logit",
                    "threshold_origin": (
                        "empty eligible-set maxima map to -inf before the prespecified binwise order statistic"
                        if state["threshold_has_neg_inf"]
                        else "finite-sample order-statistic +inf augmentation"
                    ),
                    "classification": "algorithm_defined_sentinel",
                    "threshold": encode_json_numeric(record.threshold),
                    "boundaries": encode_json_numeric(record.boundaries),
                    **state,
                })
    return pd.DataFrame(threshold_rows), diagnostic_rows


def _method_score_threshold(
    data: TrajectoryData, method: str, threshold_records: list[MethodThreshold],
    template, youden: float, hmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "Fixed 0.5":
        return data.risk, np.full(data.risk.shape, .5)
    if method == "Youden":
        return data.risk, np.full(data.risk.shape, youden)
    score = template.transform(data.score_f) if method == "PaFAR-T" else data.score_f
    boundary = np.empty(score.shape, dtype=float)
    for hospital in ("A", "B"):
        record = next(r for r in threshold_records if r.hospital == hospital and r.method == method)
        payload = _threshold_payload(record, hmax)
        mask = data.hospital == hospital
        boundary[mask] = np.broadcast_to(payload, score[mask].shape)
    return score, boundary


def _metric_from_patient(frame: pd.DataFrame) -> dict[str, float]:
    non = ~frame.event.astype(bool)
    def ratio(num, den):
        return float(num/den) if den else np.nan
    utility_den = frame.utility_best.sum()-frame.utility_inactive.sum()
    return {
        "pfa": ratio(frame.loc[non, "alerted"].sum(), non.sum()),
        "sens3": ratio(frame.valid3.sum(), frame.eval3.sum()),
        "sens0": ratio(frame.valid0.sum(), frame.eval0.sum()),
        "ppv3": ratio(frame.valid3.sum(), frame.alerted.sum()),
        "alerts_per_100d": ratio(100*frame.episodes.sum(), frame.exposure_days.sum()),
        "utility": ratio(frame.utility_observed.sum()-frame.utility_inactive.sum(), utility_den),
    }


def _bootstrap_all(
    config: RealDataConfig, patient_tables: dict[tuple[float, str], pd.DataFrame], seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = next(iter(patient_tables.values()))
    strata = first.hospital.astype(str) + "|" + first.event.astype(str)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    observed = {key: _metric_from_patient(frame) for key, frame in patient_tables.items()}
    rows = []
    for b in range(config.bootstrap_replicates):
        idx = stratified_indices(strata.to_numpy(), rng)
        for (alpha, method), frame in patient_tables.items():
            rows.append({"bootstrap": b, "alpha": alpha, "method": method, **_metric_from_patient(frame.iloc[idx])})
    samples = pd.DataFrame(rows)
    summaries = []
    for (alpha, method), metrics in observed.items():
        block = samples[(samples.alpha == alpha) & (samples.method == method)]
        for metric, value in metrics.items():
            x = block[metric].to_numpy(float); x = x[np.isfinite(x)]
            summaries.append({"alpha": alpha, "method": method, "metric": metric, "observed": value,
                              "bootstrap_se": x.std(ddof=1) if len(x)>1 else np.nan,
                              "lower_95": np.quantile(x,.025) if len(x) else np.nan,
                              "upper_95": np.quantile(x,.975) if len(x) else np.nan,
                              "valid_bootstrap": len(x)})
    comparisons = ("Pointwise-alpha", "Bonferroni", "PaFAR-T", "PaFAR-HC", "Fixed 0.5", "Youden")
    contrasts = []
    for alpha in config.alphas:
        reference = samples[(samples.alpha == alpha) & (samples.method == "PaFAR-F")].set_index("bootstrap")
        for method in comparisons:
            candidate = samples[(samples.alpha == alpha) & (samples.method == method)].set_index("bootstrap")
            for metric in ("pfa", "sens3", "sens0", "ppv3", "alerts_per_100d", "utility"):
                diff = (candidate[metric]-reference[metric]).to_numpy(float); diff = diff[np.isfinite(diff)]
                obs = observed[(alpha,method)][metric]-observed[(alpha,"PaFAR-F")][metric]
                contrasts.append({"alpha": alpha, "comparison": f"{method} minus PaFAR-F", "metric": metric,
                                  "observed_difference": obs, "bootstrap_se": diff.std(ddof=1) if len(diff)>1 else np.nan,
                                  "lower_95": np.quantile(diff,.025) if len(diff) else np.nan,
                                  "upper_95": np.quantile(diff,.975) if len(diff) else np.nan,
                                  "valid_bootstrap": len(diff)})
    return samples, pd.DataFrame(summaries), pd.DataFrame(contrasts)


def _cumulative_curves(config: RealDataConfig, patient_tables: dict[tuple[float, str], pd.DataFrame], seed: int) -> pd.DataFrame:
    methods = ("Pointwise-alpha", "Bonferroni", "PaFAR-F", "PaFAR-T", "PaFAR-HC")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows = []
    for panel in ("overall", "A", "B"):
        for method in methods:
            frame = patient_tables[(.10, method)]
            frame = frame.loc[~frame.event.astype(bool)]
            if panel != "overall": frame = frame[frame.hospital == panel]
            tau = frame.tau.to_numpy(float); horizon = frame.horizon.to_numpy(int)
            estimates = np.asarray([np.mean(tau <= np.minimum(hour,horizon)) for hour in range(1,config.hmax+1)])
            boots = np.empty((config.bootstrap_replicates, config.hmax), dtype=np.float32)
            hospital = frame.hospital.to_numpy()
            for b in range(config.bootstrap_replicates):
                idx = stratified_indices(hospital, rng) if panel == "overall" else rng.choice(len(frame),len(frame),replace=True)
                boots[b] = [np.mean(tau[idx] <= np.minimum(hour,horizon[idx])) for hour in range(1,config.hmax+1)]
            lower, upper = np.quantile(boots,[.025,.975],axis=0)
            rows.extend({"panel":panel,"method":method,"hour":hour,"estimate":estimates[hour-1],"lower_95":lower[hour-1],"upper_95":upper[hour-1],"denominator":len(frame)} for hour in range(1,config.hmax+1))
    return pd.DataFrame(rows)


def run_internal_primary(
    config: RealDataConfig, manifest: pd.DataFrame, *, resume: bool = True,
) -> dict[str, object]:
    started = time.perf_counter(); seeds = seed_registry(config.master_seed)
    cohort = manifest.loc[manifest.primary_cohort, ["patient_id","hospital_set","any_sepsis_label","reconstructed_onset","last_ICULOS"]].copy()
    split = load_fixed_internal_split(config, manifest)
    atomic_write_csv(split.groupby(["split","hospital_set","any_sepsis_label"]).size().rename("n").reset_index(), config.outputs / "qc" / "internal_split_counts.csv")
    preprocessor = fit_split_preprocessor(config, split, "internal")
    learner, grid = fit_grid(config, _split_selections(split,"train"), _split_selections(split,"validation"), preprocessor,
                             seed=seeds["xgboost"], source="internal", drop_hospital=False, resume=resume)
    validation = score_patients(config, learner, preprocessor, set(split.loc[split.split=="validation","patient_id"]), drop_hospital=False)
    calibration = score_patients(config, learner, preprocessor, set(split.loc[split.split=="calibration","patient_id"]), drop_hospital=False)
    test = score_patients(config, learner, preprocessor, set(split.loc[split.split=="test","patient_id"]), drop_hospital=False)
    template = fit_template(validation, config.tmin, config.hmax, config.template_min_patients)
    template_dict = asdict(template)
    atomic_write_json(template_dict, config.outputs / "internal_primary" / "time_template.json")
    mask = validation.eligible & np.isfinite(validation.risk)
    row, col = np.where(mask)
    val_labels = (validation.event[row] & ((validation.onset[row]-(col+1))>0) & ((validation.onset[row]-(col+1))<=6)).astype(int)
    youden = youden_threshold(validation.risk[row,col], val_labels, row)
    threshold_records: dict[float,list[MethodThreshold]] = {}
    for alpha in config.alphas:
        threshold_records[alpha] = sum((calibrate_hospital(calibration, template, hospital, alpha, config.delta, config.tmin, config.hmax) for hospital in ("A","B")), [])
    thresholds, diagnostic_rows = _serialize_threshold_records(threshold_records)
    if diagnostic_rows:
        atomic_write_json(
            {
                "status": "confirmed_algorithm_defined_sentinel",
                "record_count": len(diagnostic_rows),
                "records": diagnostic_rows,
            },
            config.outputs / "logs" / "nonfinite_threshold_diagnostic.json",
        )
    atomic_write_csv(thresholds, config.outputs / "internal_primary" / "thresholds.csv")
    scorer_path = config.data_root / "manifests" / "official_evaluation_2019" / "evaluate_sepsis_score.py"
    results, hospital_rows, first_rows = [], [], []
    patient_tables: dict[tuple[float,str],pd.DataFrame] = {}
    auroc, pr_auc = threshold_free(test)
    for alpha in config.alphas:
        for method in METHODS:
            score, boundary = _method_score_threshold(test, method, threshold_records[alpha], template, youden, config.hmax)
            predictions = test.utility_grid & np.isfinite(score) & (score > boundary)
            utility, raw_utility = normalized_utility(test, predictions, scorer_path)
            utility_observed, utility_inactive, utility_best = utility_components(test, predictions, scorer_path)
            metric, detail = evaluate(test, score, boundary, utility=utility)
            results.append({"alpha":alpha,"method":method,**metric.as_dict(),"raw_utility":raw_utility})
            patient_frame = pd.DataFrame({"patient_id":test.patient_id,"hospital":test.hospital,"event":test.event,
                                          "onset":test.onset,"horizon":test.horizon,**detail,
                                          "utility_observed":utility_observed,"utility_inactive":utility_inactive,"utility_best":utility_best})
            patient_frame["alpha"], patient_frame["method"] = alpha, method
            patient_tables[(alpha,method)] = patient_frame
            first_rows.append(patient_frame)
            for hospital in ("A","B"):
                block = patient_frame[(patient_frame.hospital==hospital)&(~patient_frame.event)]
                successes, n = int(block.alerted.sum()), len(block)
                lower, upper, upper_one = exact_binomial(successes,n)
                hospital_rows.append({"alpha":alpha,"method":method,"hospital":hospital,"alerts":successes,"n_non_events":n,
                                      "pfa":successes/n if n else np.nan,"lower_95":lower,"upper_95":upper,"upper_one_sided_95":upper_one})
    samples, bootstrap_summary, contrasts = _bootstrap_all(config, patient_tables, seeds["bootstrap"])
    results_frame = pd.DataFrame(results)
    for i,row_result in results_frame.iterrows():
        hit = bootstrap_summary[(bootstrap_summary.alpha==row_result.alpha)&(bootstrap_summary.method==row_result.method)&(bootstrap_summary.metric=="pfa")]
        if len(hit):
            results_frame.loc[i,"pfa_lower_95"] = hit.lower_95.iloc[0]; results_frame.loc[i,"pfa_upper_95"] = hit.upper_95.iloc[0]
    out = config.outputs / "internal_primary"
    atomic_write_csv(pd.concat(first_rows,ignore_index=True), out / "first_alerts.csv.gz")
    atomic_write_csv(results_frame, out / "internal_results_long.csv.gz")
    atomic_write_csv(results_frame, out / "internal_results_summary.csv")
    atomic_write_csv(pd.DataFrame(hospital_rows), out / "hospital_specific_pfa.csv")
    atomic_write_csv(pd.DataFrame([{**learner.params,"best_iteration":learner.best_iteration,"trees_used":learner.trees_used,"validation_aucpr":learner.best_score,"weighted_auroc":auroc,"trapezoidal_pr_auc":pr_auc,"pi_y":learner.pi_y}]), out / "learner_metrics.csv")
    atomic_write_csv(grid, out / "xgboost_grid.csv")
    atomic_write_csv(samples, config.outputs / "bootstrap" / "internal_bootstrap_samples.csv.gz")
    atomic_write_csv(bootstrap_summary, config.outputs / "bootstrap" / "internal_bootstrap_summary.csv")
    atomic_write_csv(contrasts, config.outputs / "bootstrap" / "internal_paired_contrasts.csv")
    curves = _cumulative_curves(config, patient_tables, seeds["bootstrap"]+1)
    atomic_write_csv(curves, out / "cumulative_false_alert.csv")
    lines = ["# Internal Primary Report","","Gate status: **PASS**","",f"Selected grid: {learner.params}",f"Best iteration: {learner.best_iteration}",f"Weighted AUROC: {auroc:.6f}",f"Trapezoidal PR-AUC: {pr_auc:.6f}",f"Elapsed seconds: {time.perf_counter()-started:.3f}"]
    (out / "INTERNAL_PRIMARY_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"split":split,"preprocessor":preprocessor,"learner":learner,"template":template,"validation":validation,"calibration":calibration,"test":test,
            "threshold_records":threshold_records,"youden":youden,"results":results_frame,"patient_tables":patient_tables,"curves":curves,
            "auroc":auroc,"pr_auc":pr_auc,"elapsed_seconds":time.perf_counter()-started}
