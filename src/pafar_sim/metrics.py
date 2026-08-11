"""Patient-level operating metrics from Equations (9)--(13), (47)--(48)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from .alerting import count_alert_episodes, first_alert


@dataclass(frozen=True)
class MetricResult:
    pfa: float
    long_stay_pfa: float
    sens3: float
    sens0: float
    premature: float
    median_lead: float
    ppv3_standardized: float
    ppv0_standardized: float
    alert_burden_100d: float
    alert_episodes_per_patient: float
    n_non_events: int
    n_events: int
    n_sens3: int
    n_sens0: int
    n_valid3: int
    n_valid0: int
    n_alert_non_events: int
    n_alert_events: int
    any_alert_rate_non_event: float
    any_alert_rate_event: float
    valid3_rate_all_events: float
    valid0_rate_all_events: float
    mean_episodes_non_event: float
    mean_episodes_event: float
    mean_exposure_days_non_event: float
    mean_exposure_days_event: float
    n_event_total: int
    n_event_evaluable3: int
    n_event_evaluable0: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_mean(x: NDArray[np.bool_]) -> float:
    return float(np.mean(x)) if x.size else np.nan


def _standardized_ppv(pi: float, valid: float, a1: float, a0: float) -> float:
    denominator = pi * a1 + (1 - pi) * a0
    return np.nan if denominator == 0 or not np.isfinite(valid) else float(pi * valid / denominator)


def evaluate_metrics(
    score: NDArray[np.floating], eligible: NDArray[np.bool_], threshold: float | NDArray[np.floating],
    event: NDArray[np.bool_], onset: NDArray[np.floating], horizon: NDArray[np.integer],
    patient_ids: NDArray[np.integer] | None = None, prevalence: float = 0.10,
) -> MetricResult:
    """Evaluate first-alert estimands and standardized episode burden."""
    event = np.asarray(event, dtype=bool)
    tau = first_alert(score, eligible, threshold)
    episodes = count_alert_episodes(score, eligible, threshold)
    non = ~event
    pfa = _safe_mean(np.isfinite(tau[non]))
    ids = np.arange(event.size) if patient_ids is None else np.asarray(patient_ids)
    exposure = eligible.sum(axis=1)
    non_idx = np.flatnonzero(non)
    if non_idx.size:
        order = np.lexsort((ids[non_idx], -exposure[non_idx]))
        q = int(np.ceil(non_idx.size / 4))
        long_idx = non_idx[order[:q]]
        long_pfa = float(np.mean(np.isfinite(tau[long_idx])))
    else:
        long_pfa = np.nan
    delta = np.full(tau.shape, np.inf)
    finite_tau = np.isfinite(tau)
    delta[finite_tau] = onset[finite_tau] - tau[finite_tau]
    hours = np.arange(1, score.shape[1] + 1)[None, :]
    warning0 = eligible & event[:, None] & (hours >= onset[:, None] - 6) & (hours <= onset[:, None])
    warning3 = eligible & event[:, None] & (hours >= onset[:, None] - 6) & (hours <= onset[:, None] - 3)
    eval0 = event & warning0.any(axis=1)
    eval3 = event & warning3.any(axis=1)
    valid0 = eval0 & np.isfinite(tau) & (delta >= 0) & (delta <= 6)
    valid3 = eval3 & np.isfinite(tau) & (delta >= 3) & (delta <= 6)
    premature = eval0 & np.isfinite(tau) & (tau < onset - 6)
    sens0, sens3 = _safe_mean(valid0[eval0]), _safe_mean(valid3[eval3])
    premature_rate = _safe_mean(premature[eval0])
    lead = float(np.median(delta[valid0])) if valid0.any() else np.nan
    a0 = pfa
    a1 = _safe_mean(np.isfinite(tau[event]))
    v3 = _safe_mean(valid3[event])
    v0 = _safe_mean(valid0[event])
    ppv3 = _standardized_ppv(prevalence, v3, a1, a0)
    ppv0 = _standardized_ppv(prevalence, v0, a1, a0)
    mu_n0 = float(np.mean(episodes[non])) if non.any() else np.nan
    mu_n1 = float(np.mean(episodes[event])) if event.any() else np.nan
    mu_e0 = float(np.mean(exposure[non] / 24)) if non.any() else np.nan
    mu_e1 = float(np.mean(exposure[event] / 24)) if event.any() else np.nan
    burden_denominator = (1 - prevalence) * mu_e0 + prevalence * mu_e1
    burden_numerator = (1 - prevalence) * mu_n0 + prevalence * mu_n1
    burden = 100 * burden_numerator / burden_denominator if np.isfinite(burden_denominator) and burden_denominator > 0 else np.nan
    return MetricResult(
        pfa, long_pfa, sens3, sens0, premature_rate, lead, ppv3, ppv0, burden,
        float(episodes.mean()) if episodes.size else np.nan, int(non.sum()), int(event.sum()),
        int(eval3.sum()), int(eval0.sum()), int(valid3.sum()), int(valid0.sum()),
        int(np.isfinite(tau[non]).sum()), int(np.isfinite(tau[event]).sum()),
        a0, a1, v3, v0, mu_n0, mu_n1, mu_e0, mu_e1,
        int(event.sum()), int(eval3.sum()), int(eval0.sum()),
    )


def standardized_from_components(result: MetricResult, prevalence: float) -> dict[str, float]:
    """Reweight Equation (47)--(48) without rerunning trajectories or alerts."""
    ppv3 = _standardized_ppv(prevalence, result.valid3_rate_all_events, result.any_alert_rate_event, result.any_alert_rate_non_event)
    ppv0 = _standardized_ppv(prevalence, result.valid0_rate_all_events, result.any_alert_rate_event, result.any_alert_rate_non_event)
    denominator = (1 - prevalence) * result.mean_exposure_days_non_event + prevalence * result.mean_exposure_days_event
    numerator = (1 - prevalence) * result.mean_episodes_non_event + prevalence * result.mean_episodes_event
    burden = 100 * numerator / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
    return {"ppv3": ppv3, "ppv0": ppv0, "alert_burden_100d": burden}


def cumulative_false_alert_curve(tau_non_events: NDArray[np.floating], horizon: NDArray[np.integer], hmax: int) -> NDArray[np.float64]:
    """Cumulative first-false-alert probability with all non-events in denominator."""
    tau = np.asarray(tau_non_events)
    h = np.asarray(horizon)
    if tau.size == 0:
        return np.full(hmax, np.nan)
    return np.asarray([np.mean(tau <= np.minimum(hour, h)) for hour in range(1, hmax + 1)])


def threshold_free_metrics(
    risk: NDArray[np.floating], labels: NDArray[np.integer], patient_index: NDArray[np.integer]
) -> tuple[float, float]:
    """Patient-equal weighted AUROC and trapezoidal PR-AUC."""
    x, y, p = np.asarray(risk), np.asarray(labels), np.asarray(patient_index)
    valid = np.isfinite(x)
    x, y, p = x[valid], y[valid], p[valid]
    if x.size == 0 or np.unique(y).size < 2:
        return np.nan, np.nan
    counts = np.bincount(p, minlength=int(p.max()) + 1)
    weights = 1.0 / counts[p]
    auroc = roc_auc_score(y, x, sample_weight=weights)
    precision, recall, _ = precision_recall_curve(y, x, sample_weight=weights)
    return float(auroc), float(auc(recall, precision))
