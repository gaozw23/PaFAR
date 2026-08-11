"""Locked patient trajectories and real-data operating metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from pafar_sim.alerting import count_alert_episodes, first_alert, threshold_matrix
from pafar_sim.score import causal_moving_average
from .feature_cache import open_cache, patient_codes
from .learner import SelectedLearner, predict_rows
from .imputation import FrozenPreprocessor
from .schema import RealDataConfig


@dataclass(frozen=True)
class TrajectoryData:
    patient_id: np.ndarray
    hospital: np.ndarray
    event: np.ndarray
    onset: np.ndarray
    horizon: np.ndarray
    risk: np.ndarray
    score_f: np.ndarray
    eligible: np.ndarray
    utility_grid: np.ndarray
    labels: np.ndarray

    def subset(self, ids: set[str] | np.ndarray) -> "TrajectoryData":
        mask = np.isin(self.patient_id, list(ids))
        return TrajectoryData(*(np.asarray(getattr(self, field))[mask] for field in self.__dataclass_fields__))


@dataclass(frozen=True)
class RealMetricResult:
    pfa: float
    sens3: float
    sens0: float
    premature: float
    median_lead: float
    ppv3: float
    ppv0: float
    alerts_per_100d: float
    episodes_per_patient: float
    utility: float
    n_non_events: int
    n_events: int
    n_sens3: int
    n_sens0: int
    n_alerted: int
    n_valid3: int
    n_valid0: int
    total_episodes: int
    exposure_days: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_patients(
    config: RealDataConfig, learner: SelectedLearner, preprocessor: FrozenPreprocessor,
    patient_ids: set[str], *, drop_hospital: bool, hmax: int | None = None,
) -> TrajectoryData:
    index = patient_codes(config)
    selected = index[index.patient_id.isin(patient_ids)].sort_values(["hospital_set", "patient_id"], kind="mergesort").reset_index(drop=True)
    n = len(selected)
    hmax = int(hmax if hmax is not None else config.hmax)
    risk = np.full((n, hmax), np.nan, dtype=np.float32)
    code_to_row = {int(code): i for i, code in enumerate(selected.patient_code)}
    for hospital in ("A", "B"):
        with open_cache(config, hospital) as cache:
            codes_needed = set(selected.loc[selected.hospital_set == hospital, "patient_code"].astype(int))
            mask = np.isin(cache["patient_code"], np.fromiter(codes_needed, dtype=np.int32)) & (np.asarray(cache["hours"]) <= hmax)
            idx = np.flatnonzero(mask)
            predictions = predict_rows(learner, preprocessor, np.asarray(cache["values"])[idx], drop_hospital=drop_hospital)
            codes = np.asarray(cache["patient_code"])[idx]
            hours = np.asarray(cache["hours"])[idx]
        for code in np.unique(codes):
            local = codes == code
            risk[code_to_row[int(code)], hours[local]-1] = predictions[local]
    smoothed = causal_moving_average(risk, config.smooth_length)
    score_f = logit(np.clip(smoothed, config.epsilon, 1-config.epsilon)).astype(np.float32)
    event = selected.any_sepsis_label.astype(bool).to_numpy()
    onset = selected.reconstructed_onset.astype(float).to_numpy()
    horizon = selected.last_ICULOS.astype(int).to_numpy()
    hours2 = np.arange(1, hmax+1)[None, :]
    eligible = (hours2 >= config.tmin) & (hours2 <= np.minimum(horizon[:, None], config.hmax)) & (hours2 < onset[:, None])
    utility_grid = (hours2 >= config.tmin) & (hours2 <= np.minimum(horizon[:, None], config.hmax))
    labels = event[:, None] & (hours2 >= (onset[:, None] - 6)) & (hours2 <= horizon[:, None])
    return TrajectoryData(selected.patient_id.to_numpy(str), selected.hospital_set.to_numpy(str), event, onset, horizon, risk, score_f, eligible, utility_grid, labels.astype(np.uint8))


def threshold_free(data: TrajectoryData) -> tuple[float, float]:
    mask = data.eligible & np.isfinite(data.risk)
    rows, cols = np.where(mask)
    y = (data.event[rows] & ((data.onset[rows] - (cols+1)) > 0) & ((data.onset[rows] - (cols+1)) <= 6)).astype(int)
    scores = data.risk[rows, cols]
    counts = np.bincount(rows, minlength=len(data.patient_id))
    weights = 1 / counts[rows]
    auroc = roc_auc_score(y, scores, sample_weight=weights)
    precision, recall, _ = precision_recall_curve(y, scores, sample_weight=weights)
    return float(auroc), float(auc(recall, precision))


def evaluate(
    data: TrajectoryData, score: np.ndarray, threshold: float | np.ndarray, *, utility: float = np.nan,
) -> tuple[RealMetricResult, dict[str, np.ndarray]]:
    tau = first_alert(score, data.eligible, threshold)
    episodes = count_alert_episodes(score, data.eligible, threshold)
    event, non = data.event, ~data.event
    delta = data.onset - tau
    hours = np.arange(1, score.shape[1]+1)[None, :]
    warning0 = data.eligible & event[:, None] & (hours >= data.onset[:, None]-6) & (hours <= data.onset[:, None])
    warning3 = data.eligible & event[:, None] & (hours >= data.onset[:, None]-6) & (hours <= data.onset[:, None]-3)
    eval0, eval3 = event & warning0.any(1), event & warning3.any(1)
    valid0 = eval0 & np.isfinite(tau) & (delta >= 0) & (delta <= 6)
    valid3 = eval3 & np.isfinite(tau) & (delta >= 3) & (delta <= 6)
    premature = eval0 & np.isfinite(tau) & (tau < data.onset-6)
    alerted = np.isfinite(tau)
    exposure_days = float(data.eligible.sum()/24)
    result = RealMetricResult(
        float(alerted[non].mean()) if non.any() else np.nan,
        float(valid3[eval3].mean()) if eval3.any() else np.nan,
        float(valid0[eval0].mean()) if eval0.any() else np.nan,
        float(premature[eval0].mean()) if eval0.any() else np.nan,
        float(np.median(delta[valid0])) if valid0.any() else np.nan,
        float(valid3.sum()/alerted.sum()) if alerted.any() else np.nan,
        float(valid0.sum()/alerted.sum()) if alerted.any() else np.nan,
        float(100*episodes.sum()/exposure_days) if exposure_days > 0 else np.nan,
        float(episodes.mean()) if len(episodes) else np.nan, float(utility),
        int(non.sum()), int(event.sum()), int(eval3.sum()), int(eval0.sum()), int(alerted.sum()),
        int(valid3.sum()), int(valid0.sum()), int(episodes.sum()), exposure_days,
    )
    detail = {"tau": tau, "episodes": episodes, "valid3": valid3, "valid0": valid0, "premature": premature,
              "alerted": alerted, "eval3": eval3, "eval0": eval0, "exposure_days": data.eligible.sum(axis=1)/24}
    return result, detail


def patient_thresholds(data: TrajectoryData, by_hospital: dict[str, float | np.ndarray]) -> np.ndarray:
    out = np.empty_like(data.score_f, dtype=float)
    for hospital, threshold in by_hospital.items():
        mask = data.hospital == hospital
        out[mask] = threshold_matrix(data.score_f[mask], threshold)
    return out
