"""Verbatim-official PhysioNet utility adapter with separate utility grid."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from pafar_sim.io_utils import file_checksum
from .scoring import TrajectoryData


OFFICIAL_SHA256 = "d0f65da3d42ce68cad80e290050bce4b8f2efc7ad3f13c0a1f70a331fbd8ff06"


def load_official_scorer(path: str | Path):
    scorer = Path(path)
    if file_checksum(scorer) != OFFICIAL_SHA256:
        raise RuntimeError("official utility scorer checksum mismatch")
    spec = importlib.util.spec_from_file_location("pafar_official_physionet2019_scorer", scorer)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utility_components(data: TrajectoryData, predictions: np.ndarray, scorer_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    module = load_official_scorer(scorer_path)
    observed = np.zeros(len(data.patient_id)); inactive = np.zeros(len(data.patient_id)); optimal = np.zeros(len(data.patient_id))
    for i in range(len(data.patient_id)):
        n = int(min(data.horizon[i], data.labels.shape[1]))
        labels = data.labels[i, :n].astype(int)
        pred = np.where(data.utility_grid[i, :n], predictions[i, :n], 0).astype(int)
        best = np.zeros(n, dtype=int)
        if np.any(labels):
            dt_early, dt_optimal, dt_late = -12, -6, 3
            t_sepsis = int(np.argmax(labels) - dt_optimal)
            best[max(0, t_sepsis+dt_early):min(t_sepsis+dt_late+1, n)] = 1
        observed[i] = module.compute_prediction_utility(labels, pred)
        inactive[i] = module.compute_prediction_utility(labels, np.zeros(n, dtype=int))
        optimal[i] = module.compute_prediction_utility(labels, best)
    return observed, inactive, optimal


def normalized_utility(data: TrajectoryData, predictions: np.ndarray, scorer_path: str | Path) -> tuple[float, float]:
    observed, inactive, optimal = utility_components(data, predictions, scorer_path)
    denominator = float(optimal.sum()-inactive.sum())
    return (float((observed.sum()-inactive.sum())/denominator) if denominator else np.nan), float(observed.sum())
