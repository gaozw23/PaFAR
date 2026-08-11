"""Native-XGBoost fitting, weighting, and locked best-iteration prediction."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray
import xgboost as xgb


@dataclass(frozen=True)
class FittedLearner:
    booster: xgb.Booster
    best_iteration: int
    best_score: float
    fitting_seconds: float

    @property
    def prediction_end(self) -> int:
        return self.best_iteration + 1

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """Predict using exactly the early-stopping-selected tree range."""
        return self.booster.predict(xgb.DMatrix(np.asarray(x, dtype=np.float32)), iteration_range=(0, self.prediction_end))


def patient_average_positive_fraction(labels: NDArray[np.integer], patient_index: NDArray[np.integer]) -> float:
    """Average each patient's positive-row fraction with equal patient weight."""
    y, p = np.asarray(labels), np.asarray(patient_index, dtype=int)
    patients = np.unique(p)
    return float(np.mean([np.mean(y[p == patient]) for patient in patients]))


def training_weights(labels: NDArray[np.integer], patient_index: NDArray[np.integer]) -> tuple[NDArray[np.float64], float]:
    """Equation (32) patient/class-balanced weights, rescaled to row mean one."""
    y, p = np.asarray(labels), np.asarray(patient_index, dtype=int)
    counts = np.bincount(p, minlength=int(p.max()) + 1)
    if np.any(counts[np.unique(p)] <= 0):
        raise ValueError("Every fitting patient must have positive Kfit")
    pi_hat = patient_average_positive_fraction(y, p)
    if not 0 < pi_hat < 1:
        raise ValueError(f"Training labels do not support class balancing: piY_hat={pi_hat}")
    weights = (1.0 / counts[p]) * (y / (2 * pi_hat) + (1 - y) / (2 * (1 - pi_hat)))
    return weights / weights.mean(), pi_hat


def validation_weights(patient_index: NDArray[np.integer]) -> NDArray[np.float64]:
    """Patient-equal validation row weights, jointly rescaled to mean one."""
    p = np.asarray(patient_index, dtype=int)
    counts = np.bincount(p, minlength=int(p.max()) + 1)
    weights = 1.0 / counts[p]
    return weights / weights.mean()


def fit_xgboost(
    train_x: NDArray[np.floating], train_y: NDArray[np.integer], train_patient: NDArray[np.integer],
    validation_x: NDArray[np.floating], validation_y: NDArray[np.integer], validation_patient: NDArray[np.integer],
    *, seed: int, num_boost_round: int, early_stopping_rounds: int,
) -> tuple[FittedLearner, float]:
    """Fit native XGBoost with validation as the only early-stopping evaluation set."""
    if np.unique(train_y).size < 2:
        raise ValueError("training labels contain only one class")
    if np.unique(validation_y).size < 2:
        raise ValueError("validation labels contain only one class")
    train_w, pi_hat = training_weights(train_y, train_patient)
    validation_w = validation_weights(validation_patient)
    dtrain = xgb.DMatrix(np.asarray(train_x, dtype=np.float32), label=train_y, weight=train_w)
    dvalidation = xgb.DMatrix(np.asarray(validation_x, dtype=np.float32), label=validation_y, weight=validation_w)
    params = {
        "objective": "binary:logistic", "max_depth": 3, "eta": 0.05,
        "min_child_weight": 5, "subsample": 0.80, "colsample_bytree": 0.80,
        "eval_metric": "aucpr", "nthread": 1, "seed": int(seed % (2**31 - 1)),
    }
    started = time.perf_counter()
    booster = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=[(dvalidation, "validation")], early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    elapsed = time.perf_counter() - started
    return FittedLearner(booster, int(booster.best_iteration), float(booster.best_score), elapsed), pi_hat

