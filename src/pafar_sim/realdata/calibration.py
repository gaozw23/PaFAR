"""Hospital-stratified calibration and exact binomial uncertainty."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from pafar_sim.calibration import (
    TimeTemplate, binwise_bonferroni_thresholds, fit_time_template, hc_threshold,
    marginal_threshold, naive_maximum_threshold, pointwise_threshold,
)
from pafar_sim.score import trajectory_max
from .scoring import TrajectoryData


@dataclass(frozen=True)
class MethodThreshold:
    method: str
    hospital: str
    alpha: float
    threshold: float | np.ndarray
    m0: int
    k: int | None
    alpha_m0: float | None
    infinite: bool
    boundaries: tuple[int, ...] | None = None


def exact_binomial(successes: int, n: int, confidence: float = .95) -> tuple[float, float, float]:
    if n <= 0:
        return np.nan, np.nan, np.nan
    tail = (1-confidence)/2
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n-successes+1))
    upper = 1.0 if successes == n else float(beta.ppf(1-tail, successes+1, n-successes))
    upper_one = 1.0 if successes == n else float(beta.ppf(confidence, successes+1, n-successes))
    return lower, upper, upper_one


def fit_template(validation: TrajectoryData, tmin: int, hmax: int, minimum: int) -> TimeTemplate:
    non = ~validation.event
    return fit_time_template(validation.score_f[non], validation.eligible[non], tmin, hmax, minimum)


def calibrate_hospital(
    calibration: TrajectoryData, template: TimeTemplate, hospital: str, alpha: float, delta: float,
    tmin: int, hmax: int,
) -> list[MethodThreshold]:
    mask = (calibration.hospital == hospital) & (~calibration.event)
    fixed, eligible = calibration.score_f[mask], calibration.eligible[mask]
    adaptive = template.transform(fixed)
    maxima_f, maxima_t = trajectory_max(fixed, eligible), trajectory_max(adaptive, eligible)
    marginal_f, marginal_t = marginal_threshold(maxima_f, alpha), marginal_threshold(maxima_t, alpha)
    hc = hc_threshold(maxima_f, alpha, delta)
    boundaries, bonf = binwise_bonferroni_thresholds(fixed, eligible, tmin, hmax, alpha)
    pointwise = pointwise_threshold(fixed, eligible, alpha)
    naive = naive_maximum_threshold(maxima_f, alpha)
    return [
        MethodThreshold("Pointwise-alpha", hospital, alpha, pointwise, len(maxima_f), None, None, bool(np.isposinf(pointwise))),
        MethodThreshold("Bonferroni", hospital, alpha, bonf, len(maxima_f), None, None, bool(np.isposinf(bonf).any()), boundaries),
        MethodThreshold("Naive maximum", hospital, alpha, naive, len(maxima_f), None, None, bool(np.isposinf(naive))),
        MethodThreshold("PaFAR-F", hospital, alpha, marginal_f.threshold, marginal_f.m0, marginal_f.index, marginal_f.alpha_m0, marginal_f.infinite),
        MethodThreshold("PaFAR-T", hospital, alpha, marginal_t.threshold, marginal_t.m0, marginal_t.index, marginal_t.alpha_m0, marginal_t.infinite),
        MethodThreshold("PaFAR-HC", hospital, alpha, hc.threshold, hc.m0, hc.index, hc.alpha_m0, hc.infinite),
    ]

