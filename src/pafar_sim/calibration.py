"""PaFAR and comparator calibration algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

from .score import trajectory_max


@dataclass(frozen=True)
class ThresholdResult:
    """A scalar calibration threshold and finite-sample metadata."""

    threshold: float
    index: int | None
    m0: int
    alpha: float
    alpha_m0: float | None
    delta: float | None = None

    @property
    def infinite(self) -> bool:
        return bool(np.isposinf(self.threshold))


@dataclass(frozen=True)
class TimeTemplate:
    """Frozen piecewise location-scale template fitted on validation non-events."""

    boundaries: tuple[int, ...]
    locations: tuple[float, ...]
    scales: tuple[float, ...]
    counts: tuple[int, ...]
    source: str = "validation_non_events"

    def bin_index(self, times: NDArray[np.integer]) -> NDArray[np.int_]:
        """Map integer decision times to [c0,c1], then (c_prev,c] bins."""
        return np.clip(np.searchsorted(np.asarray(self.boundaries[1:]), times, side="left"), 0, len(self.locations) - 1)

    def transform(self, score: NDArray[np.floating]) -> NDArray[np.float64]:
        """Standardize a patient-by-hour score matrix using the frozen template."""
        values = np.asarray(score, dtype=np.float64)
        times = np.arange(1, values.shape[1] + 1)
        idx = self.bin_index(times)
        loc = np.asarray(self.locations)[idx]
        scale = np.asarray(self.scales)[idx]
        return (values - loc[None, :]) / scale[None, :]


def minimum_marginal_m0(alpha: float) -> int:
    """Smallest m0 that can yield a finite non-augmented threshold."""
    return ceil(1.0 / alpha) - 1


def marginal_index(m0: int, alpha: float) -> tuple[int, float]:
    """Return the 1-based PaFAR order-statistic index and achieved bound."""
    if m0 < 0 or not 0 < alpha < 1:
        raise ValueError("m0 must be nonnegative and alpha must lie in (0,1)")
    k = ceil((m0 + 1) * (1 - alpha))
    return k, (m0 + 1 - k) / (m0 + 1)


def marginal_threshold(maxima: NDArray[np.floating], alpha: float) -> ThresholdResult:
    """Calibrate Equation (14), including the +infinity augmentation."""
    vals = np.asarray(maxima, dtype=np.float64)
    if vals.ndim != 1 or np.isnan(vals).any():
        raise ValueError("maxima must be a one-dimensional array without NaN")
    k, alpha_m0 = marginal_index(vals.size, alpha)
    threshold = np.inf if k == vals.size + 1 else float(np.partition(vals, k - 1)[k - 1])
    return ThresholdResult(threshold, k, vals.size, alpha, alpha_m0)


def hc_index(m0: int, alpha: float, delta: float) -> int:
    """Smallest tolerance-limit index satisfying Equation (24)."""
    if m0 < 0 or not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("invalid m0, alpha, or delta")
    candidates = np.arange(1, m0 + 2)
    ok = binom.sf(candidates - 1, m0, 1 - alpha) <= delta
    return int(candidates[np.flatnonzero(ok)[0]])


def hc_threshold(maxima: NDArray[np.floating], alpha: float, delta: float) -> ThresholdResult:
    """Calibrate the PaFAR-HC threshold from fixed-boundary maxima."""
    vals = np.asarray(maxima, dtype=np.float64)
    if vals.ndim != 1 or np.isnan(vals).any():
        raise ValueError("maxima must be one-dimensional without NaN")
    k = hc_index(vals.size, alpha, delta)
    threshold = np.inf if k == vals.size + 1 else float(np.partition(vals, k - 1)[k - 1])
    return ThresholdResult(threshold, k, vals.size, alpha, None, delta)


def initial_boundaries(tmin: int, hmax: int) -> tuple[int, ...]:
    """Create the prespecified unmerged PaFAR-T cut points."""
    if hmax < tmin:
        raise ValueError("hmax must be at least tmin")
    return tuple(sorted({tmin, hmax} | {x for x in (12, 24, 48, 72) if tmin < x < hmax}))


def bin_membership(times: NDArray[np.integer], boundaries: Sequence[int], bin_no: int) -> NDArray[np.bool_]:
    """Return membership under [c0,c1], then (c_prev,c] conventions."""
    t = np.asarray(times)
    lo, hi = boundaries[bin_no], boundaries[bin_no + 1]
    return (t >= lo) & (t <= hi) if bin_no == 0 else (t > lo) & (t <= hi)


def bin_patient_counts(eligible: NDArray[np.bool_], boundaries: Sequence[int]) -> list[int]:
    """Count patients contributing at least one eligible time to each bin."""
    times = np.arange(1, eligible.shape[1] + 1)
    return [int(np.any(eligible[:, bin_membership(times, boundaries, b)], axis=1).sum()) for b in range(len(boundaries) - 1)]


def merge_sparse_bins(
    eligible: NDArray[np.bool_], boundaries: Sequence[int], minimum: int = 50
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Repeatedly merge the latest sparse bin using the paper's direction rule."""
    cuts = list(boundaries)
    while len(cuts) > 2:
        counts = bin_patient_counts(eligible, cuts)
        sparse = [i for i, count in enumerate(counts) if count < minimum]
        if not sparse:
            break
        b = sparse[-1]
        if b > 0:
            del cuts[b]
        else:
            del cuts[1]
    return tuple(cuts), tuple(bin_patient_counts(eligible, cuts))


def weighted_quantile(values: NDArray[np.floating], weights: NDArray[np.floating], probability: float) -> float:
    """Smallest observed value whose normalized cumulative weight reaches probability."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[valid], w[valid]
    if x.size == 0 or w.sum() <= 0 or not 0 <= probability <= 1:
        raise ValueError("weighted quantile requires finite data, positive weight, and probability in [0,1]")
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    return float(x[np.searchsorted(np.cumsum(w), probability * w.sum(), side="left")])


def fit_time_template(
    score: NDArray[np.floating], eligible: NDArray[np.bool_], tmin: int, hmax: int, minimum: int = 50
) -> TimeTemplate:
    """Fit Equations (17)--(22) using patient-equal validation weights."""
    u = np.asarray(score, dtype=np.float64)
    if u.shape != eligible.shape:
        raise ValueError("score and eligible must have equal shapes")
    boundaries, counts = merge_sparse_bins(eligible, initial_boundaries(tmin, hmax), minimum)
    times = np.arange(1, u.shape[1] + 1)
    locations: list[float] = []
    scales: list[float] = []
    for b, patient_count in enumerate(counts):
        in_bin = bin_membership(times, boundaries, b)
        mask = eligible[:, in_bin] & np.isfinite(u[:, in_bin])
        contributors = np.flatnonzero(mask.any(axis=1))
        if patient_count == 0 or contributors.size == 0:
            locations.append(0.0); scales.append(1.0); continue
        vals: list[NDArray[np.float64]] = []
        weights: list[NDArray[np.float64]] = []
        block = u[:, in_bin]
        for patient in contributors:
            v = block[patient, mask[patient]]
            vals.append(v)
            weights.append(np.full(v.size, 1.0 / (contributors.size * v.size)))
        x, w = np.concatenate(vals), np.concatenate(weights)
        location = weighted_quantile(x, w, 0.5)
        mad = weighted_quantile(np.abs(x - location), w, 0.5)
        locations.append(location); scales.append(max(1.4826 * mad, 0.10))
    return TimeTemplate(boundaries, tuple(locations), tuple(scales), counts)


def pointwise_threshold(score: NDArray[np.floating], eligible: NDArray[np.bool_], alpha: float) -> float:
    """Patient-equal weighted pointwise-alpha comparator."""
    vals, weights = [], []
    for i in range(score.shape[0]):
        x = np.asarray(score[i])[eligible[i] & np.isfinite(score[i])]
        if x.size:
            vals.append(x); weights.append(np.full(x.size, 1.0 / x.size))
    if not vals:
        return np.inf
    return weighted_quantile(np.concatenate(vals), np.concatenate(weights), 1 - alpha)


def naive_maximum_threshold(maxima: NDArray[np.floating], alpha: float) -> float:
    """Hyndman-Fan type-7 quantile comparator."""
    vals = np.asarray(maxima, dtype=np.float64)
    return np.inf if vals.size == 0 else float(np.quantile(vals, 1 - alpha, method="linear"))


def binwise_bonferroni_thresholds(
    score: NDArray[np.floating], eligible: NDArray[np.bool_], tmin: int, hmax: int, alpha: float
) -> tuple[tuple[int, ...], NDArray[np.float64]]:
    """Calibrate unmerged-bin Bonferroni conformal thresholds."""
    boundaries = initial_boundaries(tmin, hmax)
    times = np.arange(1, score.shape[1] + 1)
    thresholds = []
    b0 = len(boundaries) - 1
    for b in range(b0):
        in_bin = bin_membership(times, boundaries, b)
        maxima = trajectory_max(score[:, in_bin], eligible[:, in_bin])
        thresholds.append(marginal_threshold(maxima, alpha / b0).threshold)
    return boundaries, np.asarray(thresholds)


def youden_threshold(
    risk: NDArray[np.floating], labels: NDArray[np.integer], patient_index: NDArray[np.integer]
) -> float:
    """Efficiently maximize patient-equal weighted sensitivity+specificity-1 with strict >."""
    x, y, p = np.asarray(risk), np.asarray(labels, dtype=int), np.asarray(patient_index, dtype=int)
    valid = np.isfinite(x)
    x, y, p = x[valid], y[valid], p[valid]
    if x.size == 0 or np.unique(y).size < 2:
        return np.inf
    counts = np.bincount(p, minlength=int(p.max()) + 1)
    w = 1.0 / counts[p]
    order = np.argsort(-x, kind="mergesort")
    xs, ys, ws = x[order], y[order], w[order]
    total_pos, total_neg = ws[ys == 1].sum(), ws[ys == 0].sum()
    cum_pos = np.cumsum(ws * (ys == 1)); cum_neg = np.cumsum(ws * (ys == 0))
    ends = np.r_[np.flatnonzero(xs[:-1] != xs[1:]), xs.size - 1]
    starts = np.r_[0, ends[:-1] + 1]
    above_pos = np.where(starts == 0, 0.0, cum_pos[np.maximum(starts - 1, 0)])
    above_neg = np.where(starts == 0, 0.0, cum_neg[np.maximum(starts - 1, 0)])
    thresholds = np.r_[np.inf, xs[starts], -np.inf]
    sens = np.r_[0.0, above_pos / total_pos, 1.0]
    spec = np.r_[1.0, 1.0 - above_neg / total_neg, 0.0]
    objective = sens + spec - 1.0
    best = np.flatnonzero(np.isclose(objective, objective.max(), rtol=0, atol=1e-14))
    return float(np.max(thresholds[best]))
