"""Strict first-alert and alert-episode algorithms."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def threshold_matrix(
    score: NDArray[np.floating], threshold: float | NDArray[np.floating]
) -> NDArray[np.float64]:
    """Broadcast scalar, hourly, or patient-hour thresholds."""
    values = np.asarray(score)
    c = np.asarray(threshold, dtype=np.float64)
    if c.ndim == 0:
        return np.full(values.shape, float(c))
    if c.ndim == 1 and c.size == values.shape[1]:
        return np.broadcast_to(c[None, :], values.shape)
    if c.shape == values.shape:
        return c
    raise ValueError("threshold must be scalar, hourly, or match score shape")


def first_alert(
    score: NDArray[np.floating], eligible: NDArray[np.bool_], threshold: float | NDArray[np.floating]
) -> NDArray[np.float64]:
    """Return first eligible strict crossing time, or +infinity (Equation 8)."""
    c = threshold_matrix(score, threshold)
    crossings = eligible & np.isfinite(score) & (score > c)
    any_cross = crossings.any(axis=1)
    first = np.argmax(crossings, axis=1) + 1
    return np.where(any_cross, first.astype(float), np.inf)


def count_alert_episodes(
    score: NDArray[np.floating], eligible: NDArray[np.bool_], threshold: float | NDArray[np.floating], separation: int = 6
) -> NDArray[np.int_]:
    """Count episodes after a return to/below boundary and a later strict upcrossing."""
    c = threshold_matrix(score, threshold)
    above = eligible & np.isfinite(score) & (score > c)
    out = np.zeros(score.shape[0], dtype=int)
    for i in range(score.shape[0]):
        starts = np.flatnonzero(above[i] & np.r_[True, ~above[i, :-1]])
        if starts.size == 0:
            continue
        accepted = [int(starts[0])]
        for start in starts[1:]:
            if start - accepted[-1] >= separation and np.any(~above[i, accepted[-1] + 1:start]):
                accepted.append(int(start))
        out[i] = len(accepted)
    return out


def hourly_bin_thresholds(hmax: int, boundaries: tuple[int, ...], values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Expand binwise thresholds to one value per integer hour."""
    times = np.arange(1, hmax + 1)
    idx = np.clip(np.searchsorted(np.asarray(boundaries[1:]), times, side="left"), 0, len(values) - 1)
    return np.asarray(values, dtype=float)[idx]

