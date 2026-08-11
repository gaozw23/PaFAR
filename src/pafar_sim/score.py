"""Locked-score smoothing and trajectory reduction (Equations 2, 4--7)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def q_signal(u: NDArray[np.floating] | float) -> FloatArray:
    """Prespecified pre-event signal q(u), Equation (37)."""
    x = np.asarray(u, dtype=np.float64)
    return np.where(
        (x > 3) & (x <= 6), (6 - x) / 3,
        np.where((x > 0) & (x <= 3), 1.0, 0.0),
    )


def causal_moving_average(risk: NDArray[np.floating], length: int = 3) -> FloatArray:
    """Average current and prior probabilities, retaining pre-burn-in history."""
    values = np.asarray(risk, dtype=np.float64)
    if values.ndim != 2 or length < 1:
        raise ValueError("risk must be a 2-D array and length must be positive")
    valid = np.isfinite(values)
    sums = np.cumsum(np.where(valid, values, 0.0), axis=1)
    counts = np.cumsum(valid, axis=1)
    prev_sums = np.pad(sums[:, :-length], ((0, 0), (length, 0))) if values.shape[1] > length else np.zeros_like(values)
    prev_counts = np.pad(counts[:, :-length], ((0, 0), (length, 0))) if values.shape[1] > length else np.zeros_like(values, dtype=int)
    window_sums = sums - prev_sums
    window_counts = counts - prev_counts
    return np.divide(window_sums, window_counts, out=np.full_like(values, np.nan), where=window_counts > 0)


def clipped_logit(risk: NDArray[np.floating], epsilon: float = 1e-6) -> FloatArray:
    """Convert probabilities to clipped logits, Equation (5)."""
    values = np.asarray(risk, dtype=np.float64)
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def eligible_mask(
    horizon: NDArray[np.integer], onset: NDArray[np.floating], tmin: int, hmax: int
) -> BoolArray:
    """Construct the eligible monitoring set from Equation (2)."""
    h = np.asarray(horizon, dtype=int)
    t = np.arange(1, hmax + 1, dtype=int)[None, :]
    return (t >= tmin) & (t <= h[:, None]) & (t < np.asarray(onset)[:, None])


def trajectory_max(score: NDArray[np.floating], eligible: BoolArray) -> FloatArray:
    """Compute each patient's eligible maximum; empty sets map to -infinity."""
    values = np.where(eligible & np.isfinite(score), score, -np.inf)
    return np.max(values, axis=1)


def latent_to_risk(latent: NDArray[np.floating]) -> FloatArray:
    """Apply expit before smoothing, as required by the locked pipeline."""
    return expit(np.asarray(latent, dtype=np.float64))

