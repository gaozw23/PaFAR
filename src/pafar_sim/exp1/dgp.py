"""Vectorized Experiment I data-generating process (Equations 33--39)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..score import eligible_mask, latent_to_risk, q_signal


@dataclass(frozen=True)
class Exp1Batch:
    patient_ids: NDArray[np.int_]
    horizon: NDArray[np.int_]
    event: NDArray[np.bool_]
    onset: NDArray[np.float64]
    latent: NDArray[np.float64]
    risk: NDArray[np.float64]
    eligible: NDArray[np.bool_]


def draw_monitoring_lengths(
    rng: np.random.Generator, n: int, distribution: str, target_b: bool = False
) -> NDArray[np.int_]:
    """Draw short or mixed monitoring length from Equations (33)--(34)."""
    short = np.clip(24 + rng.poisson(18, n), 24, 72)
    if distribution == "short":
        return short.astype(int)
    if distribution != "mixed":
        raise ValueError(f"Unknown stay distribution: {distribution}")
    long = np.clip(60 + rng.poisson(24, n), 48, 120)
    short_probability = 0.60 if target_b else 0.70
    return np.where(rng.random(n) < short_probability, short, long).astype(int)


def stationary_ar1(
    rng: np.random.Generator, n: int, hmax: int, rho: float, marginal_sd: float
) -> NDArray[np.float64]:
    """Generate stationary AR(1) innovations, vectorized across patients."""
    process = np.empty((n, hmax), dtype=np.float64)
    process[:, 0] = rng.normal(0, marginal_sd, n)
    scale = marginal_sd * np.sqrt(1 - rho * rho)
    innovations = rng.normal(0, scale, (n, hmax - 1))
    for t in range(1, hmax):
        process[:, t] = rho * process[:, t - 1] + innovations[:, t - 1]
    return process


def scenario_parameters(scenario: str, site: str = "A", signal: float = 1.5) -> dict[str, float | str | bool]:
    """Resolve the prespecified Experiment I scenario/site parameters."""
    if scenario not in {"S1", "S2", "S3", "S4"}:
        raise ValueError(f"Unknown Experiment I scenario: {scenario}")
    beta_t = 0.40 if scenario == "S3" else 0.0
    distribution = "short" if scenario == "S1" else "mixed"
    target_b = scenario == "S4" and site.upper() == "B"
    return {
        "beta0": -2.8 if target_b else -3.0,
        "beta_t": beta_t + (0.10 if target_b else 0.0),
        "sigma_e": 0.55 * (1.10 if target_b else 1.0),
        "distribution": distribution, "target_b": target_b, "signal": signal,
    }


def generate_exp1(
    rng: np.random.Generator, n: int, scenario: str, event_status: bool | NDArray[np.bool_],
    *, site: str = "A", hmax: int = 120, tmin: int = 6, wmax: int = 6,
    signal: float = 1.5, patient_id_start: int = 0,
) -> Exp1Batch:
    """Generate a class-specific locked score-trajectory batch."""
    pars = scenario_parameters(scenario, site, signal)
    horizon = draw_monitoring_lengths(rng, n, str(pars["distribution"]), bool(pars["target_b"]))
    event = np.full(n, event_status, dtype=bool) if np.ndim(event_status) == 0 else np.asarray(event_status, dtype=bool)
    if event.size != n:
        raise ValueError("event_status length must equal n")
    onset = np.full(n, np.inf)
    event_idx = np.flatnonzero(event)
    if event_idx.size:
        lows = tmin + wmax
        uniforms = rng.random(event_idx.size)
        onset[event_idx] = lows + np.floor(uniforms * (horizon[event_idx] - lows + 1))
    random_effect = rng.normal(0, 0.45, n)
    errors = stationary_ar1(rng, n, hmax, 0.80, float(pars["sigma_e"]))
    times = np.arange(1, hmax + 1, dtype=float)[None, :]
    lead = onset[:, None] - times
    signal_term = event[:, None] * float(pars["signal"]) * q_signal(lead)
    latent = (
        float(pars["beta0"]) + float(pars["beta_t"]) * np.log1p(times / 24)
        + random_effect[:, None] + errors + signal_term
    )
    valid = times <= horizon[:, None]
    latent = np.where(valid, latent, np.nan)
    eligible = eligible_mask(horizon, onset, tmin, hmax)
    return Exp1Batch(
        np.arange(patient_id_start, patient_id_start + n), horizon, event, onset,
        latent, latent_to_risk(latent), eligible,
    )

