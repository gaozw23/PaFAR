"""Experiment II latent EHR data-generating process (Equations 40--46)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from ..score import eligible_mask, q_signal
from ..exp1.dgp import draw_monitoring_lengths


LOADINGS = np.asarray([1.0, 0.8, -0.7, 0.6, 0.5, -0.5, 0.4, 0.3, 0, 0, 0, 0])
GAMMA = np.asarray([0.25] * 8 + [0.10] * 4)
THETA_A = np.zeros(12); THETA_A[[1, 4, 8]] = 0.15
THETA_Q = np.zeros(12); THETA_Q[[2, 6]] = 0.20
OMEGA_E2 = np.asarray([0.20, 0.15, -0.15, 0.12, 0.10, -0.10, 0, 0, 0, 0, 0, 0])


@dataclass(frozen=True)
class Exp2Batch:
    patient_ids: NDArray[np.int_]
    horizon: NDArray[np.int_]
    event: NDArray[np.bool_]
    onset: NDArray[np.float64]
    age_covariate: NDArray[np.float64]
    binary_covariate: NDArray[np.int_]
    random_effect: NDArray[np.float64]
    severity: NDArray[np.float64]
    measurements: NDArray[np.float32]
    observed: NDArray[np.bool_]
    eligible: NDArray[np.bool_]


@dataclass(frozen=True)
class NonEventSamplingInfo:
    requested: int
    candidates: int
    accepted: int
    acceptance_rate: float


def event_probabilities(age: NDArray[np.floating], binary: NDArray[np.integer], random_effect: NDArray[np.floating]) -> NDArray[np.float64]:
    """Conditional event probabilities from Equation (40)."""
    return expit(-2.3845 + 0.30 * np.asarray(age) + 0.20 * np.asarray(binary) + 0.35 * np.asarray(random_effect))


def latent_severity(
    rng: np.random.Generator, random_effect: NDArray[np.floating], event: NDArray[np.bool_],
    onset: NDArray[np.floating], hmax: int, delta_s: float,
) -> NDArray[np.float64]:
    """Generate the stationary AR(1) severity process from Equation (41)."""
    n = len(random_effect)
    c = np.empty((n, hmax), dtype=np.float64)
    c[:, 0] = rng.normal(0, 0.35 / np.sqrt(1 - 0.85**2), n)
    innovations = rng.normal(0, 0.35, (n, hmax - 1))
    for t in range(1, hmax):
        c[:, t] = 0.85 * c[:, t - 1] + innovations[:, t - 1]
    times = np.arange(1, hmax + 1)[None, :]
    return 0.40 * np.asarray(random_effect)[:, None] + c + np.asarray(event)[:, None] * delta_s * q_signal(np.asarray(onset)[:, None] - times)


def generate_exp2(
    rng: np.random.Generator, n: int, scenario: str, *, site: str = "A", hmax: int = 120,
    tmin: int = 6, wmax: int = 6, delta_s: float = 1.0, force_event: bool | None = None,
    patient_id_start: int = 0,
) -> Exp2Batch:
    """Generate baseline, latent biomarkers, and informative missingness in vectorized arrays."""
    if scenario not in {"E1", "E2", "E3"}:
        raise ValueError(f"Unknown Experiment II scenario: {scenario}")
    age = rng.normal(size=n)
    binary = rng.binomial(1, 0.5, size=n).astype(int)
    random_effect = rng.normal(size=n)
    event = rng.random(n) < event_probabilities(age, binary, random_effect)
    if force_event is not None:
        event[:] = force_event
    horizon = draw_monitoring_lengths(rng, n, "mixed", target_b=False)
    onset = np.full(n, np.inf)
    idx = np.flatnonzero(event)
    if idx.size:
        onset[idx] = tmin + wmax + np.floor(rng.random(idx.size) * (horizon[idx] - tmin - wmax + 1))
    severity = latent_severity(rng, random_effect, event, onset, hmax, delta_s)
    p = LOADINGS.size
    sigma_a = np.r_[np.full(8, 0.50), np.full(4, 0.70)]
    target_b = scenario == "E3" and site.upper() == "B"
    sigma = sigma_a * (1.10 if target_b else 1.0)
    kappa = np.where(np.arange(p) < 8, 0.20 * np.sign(LOADINGS), 0.0) if target_b else np.zeros(p)
    omega = OMEGA_E2 if scenario == "E2" else np.zeros(p)
    noise = np.empty((n, hmax, p), dtype=np.float64)
    noise[:, 0, :] = rng.normal(size=(n, p)) * sigma
    innovations = rng.normal(size=(n, hmax - 1, p)) * sigma[None, None, :] * np.sqrt(1 - 0.60**2)
    for t in range(1, hmax):
        noise[:, t, :] = 0.60 * noise[:, t - 1, :] + innovations[:, t - 1, :]
    times = np.arange(1, hmax + 1, dtype=float)
    latent_x = (
        severity[:, :, None] * LOADINGS + random_effect[:, None, None] * GAMMA
        + age[:, None, None] * THETA_A + binary[:, None, None] * THETA_Q
        + kappa + np.log1p(times[None, :, None] / 24) * omega + noise
    )
    moderate = scenario in {"E1", "E3"}
    p0 = np.r_[np.full(4, 0.70 if moderate else 0.45), np.full(8, 0.18 if moderate else 0.08)]
    informative = scenario in {"E2", "E3"}
    d = 0.50 if informative else 0.0
    xi = np.r_[np.full(4, 0.10), np.full(8, -0.10)] if target_b else np.zeros(p)
    observation_probability = expit(logit(p0)[None, None, :] + d * np.maximum(severity[:, :, None], 0) + xi)
    valid = times[None, :, None] <= horizon[:, None, None]
    observed = (rng.random((n, hmax, p)) < observation_probability) & valid
    measurements = np.where(observed, latent_x, np.nan).astype(np.float32)
    eligible = eligible_mask(horizon, onset, tmin, hmax)
    severity = np.where(times[None, :] <= horizon[:, None], severity, np.nan)
    return Exp2Batch(
        np.arange(patient_id_start, patient_id_start + n), horizon, event, onset, age, binary,
        random_effect, severity, measurements, observed, eligible,
    )


def _take_exp2(batch: Exp2Batch, indices: NDArray[np.integer]) -> Exp2Batch:
    return Exp2Batch(*(np.asarray(getattr(batch, field))[indices] for field in batch.__dataclass_fields__))


def _combine_exp2(parts: list[Exp2Batch], n: int) -> Exp2Batch:
    arrays = []
    for field in Exp2Batch.__dataclass_fields__:
        arrays.append(np.concatenate([np.asarray(getattr(part, field)) for part in parts], axis=0)[:n])
    arrays[0] = np.arange(n, dtype=int)
    return Exp2Batch(*arrays)


def generate_exp2_non_events(
    rng: np.random.Generator, n: int, scenario: str, *, order_rng: np.random.Generator,
    site: str = "B", hmax: int = 120, tmin: int = 6, wmax: int = 6,
    delta_s: float = 1.0, candidate_chunk_size: int | None = None,
) -> tuple[Exp2Batch, NonEventSamplingInfo]:
    """Sample exactly ``n`` patients from the natural conditional D=0 population."""
    if n < 1:
        raise ValueError("n must be positive")
    chunk = int(candidate_chunk_size or max(256, min(4096, 2 * n)))
    accepted_parts: list[Exp2Batch] = []
    candidates = accepted = 0
    while accepted < n:
        candidate = generate_exp2(
            rng, chunk, scenario, site=site, hmax=hmax, tmin=tmin, wmax=wmax,
            delta_s=delta_s, force_event=None, patient_id_start=candidates,
        )
        candidates += chunk
        indices = np.flatnonzero(~candidate.event)
        if indices.size:
            selected = _take_exp2(candidate, indices)
            accepted_parts.append(selected)
            accepted += len(indices)
    # Draw ordering from a dedicated stream, so nested prefixes are stable and do
    # not depend on learner/test/source-calibration streams.
    combined = _combine_exp2(accepted_parts, accepted)
    order = order_rng.permutation(accepted)[:n]
    reservoir = _take_exp2(combined, order)
    reservoir = Exp2Batch(np.arange(n, dtype=int), *(getattr(reservoir, f) for f in list(Exp2Batch.__dataclass_fields__)[1:]))
    if reservoir.event.any() or not np.isposinf(reservoir.onset).all():
        raise RuntimeError("conditional non-event sampler retained an event")
    return reservoir, NonEventSamplingInfo(n, candidates, accepted, accepted / candidates)
