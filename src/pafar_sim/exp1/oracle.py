"""Chunked independent oracle construction and conditional PFA queries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..io_utils import atomic_write_json, file_checksum
from ..rng import make_rng
from ..score import causal_moving_average, clipped_logit, trajectory_max
from .dgp import generate_exp1


def oracle_metadata(
    scenario: str, site: str, nref: int, master_seed: int, *, hmax: int, tmin: int, smooth_length: int,
) -> dict[str, int | str | bool]:
    return {
        "scenario": scenario, "site": site, "nref": int(nref), "master_seed": int(master_seed),
        "hmax": int(hmax), "tmin": int(tmin), "smooth_length": int(smooth_length),
        "independent_reference": True,
    }


def oracle_filename(
    scenario: str, site: str, nref: int, master_seed: int, *, hmax: int, tmin: int, smooth_length: int,
) -> str:
    """Encode all oracle-defining inputs in the file name."""
    return f"{scenario}_{site}_t{tmin}_h{hmax}_L{smooth_length}_seed{master_seed}_N{nref}_maxima.npy"


def load_oracle(
    path: str | Path, scenario: str, site: str, nref: int, master_seed: int,
    *, hmax: int, tmin: int, smooth_length: int,
) -> NDArray[np.float64]:
    """Load an oracle only after exact metadata and content validation."""
    target = Path(path); sidecar = target.with_suffix(".json")
    if not target.is_file() or not sidecar.is_file():
        raise FileNotFoundError(target)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid oracle metadata: {sidecar}") from exc
    expected = oracle_metadata(scenario, site, nref, master_seed, hmax=hmax, tmin=tmin, smooth_length=smooth_length)
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"Stale/incompatible oracle metadata for {target}: {mismatches}")
    if metadata.get("file_checksum") != file_checksum(target):
        raise ValueError(f"Oracle checksum mismatch: {target}")
    values = np.load(target)
    if values.ndim != 1 or values.size != nref or np.isnan(values).any() or np.any(values[:-1] > values[1:]):
        raise ValueError(f"Invalid oracle maxima array: {target}")
    return np.asarray(values, dtype=np.float64)


def build_oracle(
    output: str | Path, scenario: str, site: str, nref: int, master_seed: int,
    *, hmax: int = 120, tmin: int = 6, smooth_length: int = 3, chunk_size: int = 25_000,
) -> NDArray[np.float64]:
    """Generate sorted float64 maxima in chunks, never storing an Nref-by-H matrix."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pieces: list[NDArray[np.float64]] = []
    generated = 0
    while generated < nref:
        size = min(chunk_size, nref - generated)
        rng = make_rng(master_seed, "experiment1", scenario, site, "oracle", generated)
        batch = generate_exp1(rng, size, scenario, False, site=site, hmax=hmax, tmin=tmin, patient_id_start=generated)
        u = clipped_logit(causal_moving_average(batch.risk, smooth_length))
        pieces.append(trajectory_max(u, batch.eligible))
        generated += size
    maxima = np.sort(np.concatenate(pieces).astype(np.float64))
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, maxima)
    temp.replace(target)
    atomic_write_json({
        **oracle_metadata(scenario, site, nref, master_seed, hmax=hmax, tmin=tmin, smooth_length=smooth_length),
        "file_checksum": file_checksum(target),
    }, target.with_suffix(".json"))
    return maxima


def oracle_threshold(sorted_maxima: NDArray[np.floating], alpha: float) -> tuple[float, int]:
    """Return the ceil(Nref*(1-alpha)) 1-based oracle order statistic."""
    values = np.asarray(sorted_maxima)
    k = int(np.ceil(values.size * (1 - alpha)))
    return float(values[k - 1]), k


def conditional_pfa(sorted_maxima: NDArray[np.floating], threshold: float) -> float:
    """Estimate strict deployment exceedance using one binary search."""
    values = np.asarray(sorted_maxima)
    return float((values.size - np.searchsorted(values, threshold, side="right")) / values.size)
