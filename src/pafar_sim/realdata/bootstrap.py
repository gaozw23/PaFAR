"""Whole-patient stratified bootstrap and paired method contrasts."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def stratified_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        pieces.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(pieces)


def bootstrap_statistics(
    patient_frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], dict[str, float]],
    *, strata: list[str], replicates: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    keys = patient_frame[strata].astype(str).agg("|".join, axis=1).to_numpy()
    rows = []
    for replicate in range(replicates):
        idx = stratified_indices(keys, rng)
        rows.append({"bootstrap": replicate, **statistic(patient_frame.iloc[idx].reset_index(drop=True))})
    return pd.DataFrame(rows)


def summarize_bootstrap(samples: pd.DataFrame, observed: dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, value in observed.items():
        x = samples[metric].to_numpy(float)
        x = x[np.isfinite(x)]
        rows.append({
            "metric": metric, "observed": value, "bootstrap_se": float(x.std(ddof=1)) if len(x)>1 else np.nan,
            "lower_95": float(np.quantile(x, .025)) if len(x) else np.nan,
            "upper_95": float(np.quantile(x, .975)) if len(x) else np.nan,
            "valid_bootstrap": len(x),
        })
    return pd.DataFrame(rows)

