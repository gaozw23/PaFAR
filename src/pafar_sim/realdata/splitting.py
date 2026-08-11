"""Deterministic patient-level stratified splitting and seed registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def seed_registry(master_seed: int) -> dict[str, object]:
    names = [
        "internal_primary", *[f"internal_additional_{i:02d}" for i in range(1, 10)],
        "A_to_B_source", "B_to_A_source", "A_to_B_target_primary", "B_to_A_target_primary",
        *[f"target_additional_{i:02d}" for i in range(1, 50)], "target_reservoir_order",
        "xgboost", "bootstrap", "subgroups",
    ]
    children = np.random.SeedSequence(master_seed).spawn(len(names))
    return {name: int(child.generate_state(1, dtype=np.uint64)[0]) for name, child in zip(names, children)}


def _rounded_counts(n: int, proportions: tuple[float, ...]) -> list[int]:
    raw = np.asarray(proportions, dtype=float) * n
    base = np.floor(raw).astype(int)
    remainder = n - int(base.sum())
    order = np.argsort(-(raw - base), kind="mergesort")
    base[order[:remainder]] += 1
    return base.tolist()


def stratified_patient_split(
    patients: pd.DataFrame, proportions: dict[str, float], seed: int,
    strata: tuple[str, ...] = ("hospital_set", "any_sepsis_label"),
) -> pd.DataFrame:
    names = tuple(proportions)
    probs = tuple(float(proportions[name]) for name in names)
    if not np.isclose(sum(probs), 1.0):
        raise ValueError("split proportions must sum to one")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows: list[pd.DataFrame] = []
    for _, block in patients.groupby(list(strata), sort=True, dropna=False):
        indices = rng.permutation(len(block))
        counts = _rounded_counts(len(block), probs)
        start = 0
        for name, count in zip(names, counts):
            part = block.iloc[indices[start:start+count]].copy()
            part["split"] = name
            rows.append(part)
            start += count
    result = pd.concat(rows, ignore_index=True).sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    validate_split(result, patients, names)
    return result


def validate_split(result: pd.DataFrame, patients: pd.DataFrame, names: Iterable[str]) -> None:
    if result.patient_id.duplicated().any():
        raise ValueError("patient overlap across splits")
    if set(result.patient_id) != set(patients.patient_id):
        raise ValueError("split union differs from cohort")
    for name in names:
        block = result[result.split == name]
        if block.empty or block.any_sepsis_label.nunique() < 2:
            raise ValueError(f"split {name} lacks one event class")


def nested_non_event_prefixes(reservoir: pd.DataFrame, sizes: Iterable[int], seed: int) -> dict[int, tuple[str, ...]]:
    non = reservoir.loc[~reservoir.any_sepsis_label.astype(bool), "patient_id"].to_numpy()
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    order = non[rng.permutation(len(non))]
    out: dict[int, tuple[str, ...]] = {}
    for size in sorted(set(int(x) for x in sizes)):
        if size > len(order):
            raise ValueError(f"target reservoir has only {len(order)} non-events, needs {size}")
        out[size] = tuple(order[:size].tolist())
    prior: set[str] = set()
    for size in sorted(out):
        current = set(out[size])
        if not prior.issubset(current):
            raise AssertionError("target prefixes are not nested")
        prior = current
    return out

