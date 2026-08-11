"""Causal irregular-EHR feature extraction with training-frozen imputation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dgp import Exp2Batch


WINDOWS = (6, 12, 24)
WINDOW_STATS = ("mean", "min", "max", "sd", "change", "slope", "count")


@dataclass(frozen=True)
class RawFeatureRows:
    values: NDArray[np.float32]
    labels: NDArray[np.int_]
    patient_index: NDArray[np.int_]
    hour: NDArray[np.int_]
    names: tuple[str, ...]


@dataclass(frozen=True)
class FeaturePreprocessor:
    """Training-fitted medians and final column order."""

    medians: NDArray[np.float32]
    add_missing: NDArray[np.bool_]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    @classmethod
    def fit(cls, rows: RawFeatureRows) -> "FeaturePreprocessor":
        x = np.asarray(rows.values, dtype=np.float32)
        medians = np.zeros(x.shape[1], dtype=np.float32)
        add_missing = np.zeros(x.shape[1], dtype=bool)
        output = list(rows.names)
        imputed_tokens = ("_last", "_since", "_mean_", "_min_", "_max_", "_sd_", "_change_", "_slope_")
        for j, name in enumerate(rows.names):
            finite = np.isfinite(x[:, j])
            if finite.any():
                medians[j] = np.median(x[finite, j])
            # Freeze indicators by feature semantics, not by an accidental absence of
            # missing values in one training draw; later splits may still be undefined.
            add_missing[j] = any(token in name for token in imputed_tokens)
            if add_missing[j]:
                output.append(name + "__missing")
        return cls(medians, add_missing, rows.names, tuple(output))

    def transform(self, rows: RawFeatureRows) -> NDArray[np.float32]:
        """Apply training medians and append deterministic missingness indicators."""
        if rows.names != self.input_names:
            raise ValueError("Feature columns do not match the training-frozen order")
        x = np.asarray(rows.values, dtype=np.float32).copy()
        missing = ~np.isfinite(x)
        x[missing] = np.broadcast_to(self.medians, x.shape)[missing]
        if self.add_missing.any():
            x = np.column_stack((x, missing[:, self.add_missing].astype(np.float32)))
        return x.astype(np.float32, copy=False)


def feature_names(p: int = 12) -> tuple[str, ...]:
    """Return the stable raw feature schema."""
    names = ["A", "Q", "elapsed_hour"]
    for j in range(p):
        prefix = f"x{j + 1}"
        names.extend((f"{prefix}_last", f"{prefix}_never", f"{prefix}_since", f"{prefix}_current"))
        for window in WINDOWS:
            names.extend(f"{prefix}_{stat}_{window}h" for stat in WINDOW_STATS)
    return tuple(names)


def _window_statistics(values: NDArray[np.floating], absolute_times: NDArray[np.floating]) -> tuple[NDArray[np.float32], ...]:
    """Compute seven window summaries across patients without patient loops."""
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.nansum(values, axis=1)
    mean = np.divide(total, count, out=np.full(values.shape[0], np.nan), where=count > 0)
    minimum = np.min(np.where(valid, values, np.inf), axis=1); minimum[count == 0] = np.nan
    maximum = np.max(np.where(valid, values, -np.inf), axis=1); maximum[count == 0] = np.nan
    centered = np.where(valid, values - mean[:, None], 0.0)
    sd = np.sqrt(np.divide(np.sum(centered**2, axis=1), count - 1, out=np.full(values.shape[0], np.nan), where=count > 1))
    first_idx = np.argmax(valid, axis=1)
    last_idx = values.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    rows = np.arange(values.shape[0])
    change = values[rows, last_idx] - values[rows, first_idx]
    change[count < 2] = np.nan
    t = np.broadcast_to(absolute_times, values.shape)
    t_mean = np.divide(np.sum(np.where(valid, t, 0), axis=1), count, out=np.zeros(values.shape[0]), where=count > 0)
    denominator = np.sum(np.where(valid, (t - t_mean[:, None])**2, 0), axis=1)
    numerator = np.sum(np.where(valid, (t - t_mean[:, None]) * (values - mean[:, None]), 0), axis=1)
    slope = np.divide(numerator, denominator, out=np.full(values.shape[0], np.nan), where=denominator > 0)
    return tuple(np.asarray(x, dtype=np.float32) for x in (mean, minimum, maximum, sd, change, slope, count))


def score_history_mask(batch: Exp2Batch, tmin: int, smooth_length: int) -> NDArray[np.bool_]:
    """Rows needed to score every eligible hour with its locked causal history."""
    if smooth_length < 1:
        raise ValueError("smooth_length must be positive")
    times = np.arange(1, batch.measurements.shape[1] + 1)[None, :]
    first = max(1, int(tmin) - int(smooth_length) + 1)
    return (times >= first) & (times <= batch.horizon[:, None]) & (times < batch.onset[:, None])


def build_raw_features(batch: Exp2Batch, row_mask: NDArray[np.bool_] | None = None) -> RawFeatureRows:
    """Build causal features for an explicit patient-hour row mask."""
    n, hmax, p = batch.measurements.shape
    selected = batch.eligible if row_mask is None else np.asarray(row_mask, dtype=bool)
    if selected.shape != batch.eligible.shape:
        raise ValueError("row_mask must match batch.eligible")
    names = feature_names(p)
    total_rows = int(selected.sum())
    x = np.empty((total_rows, len(names)), dtype=np.float32)
    y = np.empty(total_rows, dtype=int)
    patient_index = np.empty(total_rows, dtype=int)
    hours = np.empty(total_rows, dtype=int)
    last_value = np.full((n, p), np.nan, dtype=np.float32)
    last_time = np.full((n, p), -1, dtype=int)
    offset = 0
    for t0 in range(hmax):
        now = batch.measurements[:, t0, :]
        observed_now = np.isfinite(now)
        last_value = np.where(observed_now, now, last_value)
        last_time = np.where(observed_now, t0 + 1, last_time)
        idx = np.flatnonzero(selected[:, t0])
        if idx.size == 0:
            continue
        block = np.empty((idx.size, len(names)), dtype=np.float32)
        block[:, :3] = np.column_stack((batch.age_covariate[idx], batch.binary_covariate[idx], np.full(idx.size, t0 + 1)))
        column = 3
        for biomarker in range(p):
            never = last_time[:, biomarker] < 0
            since = np.where(never, np.nan, (t0 + 1) - last_time[:, biomarker])
            block[:, column:column + 4] = np.column_stack((
                last_value[idx, biomarker], never[idx], since[idx], observed_now[idx, biomarker]
            ))
            column += 4
            for window in WINDOWS:
                lo = max(0, t0 - window + 1)  # integer form of (t-w,t]
                values = batch.measurements[:, lo:t0 + 1, biomarker]
                stats = _window_statistics(values, np.arange(lo + 1, t0 + 2, dtype=float))
                block[:, column:column + len(stats)] = np.column_stack([stat[idx] for stat in stats])
                column += len(stats)
        end = offset + idx.size
        x[offset:end] = block
        y[offset:end] = (batch.event[idx] & ((batch.onset[idx] - (t0 + 1)) > 0) & ((batch.onset[idx] - (t0 + 1)) <= 6)).astype(int)
        patient_index[offset:end] = idx
        hours[offset:end] = t0 + 1
        offset = end
    if offset != total_rows:
        raise RuntimeError("Feature row preallocation count mismatch")
    return RawFeatureRows(x, y, patient_index, hours, names)
