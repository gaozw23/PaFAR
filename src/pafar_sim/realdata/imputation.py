"""Training-only median fitting and frozen feature transformation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from pafar_sim.io_utils import atomic_write_json
from .schema import indicator_mask


@dataclass(frozen=True)
class FrozenPreprocessor:
    medians: np.ndarray
    add_missing: np.ndarray
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    checksum: str

    def transform(self, values: np.ndarray, *, drop_hospital: bool = False) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        if x.shape[1] != len(self.input_names):
            raise ValueError("feature width does not match frozen preprocessor")
        missing = ~np.isfinite(x)
        out = x.copy()
        out[missing] = np.broadcast_to(self.medians, out.shape)[missing]
        if self.add_missing.any():
            out = np.column_stack((out, missing[:, self.add_missing].astype(np.float32)))
        if drop_hospital:
            raw_hospital = self.input_names.index("hospital_B")
            # Missing indicators are appended, so removing the raw column preserves all later positions.
            out = np.delete(out, raw_hospital, axis=1)
        if not np.isfinite(out).all():
            raise AssertionError("frozen transform produced non-finite values")
        return out.astype(np.float32, copy=False)


def fit_preprocessor(
    matrices: list[np.ndarray], row_masks: list[np.ndarray], names: tuple[str, ...],
) -> FrozenPreprocessor:
    if len(matrices) != len(row_masks):
        raise ValueError("matrix/mask list mismatch")
    medians = np.zeros(len(names), dtype=np.float32)
    for j in range(len(names)):
        pieces = []
        for matrix, mask in zip(matrices, row_masks):
            column = np.asarray(matrix[:, j])[mask]
            finite = column[np.isfinite(column)]
            if finite.size:
                pieces.append(finite)
        if pieces:
            medians[j] = np.median(np.concatenate(pieces)).astype(np.float32)
        else:
            medians[j] = 0.0
    add = np.asarray(indicator_mask(names), dtype=bool)
    output = list(names) + [name + "__missing" for name, flag in zip(names, add) if flag]
    digest = sha256(medians.tobytes() + add.tobytes() + "\n".join(output).encode()).hexdigest()
    return FrozenPreprocessor(medians, add, names, tuple(output), digest)


def save_preprocessor(preprocessor: FrozenPreprocessor, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "medians.npy", preprocessor.medians, allow_pickle=False)
    atomic_write_json({
        "checksum": preprocessor.checksum,
        "input_names": list(preprocessor.input_names),
        "output_names": list(preprocessor.output_names),
        "add_missing": preprocessor.add_missing.tolist(),
    }, path / "preprocessor.json")

