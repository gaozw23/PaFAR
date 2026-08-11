"""External-memory native XGBoost selection with patient-equal weights."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

from pafar_sim.io_utils import atomic_write_csv, atomic_write_json, file_checksum
from .feature_builder import fitting_mask
from .feature_cache import open_cache, patient_codes
from .imputation import FrozenPreprocessor
from .schema import RealDataConfig


@dataclass(frozen=True)
class SelectedLearner:
    booster: xgb.Booster
    params: dict[str, float | int | str]
    best_iteration: int
    best_score: float
    pi_y: float
    preprocessor_checksum: str
    source: str

    @property
    def trees_used(self) -> int:
        return self.best_iteration + 1


def _selected_codes(config: RealDataConfig, patient_ids: set[str]) -> set[int]:
    index = patient_codes(config)
    return set(index.loc[index.patient_id.isin(patient_ids), "patient_code"].astype(int))


def row_indices(config: RealDataConfig, hospital: str, patient_ids: set[str], *, fitting_only: bool = True) -> np.ndarray:
    with open_cache(config, hospital) as cache:
        codes = _selected_codes(config, patient_ids)
        selected = np.isin(cache["patient_code"], np.fromiter(codes, dtype=np.int32))
        if fitting_only:
            selected &= fitting_mask(cache["hours"], cache["onset"], config.tmin, config.hmax)
        return np.flatnonzero(selected)


def weight_metadata(config: RealDataConfig, selections: dict[str, set[str]], *, class_balance: bool) -> tuple[dict[str, np.ndarray], float]:
    indices: dict[str, np.ndarray] = {}
    patient_n: dict[int, int] = {}
    patient_pos: dict[int, int] = {}
    for hospital, ids in selections.items():
        idx = row_indices(config, hospital, ids, fitting_only=True)
        indices[hospital] = idx
        with open_cache(config, hospital) as cache:
            codes = np.asarray(cache["patient_code"])[idx]
            labels = np.asarray(cache["labels"])[idx]
        unique, counts = np.unique(codes, return_counts=True)
        positives = np.bincount(codes, weights=labels, minlength=int(codes.max()) + 1)
        patient_n.update({int(code): int(count) for code, count in zip(unique, counts)})
        patient_pos.update({int(code): int(positives[code]) for code in unique})
    fractions = [patient_pos[code] / patient_n[code] for code in sorted(patient_n)]
    pi_y = float(np.mean(fractions))
    if class_balance and not 0 < pi_y < 1:
        raise ValueError(f"patient-weighted positive fraction invalid: {pi_y}")
    weight_arrays: dict[str, np.ndarray] = {}
    all_weights: list[np.ndarray] = []
    for hospital, idx in indices.items():
        with open_cache(config, hospital) as cache:
            codes = np.asarray(cache["patient_code"])[idx]
            labels = np.asarray(cache["labels"])[idx].astype(float)
        counts = np.asarray([patient_n[int(code)] for code in codes], dtype=float)
        if class_balance:
            weights = (1 / counts) * (labels / (2*pi_y) + (1-labels) / (2*(1-pi_y)))
        else:
            weights = 1 / counts
        weight_arrays[hospital] = weights
        all_weights.append(weights)
    mean = float(np.concatenate(all_weights).mean())
    for hospital in weight_arrays:
        weight_arrays[hospital] = weight_arrays[hospital] / mean
    return weight_arrays, pi_y


class FeatureDataIter(xgb.DataIter):
    def __init__(
        self, config: RealDataConfig, indices: dict[str, np.ndarray], weights: dict[str, np.ndarray],
        preprocessor: FrozenPreprocessor, *, drop_hospital: bool, cache_prefix: str,
        batch_rows: int = 20000,
    ) -> None:
        super().__init__(cache_prefix=cache_prefix, release_data=True, on_host=True)
        self.config, self.preprocessor, self.drop_hospital = config, preprocessor, drop_hospital
        self.batches: list[tuple[str, np.ndarray, np.ndarray]] = []
        for hospital in sorted(indices):
            idx, w = indices[hospital], weights[hospital]
            for start in range(0, len(idx), batch_rows):
                self.batches.append((hospital, idx[start:start+batch_rows], w[start:start+batch_rows]))
        self._position = 0

    def reset(self) -> None:
        self._position = 0

    def next(self, input_data: Callable[..., None]) -> bool:
        if self._position >= len(self.batches):
            return False
        hospital, idx, weights = self.batches[self._position]
        with open_cache(self.config, hospital) as cache:
            x = self.preprocessor.transform(np.asarray(cache["values"])[idx], drop_hospital=self.drop_hospital)
            y = np.asarray(cache["labels"])[idx]
        input_data(data=x, label=y, weight=weights)
        self._position += 1
        return True


def create_matrix(
    config: RealDataConfig, selections: dict[str, set[str]], preprocessor: FrozenPreprocessor,
    *, class_balance: bool, drop_hospital: bool, cache_prefix: Path, ref: xgb.DMatrix | None = None,
) -> tuple[xgb.ExtMemQuantileDMatrix, float, dict[str, np.ndarray]]:
    weights, pi_y = weight_metadata(config, selections, class_balance=class_balance)
    indices = {h: row_indices(config, h, ids, fitting_only=True) for h, ids in selections.items()}
    iterator = FeatureDataIter(config, indices, weights, preprocessor, drop_hospital=drop_hospital, cache_prefix=str(cache_prefix))
    matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=256, nthread=config.xgb_fixed["nthread"], ref=ref)
    return matrix, pi_y, indices


def _selection_key(row: dict[str, object]) -> tuple[float, int, float, int, int]:
    return (-float(row["best_score"]), int(row["max_depth"]), float(row["eta"]), -int(row["min_child_weight"]), int(row["trees_used"]))


def fit_grid(
    config: RealDataConfig, train: dict[str, set[str]], validation: dict[str, set[str]],
    preprocessor: FrozenPreprocessor, *, seed: int, source: str, drop_hospital: bool,
    resume: bool = True,
) -> tuple[SelectedLearner, pd.DataFrame]:
    model_dir = config.outputs / "models" / source
    checkpoint_dir = config.outputs / "checkpoints" / source
    model_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dtrain, pi_y, _ = create_matrix(config, train, preprocessor, class_balance=True, drop_hospital=drop_hospital, cache_prefix=checkpoint_dir / "train")
    dval, _, _ = create_matrix(config, validation, preprocessor, class_balance=False, drop_hospital=drop_hospital, cache_prefix=checkpoint_dir / "validation", ref=dtrain)
    rows: list[dict[str, object]] = []
    process = psutil.Process()
    for grid_no, (depth, eta, child) in enumerate(config.xgb_grid):
        stem = f"grid_{grid_no:02d}_d{depth}_e{eta:.2f}_c{child}"
        model_path, meta_path = model_dir / f"{stem}.ubj", checkpoint_dir / f"{stem}.json"
        if resume and model_path.is_file() and meta_path.is_file():
            row = json.loads(meta_path.read_text(encoding="utf-8"))
            if row.get("config_checksum") == config.checksum and row.get("preprocessor_checksum") == preprocessor.checksum and row.get("model_sha256") == file_checksum(model_path):
                rows.append(row); continue
        params = {
            "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": config.xgb_fixed["tree_method"],
            "max_depth": depth, "eta": eta, "min_child_weight": child,
            "subsample": config.xgb_fixed["subsample"], "colsample_bytree": config.xgb_fixed["colsample_bytree"],
            "nthread": config.xgb_fixed["nthread"], "seed": int(seed % (2**31-1)), "max_bin": 256,
        }
        started = time.perf_counter()
        booster = xgb.train(
            params, dtrain, num_boost_round=int(config.xgb_fixed["num_boost_round"]),
            evals=[(dval, "validation")], early_stopping_rounds=int(config.xgb_fixed["early_stopping_rounds"]),
            verbose_eval=False,
        )
        elapsed = time.perf_counter() - started
        booster.save_model(model_path)
        row = {
            "grid_no": grid_no, "max_depth": depth, "eta": eta, "min_child_weight": child,
            "best_score": float(booster.best_score), "best_iteration": int(booster.best_iteration),
            "trees_used": int(booster.best_iteration)+1, "fitting_seconds": elapsed,
            "seed": int(seed), "peak_rss_bytes": process.memory_info().rss,
            "model_path": model_path.relative_to(config.root).as_posix(), "model_sha256": file_checksum(model_path),
            "config_checksum": config.checksum, "preprocessor_checksum": preprocessor.checksum,
        }
        atomic_write_json(row, meta_path)
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["grid_no"], kind="mergesort").reset_index(drop=True)
    if len(grid) != 18 or grid.best_iteration.isna().any():
        raise RuntimeError("XGBoost gate failed: incomplete grid or invalid best iteration")
    best = min(rows, key=_selection_key)
    booster = xgb.Booster()
    booster.load_model(config.root / str(best["model_path"]))
    selected_path = model_dir / "selected_model.ubj"
    booster.save_model(selected_path)
    atomic_write_csv(grid, config.outputs / ("internal_primary" if source == "internal" else "transfer_primary") / f"{source}_xgboost_grid.csv")
    atomic_write_json(best, model_dir / "selected_model.json")
    del dval, dtrain
    gc.collect()
    return SelectedLearner(booster, {k: best[k] for k in ("max_depth", "eta", "min_child_weight")}, int(best["best_iteration"]), float(best["best_score"]), pi_y, preprocessor.checksum, source), grid


def predict_rows(learner: SelectedLearner, preprocessor: FrozenPreprocessor, values: np.ndarray, *, drop_hospital: bool, batch_rows: int = 20000) -> np.ndarray:
    predictions = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), batch_rows):
        x = preprocessor.transform(np.asarray(values[start:start+batch_rows]), drop_hospital=drop_hospital)
        predictions[start:start+len(x)] = learner.booster.inplace_predict(x, iteration_range=(0, learner.trees_used)).astype(np.float32)
    return predictions


def fit_locked_configuration(
    config: RealDataConfig, train: dict[str,set[str]], validation: dict[str,set[str]],
    preprocessor: FrozenPreprocessor, selected_params: dict[str,float|int|str], *, seed: int,
    source: str, drop_hospital: bool,
) -> SelectedLearner:
    """Refit one prespecified configuration for split-sensitivity analysis."""
    cache_dir=config.outputs/"checkpoints"/source; model_dir=config.outputs/"models"/source
    cache_dir.mkdir(parents=True,exist_ok=True); model_dir.mkdir(parents=True,exist_ok=True)
    dtrain,pi_y,_=create_matrix(config,train,preprocessor,class_balance=True,drop_hospital=drop_hospital,cache_prefix=cache_dir/"train")
    dval,_,_=create_matrix(config,validation,preprocessor,class_balance=False,drop_hospital=drop_hospital,cache_prefix=cache_dir/"validation",ref=dtrain)
    params={"objective":"binary:logistic","eval_metric":"aucpr","tree_method":config.xgb_fixed["tree_method"],
            "max_depth":int(selected_params["max_depth"]),"eta":float(selected_params["eta"]),"min_child_weight":int(selected_params["min_child_weight"]),
            "subsample":config.xgb_fixed["subsample"],"colsample_bytree":config.xgb_fixed["colsample_bytree"],
            "nthread":config.xgb_fixed["nthread"],"seed":int(seed%(2**31-1)),"max_bin":256}
    booster=xgb.train(params,dtrain,num_boost_round=int(config.xgb_fixed["num_boost_round"]),evals=[(dval,"validation")],
                      early_stopping_rounds=int(config.xgb_fixed["early_stopping_rounds"]),verbose_eval=False)
    booster.save_model(model_dir/"selected_model.ubj")
    meta={"params":selected_params,"best_iteration":int(booster.best_iteration),"trees_used":int(booster.best_iteration)+1,
          "best_score":float(booster.best_score),"seed":int(seed),"preprocessor_checksum":preprocessor.checksum}
    atomic_write_json(meta,model_dir/"selected_model.json")
    del dval, dtrain
    gc.collect()
    return SelectedLearner(booster,dict(selected_params),int(booster.best_iteration),float(booster.best_score),pi_y,preprocessor.checksum,source)
