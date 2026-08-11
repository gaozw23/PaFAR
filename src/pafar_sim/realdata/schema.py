"""Schema and immutable configuration objects for the real-data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml


LONGITUDINAL_COLUMNS = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
)
BASELINE_COLUMNS = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")
EXPECTED_COLUMNS = LONGITUDINAL_COLUMNS + BASELINE_COLUMNS + ("ICULOS", "SepsisLabel")
WINDOWS = (6, 12, 24)
WINDOW_STATS = ("mean", "min", "max", "sd", "change", "slope", "count")


@dataclass(frozen=True)
class RealDataConfig:
    root: Path
    raw_a: Path
    raw_b: Path
    expected_a: int
    expected_b: int
    master_seed: int
    tmin: int
    hmax: int
    wmax: int
    smooth_length: int
    epsilon: float
    scale_floor: float
    alphas: tuple[float, ...]
    delta: float
    template_bins: tuple[int, ...]
    template_min_patients: int
    target_m0: tuple[int, ...]
    bootstrap_replicates: int
    block_patients: int
    max_ram_fraction: float
    xgb_grid: tuple[tuple[int, float, int], ...]
    xgb_fixed: dict[str, Any]
    raw: dict[str, Any]
    checksum: str

    @property
    def outputs(self) -> Path:
        return self.root / "outputs" / "realdata"

    @property
    def data_root(self) -> Path:
        return self.root / "data" / "physionet2019"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_config(path: str | Path) -> RealDataConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root_spec = Path(raw.get("project_root", "."))
    root = (cfg_path.parent.parent / root_spec).resolve() if not root_spec.is_absolute() else root_spec.resolve()
    analysis, xgb = raw["analysis"], raw["xgboost"]
    grid = tuple(
        (int(depth), float(eta), int(child))
        for depth in xgb["max_depth"] for eta in xgb["eta"] for child in xgb["min_child_weight"]
    )
    fixed = {k: xgb[k] for k in (
        "subsample", "colsample_bytree", "num_boost_round",
        "early_stopping_rounds", "tree_method", "nthread",
    )}
    digest = sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    return RealDataConfig(
        root=root,
        raw_a=(root / raw["raw"]["A"]).resolve(),
        raw_b=(root / raw["raw"]["B"]).resolve(),
        expected_a=int(raw["expected_counts"]["A"]),
        expected_b=int(raw["expected_counts"]["B"]),
        master_seed=int(raw["master_seed"]),
        tmin=int(analysis["tmin"]), hmax=int(analysis["hmax"]),
        wmax=int(analysis["wmax"]), smooth_length=int(analysis["smooth_length"]),
        epsilon=float(analysis["epsilon"]), scale_floor=float(analysis["scale_floor"]),
        alphas=tuple(float(x) for x in analysis["alpha"]), delta=float(analysis["delta"]),
        template_bins=tuple(int(x) for x in analysis["template_bins"]),
        template_min_patients=int(analysis["template_min_patients"]),
        target_m0=tuple(int(x) for x in raw["target_m0"]),
        bootstrap_replicates=int(raw["bootstrap_replicates"]),
        block_patients=int(raw["feature_cache"]["block_patients"]),
        max_ram_fraction=float(raw["feature_cache"]["max_ram_fraction"]),
        xgb_grid=grid, xgb_fixed=fixed, raw=raw, checksum=digest,
    )


def raw_feature_names(include_hospital: bool = True) -> tuple[str, ...]:
    names = list(BASELINE_COLUMNS) + ["ICULOS"]
    if include_hospital:
        names.append("hospital_B")
    for variable in LONGITUDINAL_COLUMNS:
        names.extend((f"{variable}_last", f"{variable}_never", f"{variable}_since", f"{variable}_current"))
        for window in WINDOWS:
            names.extend(f"{variable}_{stat}_{window}h" for stat in WINDOW_STATS)
    return tuple(names)


def indicator_mask(names: tuple[str, ...]) -> tuple[bool, ...]:
    tokens = ("_last", "_since", "_mean_", "_min_", "_max_", "_sd_", "_change_", "_slope_")
    return tuple(any(token in name for token in tokens) for name in names)

