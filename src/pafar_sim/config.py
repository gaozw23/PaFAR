"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from copy import deepcopy
import json

import yaml


@dataclass(frozen=True)
class LoadedConfig:
    """Parsed YAML configuration plus its source checksum."""

    path: Path
    data: dict[str, Any]
    checksum: str


def load_config(path: str | Path) -> LoadedConfig:
    """Load a YAML configuration and enforce common statistical constraints."""
    config_path = Path(path).resolve()
    raw = config_path.read_bytes()
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    for key in ("master_seed", "alpha", "delta", "tmin", "hmax", "smooth_length"):
        if key not in data:
            raise ValueError(f"Missing required configuration key: {key}")
    if not 0 < float(data["alpha"]) < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not 0 < float(data["delta"]) < 1:
        raise ValueError("delta must be strictly between zero and one")
    if int(data["smooth_length"]) < 1:
        raise ValueError("smooth_length must be positive")
    if int(data["hmax"]) < int(data["tmin"]):
        raise ValueError("hmax must be at least tmin")
    return LoadedConfig(config_path, data, sha256(raw).hexdigest())


def project_root() -> Path:
    """Return the project root without relying on the invocation directory."""
    return Path(__file__).resolve().parents[2]


def apply_condition(data: dict[str, Any], condition: str | None) -> dict[str, Any]:
    """Deep-merge a named sensitivity condition into a base configuration."""
    result = deepcopy(data)
    conditions = result.pop("conditions", {})
    if condition is None:
        result.setdefault("condition", "primary")
        return result
    if condition not in conditions:
        raise ValueError(f"Unknown condition {condition!r}; available: {sorted(conditions)}")
    override = conditions[condition]
    if not isinstance(override, dict):
        raise ValueError(f"Condition {condition!r} must be a mapping")
    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    merge(result, override)
    result["condition"] = condition
    return result


def effective_config_checksum(data: dict[str, Any]) -> str:
    """Hash the fully resolved configuration, including condition/seed overrides."""
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()
