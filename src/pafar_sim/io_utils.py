"""Atomic checkpoints, checksums, and run manifests."""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rng import bit_generator_name


def file_checksum(path: str | Path) -> str:
    """Compute a SHA-256 checksum without loading a large file at once."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def implementation_checksum() -> str:
    """Hash implementation sources that determine checkpoint semantics."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "pafar_sim").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    digest = sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(data: Any, path: str | Path) -> None:
    """Write strict JSON through a sibling temporary file and atomic replace."""
    from .realdata.json_utils import encode_json_numeric

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(encode_json_numeric(data), stream, indent=2, sort_keys=True, allow_nan=False)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically write CSV or gzip-compressed CSV according to the suffix."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if target.name.endswith(".csv.gz") else ".csv"
    fd, name = tempfile.mkstemp(prefix=target.stem + ".", suffix=suffix + ".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        compression = "gzip" if target.name.endswith(".gz") else None
        frame.to_csv(temp, index=False, compression=compression)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def checkpoint_complete(path: str | Path, config_checksum: str) -> bool:
    """Validate a finished checkpoint and its sidecar checksum record."""
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".json")
    if not target.is_file() or not sidecar.is_file() or ".tmp" in target.name:
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("complete") is True
        and meta.get("config_checksum") == config_checksum
        and meta.get("implementation_checksum") == implementation_checksum()
        and meta.get("file_checksum") == file_checksum(target)
    )


def write_checkpoint(frame: pd.DataFrame, path: str | Path, config_checksum: str) -> None:
    """Write an independent replicate checkpoint and completion sidecar."""
    target = Path(path)
    atomic_write_csv(frame, target)
    atomic_write_json(
        {
            "complete": True,
            "config_checksum": config_checksum,
            "implementation_checksum": implementation_checksum(),
            "file_checksum": file_checksum(target),
        },
        target.with_suffix(target.suffix + ".json"),
    )


def environment_manifest(root: str | Path, config_checksum: str, master_seed: int) -> dict[str, Any]:
    """Capture environment, RNG, configuration, and immutable-PDF checksums."""
    root_path = Path(root)
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "joblib", "yaml"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "bit_generator": bit_generator_name(),
        "master_seed": int(master_seed),
        "config_checksum": config_checksum,
        "pdf_checksums": {
            "PaFAR.pdf": file_checksum(root_path / "PaFAR.pdf"),
            "v40i08.pdf": file_checksum(root_path / "v40i08.pdf"),
        },
        "production_simulation_run": False,
    }


def json_value(value: Any) -> str:
    """Serialize vector-valued threshold metadata into a CSV-safe string."""
    from .realdata.json_utils import dumps_json_numeric

    return dumps_json_numeric(value, separators=(",", ":"))
