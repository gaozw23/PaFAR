"""Recovery invariants and the second-generation real-data analysis lock."""
from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import pandas as pd

from pafar_sim.io_utils import atomic_write_csv, atomic_write_json, file_checksum
from .feature_cache import compute_cache_id, feature_definition_checksum
from .raw_io import verify_raw_unchanged
from .schema import RealDataConfig, canonical_json
from .splitting import seed_registry, stratified_patient_split
from .utility import OFFICIAL_SHA256


FAILED_INPUT_SHA256 = {
    "patient_manifest.csv.gz": "50787657ecc16e1cb1255209b76d0df62067f67bf73ae452baec9675eeaadabc",
    "primary_cohort_ids.csv": "c31277e654ff7babd23f693df176c703e8338728ad3154c25002bbf031d5ba60",
    "exclusion_log.csv": "5d410566eff4553adc21871630776b6435b0ba84e3f9ba81671a43db0fc33238",
    "raw_file_manifest.csv.gz": "c646a94a64220e7661760c03c3cacfe20bc305479461e6510b5409c742cdeb52",
    "raw_file_manifest.json": "86f271c3fabe990775ff8da2f752062212871589b47ac4467af578a281f0ba52",
}

FEATURE_PRODUCING_MODULE_SHA256 = {
    "raw_io.py": "6d8de1840571eea7a66a33636aa17d152ac74f5f7d5af31fe651145d492daaa0",
    "cohort.py": "ea4649e8008876ba4848e4e4c2befb17c42e1cf223ee8c121edc953fc3e734b0",
    "splitting.py": "b68caf978c0fdf2234f79c5368619b26406948523ad1313a3dc5d5042ae3b438",
    "feature_builder.py": "c8e64ad438b95251d0964932a9268bdd1e85a0a7e27245b0cfb901572d7ff92d",
    "feature_cache.py": "a9e1de1e36f2f2715671fc079b28d658be008501377963047044ba824186507d",
    "imputation.py": "cf99f10b2232b2bb0f92df017e6bacd3d2d3363b8de4d8cc2a7ed99ada97450e",
}


def feature_producing_module_checksums(config: RealDataConfig) -> dict[str, str]:
    """Hash only modules capable of changing the frozen feature cache."""
    base = config.root / "src" / "pafar_sim" / "realdata"
    return {name: file_checksum(base / name) for name in FEATURE_PRODUCING_MODULE_SHA256}


def load_frozen_patient_manifest(config: RealDataConfig) -> pd.DataFrame:
    paths = {
        "patient_manifest.csv.gz": config.data_root / "processed" / "patient_manifest.csv.gz",
        "primary_cohort_ids.csv": config.data_root / "processed" / "primary_cohort_ids.csv",
        "exclusion_log.csv": config.data_root / "processed" / "exclusion_log.csv",
        "raw_file_manifest.csv.gz": config.data_root / "manifests" / "raw_file_manifest.csv.gz",
        "raw_file_manifest.json": config.data_root / "manifests" / "raw_file_manifest.json",
    }
    changed = {}
    for name, path in paths.items():
        observed = file_checksum(path) if path.is_file() else "missing"
        if observed != FAILED_INPUT_SHA256[name]:
            changed[name] = {"expected": FAILED_INPUT_SHA256[name], "observed": observed}
    if changed:
        raise RuntimeError(f"frozen raw/cohort artifacts changed: {changed}")
    manifest = pd.read_csv(paths["patient_manifest.csv.gz"], float_precision="round_trip")
    primary = manifest[manifest.primary_cohort]
    counts = primary.groupby(["hospital_set", "any_sepsis_label"]).size().to_dict()
    expected = {("A", False): 18546, ("A", True): 1585, ("B", False): 18858, ("B", True): 906}
    if len(primary) != 39895 or counts != expected:
        raise RuntimeError(f"locked primary cohort counts changed: n={len(primary)}, counts={counts}")
    return manifest


def cohort_manifest_checksum(config: RealDataConfig) -> str:
    payload = {name: FAILED_INPUT_SHA256[name] for name in (
        "patient_manifest.csv.gz", "primary_cohort_ids.csv", "exclusion_log.csv"
    )}
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def split_assignment_checksum(split: pd.DataFrame) -> str:
    columns = ["patient_id", "hospital_set", "any_sepsis_label", "split"]
    ordered = split[columns].copy().sort_values("patient_id", kind="mergesort")
    text = "\n".join("|".join(map(str, row)) for row in ordered.itertuples(index=False, name=None))
    return sha256(text.encode("utf-8")).hexdigest()


def load_fixed_internal_split(config: RealDataConfig, manifest: pd.DataFrame) -> pd.DataFrame:
    path = config.data_root / "processed" / "internal_primary_split.csv"
    cohort = manifest.loc[
        manifest.primary_cohort,
        ["patient_id", "hospital_set", "any_sepsis_label", "reconstructed_onset", "last_ICULOS"],
    ].copy()
    expected = stratified_patient_split(
        cohort, config.raw["split"]["internal"], seed_registry(config.master_seed)["internal_primary"]
    )
    if path.is_file():
        observed = pd.read_csv(path, float_precision="round_trip")
        if split_assignment_checksum(observed) != split_assignment_checksum(expected):
            raise RuntimeError("persisted internal primary split differs from the prespecified deterministic assignment")
        return observed
    # The failed run stopped before persisting a split.  Materialize the one and
    # only deterministic assignment implied by the already-locked seed/design.
    atomic_write_csv(expected, path)
    return expected


def memmap_implementation_checksum(config: RealDataConfig) -> str:
    digest = sha256()
    for relative in (
        "src/pafar_sim/realdata/memmap_utils.py",
        "src/pafar_sim/realdata/feature_cache.py",
    ):
        path = config.root / relative
        digest.update(relative.encode("ascii")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def create_lock_v2(
    config: RealDataConfig, manifest: pd.DataFrame, *,
    source_checksum: str, effective_config_checksum: str,
) -> dict[str, Any]:
    failed_lock = config.outputs / "REALDATA_LOCK.json"
    raw_summary = json.loads((config.data_root / "manifests" / "raw_file_manifest.json").read_text(encoding="utf-8"))
    split = load_fixed_internal_split(config, manifest)
    split_checksum = split_assignment_checksum(split)
    cohort_checksum = cohort_manifest_checksum(config)
    feature_checksum = feature_definition_checksum(config)
    cache_id = compute_cache_id(
        raw_manifest_checksum=raw_summary["aggregate_manifest_sha256"],
        cohort_checksum=cohort_checksum, split_checksum=split_checksum,
        feature_checksum=feature_checksum, source_checksum=source_checksum,
        config_checksum=effective_config_checksum, master_seed=config.master_seed,
    )
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "joblib", "pyyaml", "psutil"):
        packages[name] = importlib.metadata.version(name)
    simulation = json.loads((config.outputs / "manifests" / "simulation_protected_baseline.json").read_text(encoding="utf-8"))
    lock = {
        "lock_version": 2,
        "status": "locked_for_fresh_cache_build",
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "parent_failed_lock_sha256": file_checksum(failed_lock),
        "raw_manifest_checksum": raw_summary["aggregate_manifest_sha256"],
        "raw_manifest_file_sha256": FAILED_INPUT_SHA256["raw_file_manifest.csv.gz"],
        "cohort_manifest_checksum": cohort_checksum,
        "cohort_artifact_sha256": {key: FAILED_INPUT_SHA256[key] for key in (
            "patient_manifest.csv.gz", "primary_cohort_ids.csv", "exclusion_log.csv"
        )},
        "internal_split_checksum": split_checksum,
        "internal_split_seed": seed_registry(config.master_seed)["internal_primary"],
        "source_checksum": source_checksum,
        "effective_config_checksum": effective_config_checksum,
        "memmap_implementation_checksum": memmap_implementation_checksum(config),
        "feature_definition_checksum": feature_checksum,
        "master_seed": config.master_seed,
        "cache_id": cache_id,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": packages,
        "official_scorer_checksum": OFFICIAL_SHA256,
        "simulation_aggregate_checksum": simulation["aggregate_sha256"],
        "invariants": {
            "raw_manifest_unchanged": True,
            "primary_cohort_unchanged": True,
            "split_seed_unchanged": True,
            "split_assignments_deterministic_and_frozen": True,
            "statistical_definitions_unchanged": True,
            "engineering_cache_implementation_only_change": True,
        },
    }
    target = config.outputs / "REALDATA_LOCK_V2.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in existing.items() if k != "timestamp_utc"}
        current = {k: v for k, v in lock.items() if k != "timestamp_utc"}
        if comparable != current:
            raise RuntimeError("existing REALDATA_LOCK_V2 does not match current recovery identity")
        return existing
    atomic_write_json(lock, target)
    return lock


def verify_lock_v2(
    config: RealDataConfig, *, source_checksum: str, effective_config_checksum: str,
) -> dict[str, Any]:
    path = config.outputs / "REALDATA_LOCK_V2.json"
    if not path.is_file():
        raise RuntimeError("REALDATA_LOCK_V2.json is missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "source_checksum": source_checksum,
        "effective_config_checksum": effective_config_checksum,
        "memmap_implementation_checksum": memmap_implementation_checksum(config),
        "master_seed": config.master_seed,
    }
    mismatches = {key: (lock.get(key), value) for key, value in expected.items() if lock.get(key) != value}
    if mismatches:
        raise RuntimeError(f"V2 code/config/seed changed after lock: {mismatches}")
    return lock


def create_lock_v2_1(
    config: RealDataConfig, manifest: pd.DataFrame, *,
    source_checksum: str, effective_config_checksum: str,
) -> dict[str, Any]:
    """Create an engineering lock for the serialization-only source change."""
    parent_path = config.outputs / "REALDATA_LOCK_V2.json"
    if not parent_path.is_file():
        raise RuntimeError("parent REALDATA_LOCK_V2.json is missing")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    split = load_fixed_internal_split(config, manifest)
    module_after = feature_producing_module_checksums(config)
    if module_after != FEATURE_PRODUCING_MODULE_SHA256:
        raise RuntimeError(f"feature-producing modules changed: {module_after}")
    expected_parent = {
        "master_seed": config.master_seed,
        "effective_config_checksum": effective_config_checksum,
        "cohort_manifest_checksum": cohort_manifest_checksum(config),
        "internal_split_checksum": split_assignment_checksum(split),
        "feature_definition_checksum": feature_definition_checksum(config),
        "memmap_implementation_checksum": memmap_implementation_checksum(config),
    }
    mismatches = {key: (parent.get(key), value) for key, value in expected_parent.items() if parent.get(key) != value}
    if mismatches:
        raise RuntimeError(f"V2 parent is incompatible with V2.1: {mismatches}")
    cache_root = config.data_root / "cache" / f"realdata_features_{parent['cache_id']}"
    marker_path = cache_root / "CACHE_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_cache = {
        "cache_id": parent["cache_id"], "status": "complete",
        "patient_count": 39895, "row_count": 1448195,
        "feature_count": 857, "actual_block_count": 161,
    }
    cache_mismatches = {key: (marker.get(key), value) for key, value in expected_cache.items() if marker.get(key) != value}
    if cache_mismatches:
        raise RuntimeError(f"frozen feature cache marker changed: {cache_mismatches}")
    lock = {
        "lock_version": "2.1",
        "status": "locked_for_serialization_fix_resume",
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "parent_lock": "outputs/realdata/REALDATA_LOCK_V2.json",
        "parent_lock_sha256": file_checksum(parent_path),
        "master_seed": parent["master_seed"],
        "raw_manifest_checksum": parent["raw_manifest_checksum"],
        "cohort_manifest_checksum": parent["cohort_manifest_checksum"],
        "internal_split_checksum": parent["internal_split_checksum"],
        "feature_cache_id": parent["cache_id"],
        "feature_definition_checksum": parent["feature_definition_checksum"],
        "effective_config_checksum": parent["effective_config_checksum"],
        "old_source_checksum": parent["source_checksum"],
        "new_source_checksum": source_checksum,
        "patch_scope": "output serialization only",
        "feature_producing_module_checksums_before": FEATURE_PRODUCING_MODULE_SHA256,
        "feature_producing_module_checksums_after": module_after,
        "feature_producing_modules_unchanged": True,
        "cache_completion_marker_sha256": file_checksum(marker_path),
        "cache_invariants": expected_cache,
        "simulation_aggregate_checksum": parent["simulation_aggregate_checksum"],
    }
    atomic_write_json(lock, config.outputs / "REALDATA_LOCK_V2_1.json")
    return lock


def verify_lock_v2_1(
    config: RealDataConfig, *, source_checksum: str, effective_config_checksum: str,
) -> dict[str, Any]:
    """Verify V2 cache compatibility while binding the patched output code."""
    path = config.outputs / "REALDATA_LOCK_V2_1.json"
    if not path.is_file():
        raise RuntimeError("REALDATA_LOCK_V2_1.json is missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    parent_path = config.outputs / "REALDATA_LOCK_V2.json"
    module_checksums = feature_producing_module_checksums(config)
    expected = {
        "new_source_checksum": source_checksum,
        "effective_config_checksum": effective_config_checksum,
        "master_seed": config.master_seed,
        "parent_lock_sha256": file_checksum(parent_path),
        "feature_producing_module_checksums_after": module_checksums,
        "feature_cache_id": json.loads(parent_path.read_text(encoding="utf-8"))["cache_id"],
        "patch_scope": "output serialization only",
    }
    mismatches = {key: (lock.get(key), value) for key, value in expected.items() if lock.get(key) != value}
    if module_checksums != FEATURE_PRODUCING_MODULE_SHA256:
        mismatches["feature_producing_modules"] = (module_checksums, FEATURE_PRODUCING_MODULE_SHA256)
    marker = config.data_root / "cache" / f"realdata_features_{lock.get('feature_cache_id')}" / "CACHE_COMPLETE.json"
    if not marker.is_file() or file_checksum(marker) != lock.get("cache_completion_marker_sha256"):
        mismatches["cache_completion_marker_sha256"] = (
            file_checksum(marker) if marker.is_file() else "missing",
            lock.get("cache_completion_marker_sha256"),
        )
    if mismatches:
        raise RuntimeError(f"V2.1 engineering lock mismatch: {mismatches}")
    return lock


def verify_raw_from_frozen_manifest(config: RealDataConfig) -> tuple[bool, list[str]]:
    raw = pd.read_csv(config.data_root / "manifests" / "raw_file_manifest.csv.gz", dtype={"sha256": "string"})
    return verify_raw_unchanged(config, raw)
