"""Versioned, immutable, Windows-safe real-data feature caches."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil

from pafar_sim.io_utils import atomic_write_csv, file_checksum
from .feature_builder import build_patient_features, fitting_mask
from .memmap_utils import (
    assert_no_open_files,
    bounded_replace,
    close_memmap_tree,
    flush_and_close_memmap,
    open_files_under,
)
from .raw_io import read_patient
from .schema import RealDataConfig, canonical_json, raw_feature_names


ARRAY_SPECS: tuple[tuple[str, Any, bool], ...] = (
    ("values", np.float32, True),
    ("patient_code", np.int32, False),
    ("hours", np.int16, False),
    ("labels", np.uint8, False),
    ("onset", np.float32, False),
    ("horizon", np.int16, False),
    ("event", np.uint8, False),
)


@dataclass(frozen=True)
class CachePaths:
    hospital: str
    root: Path
    values: Path
    patient_code: Path
    hours: Path
    labels: Path
    onset: Path
    horizon: Path
    event: Path
    index: Path


class CacheHandle(dict[str, Any]):
    """Dictionary-compatible context manager for a group of read-only maps."""

    def close(self) -> None:
        close_memmap_tree(self)
        for key in tuple(self):
            if key != "meta":
                self.pop(key, None)

    def __enter__(self) -> "CacheHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _write_small_json(payload: dict[str, Any], target: Path) -> list[dict[str, Any]]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return bounded_replace(Path(name), target)
    except BaseException:
        try:
            Path(name).unlink(missing_ok=True)
        finally:
            raise


def feature_definition_checksum(config: RealDataConfig) -> str:
    builder = config.root / "src" / "pafar_sim" / "realdata" / "feature_builder.py"
    digest = sha256(builder.read_bytes())
    digest.update("\n".join(raw_feature_names(True)).encode("utf-8"))
    digest.update(canonical_json({
        "tmin": config.tmin,
        "hmax": config.hmax,
        "smooth_length": config.smooth_length,
        "include_hospital": True,
    }).encode("utf-8"))
    return digest.hexdigest()


def compute_cache_id(
    *, raw_manifest_checksum: str, cohort_checksum: str, split_checksum: str,
    feature_checksum: str, source_checksum: str, config_checksum: str,
    master_seed: int,
) -> str:
    payload = {
        "raw_manifest_checksum": raw_manifest_checksum,
        "cohort_checksum": cohort_checksum,
        "split_checksum": split_checksum,
        "feature_checksum": feature_checksum,
        "source_checksum": source_checksum,
        "effective_config_checksum": config_checksum,
        "master_seed": int(master_seed),
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_directory(config: RealDataConfig, cache_id: str | None = None) -> Path:
    if cache_id is None:
        lock_path = config.outputs / "REALDATA_LOCK_V2.json"
        if not lock_path.is_file():
            raise RuntimeError("REALDATA_LOCK_V2.json is required to resolve the active feature cache")
        cache_id = json.loads(lock_path.read_text(encoding="utf-8"))["cache_id"]
    return config.data_root / "cache" / f"realdata_features_{cache_id}"


def cache_paths(
    config: RealDataConfig, hospital: str, *, cache_root: Path | None = None,
) -> CachePaths:
    root = cache_root or cache_directory(config)
    prefix = root / "arrays" / f"features_{hospital}"
    return CachePaths(
        hospital=hospital, root=root,
        values=prefix.with_suffix(".values.npy"),
        patient_code=prefix.with_suffix(".patient.npy"),
        hours=prefix.with_suffix(".hours.npy"),
        labels=prefix.with_suffix(".labels.npy"),
        onset=prefix.with_suffix(".onset.npy"),
        horizon=prefix.with_suffix(".horizon.npy"),
        event=prefix.with_suffix(".event.npy"),
        index=root / "metadata" / f"hospital_{hospital}.json",
    )


def _shape_for(name: str, n_rows: int, n_features: int) -> tuple[int, ...]:
    return (n_rows, n_features) if name == "values" else (n_rows,)


def _array_paths(paths: CachePaths) -> dict[str, Path]:
    return {name: getattr(paths, name) for name, _, _ in ARRAY_SPECS}


def _block_checksum(arrays: dict[str, np.memmap], start: int, stop: int) -> str:
    digest = sha256()
    for name, _, _ in ARRAY_SPECS:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(arrays[name][start:stop]).tobytes(order="C"))
    return digest.hexdigest()


def _build_hospital_cache(
    config: RealDataConfig, patient_block: pd.DataFrame, hospital: str,
    root: Path, names: tuple[str, ...],
) -> dict[str, Any]:
    paths = cache_paths(config, hospital, cache_root=root)
    paths.values.parent.mkdir(parents=True, exist_ok=True)
    (root / "blocks").mkdir(parents=True, exist_ok=True)
    n_rows = int(sum(
        max(0, int(row.last_ICULOS) - max(1, config.tmin - config.smooth_length + 1) + 1)
        for row in patient_block.itertuples()
    ))
    arrays: dict[str, np.memmap] = {}
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    offset = 0
    block_records: list[dict[str, Any]] = []
    try:
        for name, dtype, fortran in ARRAY_SPECS:
            arrays[name] = np.lib.format.open_memmap(
                _array_paths(paths)[name], mode="w+", dtype=dtype,
                shape=_shape_for(name, n_rows, len(names)), fortran_order=fortran,
            )
        for block_no, start in enumerate(range(0, len(patient_block), config.block_patients)):
            patients = patient_block.iloc[start:start + config.block_patients]
            row_start = offset
            for row in patients.itertuples(index=False):
                frame = read_patient(config.root / row.source_file)
                features = build_patient_features(
                    frame, hospital=hospital, onset=float(row.reconstructed_onset),
                    tmin=config.tmin, hmax=int(row.last_ICULOS),
                    smooth_length=config.smooth_length, include_hospital=True,
                )
                end = offset + len(features.hours)
                arrays["values"][offset:end] = features.values
                arrays["patient_code"][offset:end] = int(row.patient_code)
                arrays["hours"][offset:end] = features.hours
                arrays["labels"][offset:end] = features.labels
                arrays["onset"][offset:end] = float(row.reconstructed_onset)
                arrays["horizon"][offset:end] = int(row.last_ICULOS)
                arrays["event"][offset:end] = int(bool(row.any_sepsis_label))
                offset = end
            row_stop = offset
            record = {
                "hospital": hospital, "block": block_no,
                "patient_start": start,
                "patient_stop": min(start + config.block_patients, len(patient_block)),
                "patient_count": len(patients),
                "row_start": row_start, "row_stop": row_stop,
                "row_count": row_stop - row_start,
                "first_patient_id": str(patients.patient_id.iloc[0]),
                "last_patient_id": str(patients.patient_id.iloc[-1]),
                "content_sha256": _block_checksum(arrays, row_start, row_stop),
            }
            _write_small_json(record, root / "blocks" / f"{hospital}_block_{block_no:04d}.json")
            block_records.append(record)
            peak_rss = max(peak_rss, process.memory_info().rss)
        if offset != n_rows:
            raise RuntimeError(f"feature row mismatch for hospital {hospital}: {offset} != {n_rows}")
    finally:
        flush_and_close_memmap(arrays)
        arrays.clear()
        gc.collect()
    assert_no_open_files(root)
    array_meta: dict[str, Any] = {}
    for name, dtype, fortran in ARRAY_SPECS:
        path = _array_paths(paths)[name]
        array_meta[name] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_checksum(path),
            "dtype": np.dtype(dtype).str,
            "shape": list(_shape_for(name, n_rows, len(names))),
            "fortran_order": bool(fortran),
        }
    metadata = {
        "status": "built", "hospital": hospital,
        "n_patients": int(len(patient_block)), "n_rows": n_rows,
        "n_features": len(names), "feature_names": list(names),
        "feature_name_checksum": sha256("\n".join(names).encode()).hexdigest(),
        "expected_block_count": math.ceil(len(patient_block) / config.block_patients),
        "actual_block_count": len(block_records),
        "blocks": block_records, "arrays": array_meta,
        "peak_rss_bytes": int(peak_rss),
        "elapsed_seconds": time.perf_counter() - started,
        "writer_pid": os.getpid(), "open_files_after_close": open_files_under(root),
        "worker_return_contains_memmap": False,
    }
    _write_small_json(metadata, paths.index)
    return metadata


def _load_marker(root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    complete = root / "CACHE_COMPLETE.json"
    if not complete.is_file():
        building = root / "CACHE_BUILDING.json"
        state = "CACHE_BUILDING only" if building.is_file() else "missing"
        if require_complete:
            raise RuntimeError(f"feature cache is incomplete ({state}): {root}")
        return {}
    marker = json.loads(complete.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError(f"invalid cache completion marker: {complete}")
    return marker


def commit_completion_marker(root: Path, marker: dict[str, Any], *, validated: bool) -> list[dict[str, Any]]:
    """Commit the sole cache-validity marker after validation and handle audit."""
    target = root / "CACHE_COMPLETE.json"
    if not validated:
        raise RuntimeError("completion marker cannot be written before validation")
    if target.exists():
        raise RuntimeError("completion marker already exists")
    assert_no_open_files(root)
    return _write_small_json(marker, target)


def _open_cache_at(config: RealDataConfig, hospital: str, root: Path) -> CacheHandle:
    marker = _load_marker(root)
    paths = cache_paths(config, hospital, cache_root=root)
    meta = json.loads(paths.index.read_text(encoding="utf-8"))
    if marker.get("cache_id") != root.name.removeprefix("realdata_features_"):
        raise RuntimeError("cache directory and completion-marker IDs disagree")
    handle = CacheHandle(meta=meta)
    try:
        for name, _, _ in ARRAY_SPECS:
            handle[name] = np.load(_array_paths(paths)[name], mmap_mode="r", allow_pickle=False)
        return handle
    except BaseException:
        handle.close()
        raise


def open_cache(config: RealDataConfig, hospital: str) -> CacheHandle:
    return _open_cache_at(config, hospital, cache_directory(config))


def _validate_cache(
    config: RealDataConfig, manifest: pd.DataFrame, root: Path, *,
    cache_id: str, require_complete: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if require_complete:
        marker = _load_marker(root)
        if marker.get("cache_id") != cache_id:
            raise RuntimeError("requested cache_id differs from completion marker")
    names = raw_feature_names(True)
    index_path = root / "metadata" / "patient_index.csv.gz"
    index = pd.read_csv(index_path, float_precision="round_trip")
    expected = manifest.loc[manifest.primary_cohort].sort_values(
        ["hospital_set", "patient_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(index) != len(expected) or index.patient_id.astype(str).tolist() != expected.patient_id.astype(str).tolist():
        raise RuntimeError("cache patient index does not match the locked primary cohort")
    validation_rows: list[dict[str, Any]] = []
    total_rows = total_blocks = total_bytes = 0
    peak_rss = psutil.Process().memory_info().rss
    reference_error = 0.0
    for hospital in ("A", "B"):
        paths = cache_paths(config, hospital, cache_root=root)
        meta = json.loads(paths.index.read_text(encoding="utf-8"))
        expected_hospital = index[index.hospital_set == hospital]
        if meta["n_patients"] != len(expected_hospital):
            raise RuntimeError(f"patient count mismatch for {hospital}")
        arrays = CacheHandle(meta=meta)
        try:
            for name, dtype, _ in ARRAY_SPECS:
                path = _array_paths(paths)[name]
                recorded = meta["arrays"][name]
                if path.stat().st_size != recorded["size_bytes"] or file_checksum(path) != recorded["sha256"]:
                    raise RuntimeError(f"array checksum mismatch: {path}")
                arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(arrays[name].shape) != recorded["shape"] or arrays[name].dtype != np.dtype(dtype):
                    raise RuntimeError(f"array shape/dtype mismatch: {path}")
                total_bytes += path.stat().st_size
            codes = np.asarray(arrays["patient_code"])
            hours = np.asarray(arrays["hours"])
            onset = np.asarray(arrays["onset"])
            pairs = np.empty(len(codes), dtype=[("patient", "<i4"), ("hour", "<i2")])
            pairs["patient"], pairs["hour"] = codes, hours
            if len(np.unique(pairs)) != len(pairs):
                raise RuntimeError(f"duplicate patient/time rows in {hospital}")
            expected_codes = set(expected_hospital.patient_code.astype(int))
            if set(np.unique(codes).astype(int)) != expected_codes:
                raise RuntimeError(f"missing or unexpected patient code in {hospital}")
            fit = fitting_mask(hours, onset, config.tmin, config.hmax)
            if np.any(hours[fit] >= onset[fit]):
                raise RuntimeError(f"post-onset fitting rows detected in {hospital}")
            by_code = pd.DataFrame({"code": codes, "hour": hours}).groupby("code").hour.agg(set)
            horizons = expected_hospital.set_index("patient_code").last_ICULOS
            support_codes = [int(code) for code, horizon in horizons.items() if int(horizon) >= config.tmin]
            if any(not {4, 5, 6}.issubset(by_code.loc[code]) for code in support_codes):
                raise RuntimeError(f"pre-burn/t=6 history support failed in {hospital}")
            for block in meta["blocks"]:
                observed = _block_checksum(arrays, int(block["row_start"]), int(block["row_stop"]))
                if observed != block["content_sha256"]:
                    raise RuntimeError(f"block checksum mismatch: {hospital}/{block['block']}")
                validation_rows.append({
                    "hospital": hospital, "block": block["block"],
                    "patient_count": block["patient_count"], "row_count": block["row_count"],
                    "content_sha256": observed, "checksum_ok": True,
                })
            sample = expected_hospital.iloc[:10]
            for row in sample.itertuples(index=False):
                frame = read_patient(config.root / row.source_file)
                ref = build_patient_features(
                    frame, hospital=hospital, onset=float(row.reconstructed_onset),
                    tmin=config.tmin, hmax=int(row.last_ICULOS),
                    smooth_length=config.smooth_length, include_hospital=True,
                )
                loc = np.flatnonzero(codes == int(row.patient_code))
                observed = np.asarray(arrays["values"])[loc]
                serialized_reference = ref.values.astype(np.float32)
                finite = np.isfinite(serialized_reference) & np.isfinite(observed)
                error = float(np.max(np.abs(serialized_reference[finite] - observed[finite]))) if finite.any() else 0.0
                if not np.array_equal(serialized_reference, observed, equal_nan=True):
                    raise RuntimeError(f"independent feature mismatch for {row.patient_id}")
                reference_error = max(reference_error, error)
            total_rows += int(meta["n_rows"])
            total_blocks += int(meta["actual_block_count"])
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        finally:
            arrays.close()
            del arrays
            gc.collect()
        assert_no_open_files(root)
    report = {
        "cache_id": cache_id, "status": "validated",
        "patient_count": int(len(index)), "row_count": int(total_rows),
        "feature_count": len(names), "block_count": int(total_blocks),
        "feature_name_checksum": sha256("\n".join(names).encode()).hexdigest(),
        "array_disk_bytes": int(total_bytes), "validation_peak_rss_bytes": int(peak_rss),
        "independent_reference_max_error": reference_error,
        "no_open_handle_verification": not open_files_under(root),
        "duplicate_patient_time_rows": 0,
        "post_onset_fitting_rows": 0,
        "preburn_t4_t5_available": True, "t6_history_support": True,
        "all_blocks_readable": True, "all_block_checksums_valid": True,
    }
    return report, pd.DataFrame(validation_rows)


def build_feature_cache(
    config: RealDataConfig, manifest: pd.DataFrame, *, cache_id: str,
    identity: dict[str, Any], n_jobs: int = 4, fresh: bool = False,
    cache_root_override: Path | None = None, write_outputs: bool = True,
) -> dict[str, Any]:
    root = cache_root_override or cache_directory(config, cache_id)
    complete = root / "CACHE_COMPLETE.json"
    if complete.is_file() and not fresh:
        before = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
        report, validation = _validate_cache(config, manifest, root, cache_id=cache_id, require_complete=True)
        after = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
        if before != after:
            raise RuntimeError("resume modified an immutable complete cache")
        report["resumed"] = True
        report["resume_mtime_unchanged"] = True
        if write_outputs:
            _write_feature_reports(config, root, identity, report, validation)
        return report
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete or existing cache directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("arrays", "blocks", "metadata"):
        (root / name).mkdir(exist_ok=True)
    started = time.perf_counter()
    building = {
        "cache_id": cache_id, "status": "building", "identity": identity,
        "started_utc": pd.Timestamp.utcnow().isoformat(), "builder_pid": os.getpid(),
        "complete_marker_exists": False,
    }
    _write_small_json(building, root / "CACHE_BUILDING.json")
    names = raw_feature_names(True)
    patient_index = manifest.loc[
        manifest.primary_cohort,
        ["patient_id", "hospital_set", "source_file", "reconstructed_onset", "last_ICULOS", "any_sepsis_label"],
    ].copy().sort_values(["hospital_set", "patient_id"], kind="mergesort").reset_index(drop=True)
    patient_index["patient_code"] = np.arange(len(patient_index), dtype=np.int32)
    atomic_write_csv(patient_index, root / "metadata" / "patient_index.csv.gz")
    blocks = {hospital: patient_index[patient_index.hospital_set == hospital].copy() for hospital in ("A", "B")}
    parent = psutil.Process()
    child_pids_before = {child.pid for child in parent.children(recursive=True)}
    if n_jobs > 1:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(2, n_jobs), mp_context=context) as executor:
            futures = [executor.submit(_build_hospital_cache, config, blocks[h], h, root, names) for h in ("A", "B")]
            hospital_meta = [future.result() for future in futures]
    else:
        hospital_meta = [_build_hospital_cache(config, blocks[h], h, root, names) for h in ("A", "B")]
    gc.collect()
    child_pids_after = {child.pid for child in parent.children(recursive=True) if child.is_running()}
    leaked_children = sorted(child_pids_after - child_pids_before)
    if leaked_children:
        raise RuntimeError(f"feature-cache workers did not exit: {leaked_children}")
    assert_no_open_files(root)
    report, validation = _validate_cache(config, patient_index.assign(primary_cohort=True), root, cache_id=cache_id, require_complete=False)
    marker = {
        "cache_id": cache_id, "status": "complete", "identity": identity,
        "expected_block_count": int(sum(item["expected_block_count"] for item in hospital_meta)),
        "actual_block_count": int(sum(item["actual_block_count"] for item in hospital_meta)),
        "row_count": int(sum(item["n_rows"] for item in hospital_meta)),
        "patient_count": int(sum(item["n_patients"] for item in hospital_meta)),
        "feature_count": len(names),
        "feature_name_checksum": sha256("\n".join(names).encode()).hexdigest(),
        "arrays": {item["hospital"]: item["arrays"] for item in hospital_meta},
        "split_checksum": identity["split_checksum"],
        "source_checksum": identity["source_checksum"],
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "no_open_handle_verification": True,
        "active_child_workers": 0,
        "validation_results": report,
        "completion_marker_written_last": True,
    }
    replace_audit = commit_completion_marker(root, marker, validated=True)
    report.update({
        "status": "complete", "resumed": False,
        "resume_mtime_unchanged": None,
        "build_wall_seconds": time.perf_counter() - started,
        "build_peak_rss_bytes": max(int(item["peak_rss_bytes"]) for item in hospital_meta),
        "replace_retry_audit": replace_audit,
        "completion_marker_written_last": True,
        "active_child_workers": 0,
    })
    if write_outputs:
        _write_feature_reports(config, root, identity, report, validation)
    return report


def _write_feature_reports(
    config: RealDataConfig, root: Path, identity: dict[str, Any],
    report: dict[str, Any], validation: pd.DataFrame,
) -> None:
    out = config.outputs / "feature_cache"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cache_directory": root.relative_to(config.root).as_posix(),
        "identity": identity, "report": report,
        "complete_marker_sha256": file_checksum(root / "CACHE_COMPLETE.json"),
    }
    _write_small_json(manifest, out / "full_feature_cache_manifest.json")
    atomic_write_csv(validation, out / "full_feature_cache_validation.csv")
    lines = [
        "# Full Feature Cache Report", "", "Gate status: **PASS**", "",
        f"- cache_id: `{report['cache_id']}`",
        f"- patients: {report['patient_count']:,}",
        f"- rows: {report['row_count']:,}",
        f"- features: {report['feature_count']:,}",
        f"- blocks: {report['block_count']:,}",
        f"- array disk bytes: {report['array_disk_bytes']:,}",
        f"- no open handles: {report['no_open_handle_verification']}",
        f"- all block checksums valid: {report['all_block_checksums_valid']}",
        f"- independent reference max error: {report['independent_reference_max_error']}",
        f"- resumed: {report.get('resumed', False)}",
        f"- resume mtime unchanged: {report.get('resume_mtime_unchanged')}",
        f"- build wall seconds: {report.get('build_wall_seconds')}",
        f"- build peak RSS bytes: {report.get('build_peak_rss_bytes')}",
    ]
    (out / "FULL_FEATURE_CACHE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_feature_cache(
    config: RealDataConfig, manifest: pd.DataFrame, *, cache_id: str | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    root = cache_directory(config, cache_id)
    marker = _load_marker(root)
    resolved_id = str(marker["cache_id"])
    before = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    report, validation = _validate_cache(config, manifest, root, cache_id=resolved_id, require_complete=True)
    with _open_cache_at(config, "A", root), _open_cache_at(config, "B", root):
        pass
    assert_no_open_files(root)
    after = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    report.update({"status": "complete", "resumed": True, "resume_mtime_unchanged": before == after})
    if before != after:
        raise RuntimeError("read-only validation modified immutable cache mtimes")
    if write_outputs:
        identity = marker["identity"]
        _write_feature_reports(config, root, identity, report, validation)
    return report


def patient_codes(config: RealDataConfig) -> pd.DataFrame:
    root = cache_directory(config)
    _load_marker(root)
    return pd.read_csv(root / "metadata" / "patient_index.csv.gz", float_precision="round_trip")
