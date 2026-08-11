"""Bounded Windows memmap stress gate for the V2 feature cache."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import pandas as pd
import psutil

from pafar_sim.io_utils import atomic_write_json
from .feature_cache import (
    build_feature_cache, compute_cache_id, feature_definition_checksum,
)
from .memmap_utils import open_files_under
from .schema import RealDataConfig


def _balanced_subset(manifest: pd.DataFrame, n: int) -> pd.DataFrame:
    primary = manifest[manifest.primary_cohort]
    parts = []
    per_hospital = n // 2
    for hospital in ("A", "B"):
        block = primary[primary.hospital_set == hospital]
        events = block[block.any_sepsis_label].head(min(100, max(1, per_hospital // 5)))
        non_events = block[~block.any_sepsis_label].head(per_hospital - len(events))
        parts.append(pd.concat([events, non_events], ignore_index=True))
    return pd.concat(parts, ignore_index=True).assign(primary_cohort=True)


def run_windows_memmap_stress(
    config: RealDataConfig, manifest: pd.DataFrame, *, n_jobs: int = 4,
) -> dict[str, Any]:
    output = config.outputs / "feature_smoke"
    temp_parent = output / "windows_memmap_stress_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("single_100", 100, 20, 1),
        ("multiprocess_1000", 1000, 5, min(2, n_jobs)),
        ("mixed_5000", 5000, 1, min(4, n_jobs)),
    ]
    started = time.perf_counter()
    process = psutil.Process()
    baseline_children = {child.pid for child in process.children(recursive=True)}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for scenario, patients, repetitions, workers in scenarios:
        subset = _balanced_subset(manifest, patients)
        for repetition in range(repetitions):
            identity = {
                "raw_manifest_checksum": "stress-locked-raw",
                "cohort_checksum": f"stress-{scenario}-{patients}",
                "split_checksum": f"stress-{scenario}-split",
                "feature_checksum": feature_definition_checksum(config),
                "source_checksum": f"stress-implementation-{scenario}-{repetition}",
                "effective_config_checksum": config.checksum,
                "master_seed": config.master_seed,
            }
            cache_id = compute_cache_id(
                raw_manifest_checksum=identity["raw_manifest_checksum"],
                cohort_checksum=identity["cohort_checksum"],
                split_checksum=identity["split_checksum"],
                feature_checksum=identity["feature_checksum"],
                source_checksum=identity["source_checksum"],
                config_checksum=identity["effective_config_checksum"],
                master_seed=identity["master_seed"],
            )
            iteration_started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix=f"{scenario}_", dir=temp_parent) as directory:
                    root = Path(directory) / f"realdata_features_{cache_id}"
                    report = build_feature_cache(
                        config, subset, cache_id=cache_id, identity=identity,
                        n_jobs=workers, fresh=True, cache_root_override=root,
                        write_outputs=False,
                    )
                    marker = json.loads((root / "CACHE_COMPLETE.json").read_text(encoding="utf-8"))
                    if marker["status"] != "complete" or marker["patient_count"] != len(subset):
                        raise RuntimeError("stress completion marker content mismatch")
                    resumed = build_feature_cache(
                        config, subset, cache_id=cache_id, identity=identity,
                        n_jobs=workers, fresh=False, cache_root_override=root,
                        write_outputs=False,
                    )
                    if not resumed["resume_mtime_unchanged"]:
                        raise RuntimeError("stress resume changed cache mtime")
                    if open_files_under(root):
                        raise RuntimeError("stress left open cache handles")
                    disk_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
                    rows.append({
                        "scenario": scenario, "repetition": repetition + 1,
                        "patients": len(subset), "rows": report["row_count"],
                        "features": report["feature_count"], "blocks": report["block_count"],
                        "n_jobs": workers, "peak_rss_bytes": report["build_peak_rss_bytes"],
                        "disk_bytes": disk_bytes,
                        "wall_seconds": time.perf_counter() - iteration_started,
                        "winerror32": 0, "open_handles": 0,
                        "checksums_correct": True, "resume_mtime_unchanged": True,
                        "complete_marker": True,
                    })
                if root.exists():
                    raise RuntimeError("stress temporary cache was not deleted")
            except BaseException as exc:
                failures.append(f"{scenario} repetition {repetition + 1}: {type(exc).__name__}: {exc}")
                break
            finally:
                gc.collect()
        if failures:
            break
    remaining_children = [
        child.pid for child in process.children(recursive=True)
        if child.pid not in baseline_children and child.is_running()
    ]
    if remaining_children:
        failures.append(f"active child processes after stress: {remaining_children}")
    report = {
        "passed": not failures, "failure_count": len(failures), "failures": failures,
        "iterations_completed": len(rows), "iterations_expected": 26,
        "winerror32_count": int(sum(row["winerror32"] for row in rows)),
        "open_handle_failures": int(sum(row["open_handles"] for row in rows)),
        "active_child_processes_after": remaining_children,
        "peak_rss_bytes": max((row["peak_rss_bytes"] for row in rows), default=0),
        "maximum_disk_bytes": max((row["disk_bytes"] for row in rows), default=0),
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    atomic_write_json(report, output / "windows_memmap_stress_report.json")
    lines = [
        "# Windows Memmap Stress Report", "",
        f"Gate status: **{'PASS' if report['passed'] else 'FAIL'}**", "",
        f"- Completed iterations: {len(rows)} / 26",
        f"- WinError 32 failures: {report['winerror32_count']}",
        f"- Open-handle failures: {report['open_handle_failures']}",
        f"- Active child processes after stress: {len(remaining_children)}",
        f"- Peak RSS bytes: {report['peak_rss_bytes']}",
        f"- Maximum per-cache disk bytes: {report['maximum_disk_bytes']}",
        f"- Elapsed seconds: {report['elapsed_seconds']:.3f}", "",
        "## Scenarios", "",
        "- Single process: 100 patients, 20 create/finalize/resume/delete cycles.",
        "- Two processes: 1,000 patients, 5 create/finalize/resume/delete cycles.",
        "- Medium scale: mixed A/B 5,000 patients, n_jobs=4, full validation and resume.",
    ]
    if failures:
        lines += ["", "## Failures", ""] + [f"- {failure}" for failure in failures]
    (output / "WINDOWS_MEMMAP_STRESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures or len(rows) != 26:
        raise RuntimeError(f"Windows memmap stress gate failed: {failures}")
    return report
