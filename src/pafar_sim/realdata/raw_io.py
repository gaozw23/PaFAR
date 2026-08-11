"""Read-only enumeration, hashing, and parsing of raw PhysioNet PSV files."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd

from pafar_sim.io_utils import atomic_write_csv, atomic_write_json, file_checksum
from .schema import EXPECTED_COLUMNS, RealDataConfig


@dataclass(frozen=True)
class RawAuditResult:
    manifest: pd.DataFrame
    counts: dict[str, int]
    passed: bool
    elapsed_seconds: float


def enumerate_psv(config: RealDataConfig) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for hospital, directory in (("A", config.raw_a), ("B", config.raw_b)):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing raw directory: {directory}")
        out.extend((hospital, p) for p in sorted(directory.glob("*.psv"), key=lambda x: x.name))
    return out


def parse_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        line = stream.readline().rstrip("\r\n")
    return tuple(line.split("|"))


def read_patient(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        Path(path), sep="|", dtype=np.float64, na_values=["NaN", "nan", "", " "],
        keep_default_na=True, engine="c",
    )
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Schema mismatch in {path}")
    return frame


def audit_raw_files(config: RealDataConfig, *, write_outputs: bool = True, n_jobs: int = 4) -> RawAuditResult:
    started = time.perf_counter()
    files = enumerate_psv(config)
    def inspect(item: tuple[str,Path]) -> dict[str,object]:
        hospital,path=item
        stat = path.stat()
        patient_id = path.stem
        corrupt = False
        error = ""
        header: tuple[str, ...] = ()
        try:
            header = parse_header(path)
            if stat.st_size == 0:
                corrupt, error = True, "empty"
            elif header != EXPECTED_COLUMNS:
                error = "schema_mismatch"
            else:
                # A final-byte read catches inaccessible/truncated files without parsing twice.
                with path.open("rb") as stream:
                    stream.seek(-1, 2)
                    stream.read(1)
        except Exception as exc:  # audit records the exact file instead of skipping it
            corrupt, error = True, f"{type(exc).__name__}: {exc}"
        return {
            "patient_id": patient_id, "hospital_set": hospital,
            "source_file": path.relative_to(config.root).as_posix(),
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": file_checksum(path), "header": "|".join(header),
            "schema_ok": header == EXPECTED_COLUMNS, "empty": stat.st_size == 0,
            "corrupt": corrupt, "read_error": error,
        }
    with ThreadPoolExecutor(max_workers=n_jobs,thread_name_prefix="raw-audit") as pool:
        rows=list(pool.map(inspect,files,chunksize=32))
    names=[str(row["patient_id"]) for row in rows]
    duplicate_names=len(names)-len(set(names))
    manifest = pd.DataFrame(rows).sort_values(["hospital_set", "patient_id"], kind="mergesort").reset_index(drop=True)
    counts = {
        "A_psv": int((manifest.hospital_set == "A").sum()),
        "B_psv": int((manifest.hospital_set == "B").sum()),
        "total_psv": int(len(manifest)),
        "empty": int(manifest["empty"].sum()),
        "corrupt": int(manifest.corrupt.sum()),
        "schema_mismatch": int((~manifest.schema_ok).sum()),
        "duplicate_patient_ids": int(manifest.patient_id.duplicated().sum()),
        "duplicate_names_seen": int(duplicate_names),
        "A_non_psv": len([p for p in config.raw_a.iterdir() if p.is_file() and p.suffix.lower() != ".psv"]),
        "B_non_psv": len([p for p in config.raw_b.iterdir() if p.is_file() and p.suffix.lower() != ".psv"]),
    }
    passed = (
        counts["A_psv"] == config.expected_a and counts["B_psv"] == config.expected_b
        and counts["empty"] == counts["corrupt"] == counts["schema_mismatch"]
        == counts["duplicate_patient_ids"] == 0
    )
    elapsed = time.perf_counter() - started
    if write_outputs:
        manifest_dir = config.data_root / "manifests"
        qc_dir = config.outputs / "qc"
        atomic_write_csv(manifest, manifest_dir / "raw_file_manifest.csv.gz")
        summary = {
            "passed": passed, "counts": counts, "elapsed_seconds": elapsed,
            "columns": list(EXPECTED_COLUMNS),
            "aggregate_manifest_sha256": sha256(
                "\n".join(f"{r.patient_id}|{r.sha256}|{r.size_bytes}|{r.mtime_ns}" for r in manifest.itertuples()).encode()
            ).hexdigest(),
        }
        atomic_write_json(summary, manifest_dir / "raw_file_manifest.json")
        atomic_write_csv(pd.DataFrame([counts]), qc_dir / "raw_file_counts.csv")
        atomic_write_csv(manifest[["patient_id", "hospital_set", "schema_ok", "corrupt", "read_error"]], qc_dir / "schema_audit.csv")
        lines = ["# Raw Data Audit Report", "", f"Gate status: **{'PASS' if passed else 'FAIL'}**", ""]
        lines += [f"- {key}: {value}" for key, value in counts.items()]
        lines += ["", f"Elapsed seconds: {elapsed:.3f}"]
        target = qc_dir / "RAW_DATA_AUDIT_REPORT.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RawAuditResult(manifest, counts, passed, elapsed)


def verify_raw_unchanged(config: RealDataConfig, manifest: pd.DataFrame) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        path = config.root / row.source_file
        if not path.is_file():
            failures.append(f"missing:{row.source_file}")
            continue
        stat = path.stat()
        if stat.st_size != row.size_bytes or stat.st_mtime_ns != row.mtime_ns or file_checksum(path) != row.sha256:
            failures.append(f"changed:{row.source_file}")
    return not failures, failures
