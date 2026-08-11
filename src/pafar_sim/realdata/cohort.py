"""Patient-level QC, onset reconstruction, and primary landmark cohort."""
from __future__ import annotations

from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pafar_sim.io_utils import atomic_write_csv
from .raw_io import read_patient
from .schema import BASELINE_COLUMNS, LONGITUDINAL_COLUMNS, RealDataConfig


def _unique_or_nan(values: np.ndarray) -> tuple[float, bool]:
    finite = values[np.isfinite(values)]
    unique = np.unique(finite)
    return (float(unique[0]) if unique.size else np.nan), bool(unique.size <= 1)


def patient_metadata(path: Path, hospital: str, file_hash: str, config: RealDataConfig) -> dict[str, object]:
    frame = read_patient(path)
    iculos = frame.ICULOS.to_numpy(float)
    labels = frame.SepsisLabel.to_numpy(float)
    label_valid = bool(np.isin(labels, [0.0, 1.0]).all())
    positive = np.flatnonzero(labels == 1)
    any_event = bool(positive.size)
    first_positive_row = int(positive[0]) if any_event else -1
    first_positive_iculos = float(iculos[first_positive_row]) if any_event else np.nan
    onset = first_positive_iculos + 6 if any_event else np.inf
    first_row_positive = bool(any_event and first_positive_row == 0)
    label_monotone = bool(label_valid and not np.any(np.diff(labels) < 0))
    baseline: dict[str, float] = {}
    baseline_ok = True
    for column in BASELINE_COLUMNS:
        value, ok = _unique_or_nan(frame[column].to_numpy(float))
        baseline[column], baseline_ok = value, baseline_ok and ok
    horizon = float(iculos[-1]) if iculos.size else np.nan
    onset_within = bool(not any_event or (np.isfinite(onset) and onset <= horizon))
    left_truncated = first_row_positive
    iculos_integer = bool(np.isfinite(iculos).all() and np.equal(iculos, np.floor(iculos)).all())
    increasing = bool(iculos.size > 0 and np.all(np.diff(iculos) > 0))
    duplicate = bool(np.unique(iculos).size != iculos.size)
    has_gap = bool(iculos.size > 1 and np.any(np.diff(iculos) != 1))
    onset_reconstructable = bool(label_valid and label_monotone and not first_row_positive and onset_within)
    primary = bool(
        np.isfinite(horizon) and horizon >= config.tmin and baseline_ok and label_valid and label_monotone
        and iculos_integer and increasing and not duplicate
        and (not any_event or (onset_reconstructable and onset > config.tmin))
    )
    reasons: list[str] = []
    if not baseline_ok: reasons.append("baseline_inconsistent")
    if not label_valid: reasons.append("invalid_label")
    if not label_monotone: reasons.append("nonmonotone_label")
    if not iculos_integer: reasons.append("noninteger_ICULOS")
    if not increasing: reasons.append("ICULOS_not_strictly_increasing")
    if duplicate: reasons.append("duplicate_hour")
    if not np.isfinite(horizon) or horizon < config.tmin: reasons.append("below_landmark")
    if any_event and first_row_positive: reasons.append("left_truncated")
    if any_event and not onset_within: reasons.append("onset_outside_record")
    if any_event and not onset_reconstructable and not first_row_positive and onset_within: reasons.append("onset_unreconstructable")
    stop = min(horizon, config.hmax) if np.isfinite(horizon) else -np.inf
    times = iculos
    eligible = (times >= config.tmin) & (times <= stop) & (times < onset)
    warning0 = eligible & any_event & (times >= onset - config.wmax) & (times <= onset)
    warning3 = eligible & any_event & (times >= onset - config.wmax) & (times <= onset - 3)
    row: dict[str, object] = {
        "patient_id": path.stem, "hospital_set": hospital,
        "source_file": path.relative_to(config.root).as_posix(), "file_sha256": file_hash,
        "n_rows": len(frame), "first_ICULOS": iculos[0] if iculos.size else np.nan,
        "last_ICULOS": horizon, "min_ICULOS": np.min(iculos) if iculos.size else np.nan,
        "max_ICULOS": np.max(iculos) if iculos.size else np.nan,
        "ICULOS_integer": iculos_integer, "ICULOS_strictly_increasing": increasing,
        "ICULOS_has_gap": has_gap, "duplicate_hour": duplicate,
        "age": baseline["Age"], "gender": baseline["Gender"], "unit1": baseline["Unit1"],
        "unit2": baseline["Unit2"], "hosp_adm_time": baseline["HospAdmTime"],
        "baseline_consistent": baseline_ok, "any_sepsis_label": any_event,
        "first_positive_row": first_positive_row, "first_positive_ICULOS": first_positive_iculos,
        "label_monotone": label_monotone, "label_valid": label_valid,
        "reconstructed_onset": onset, "onset_within_record": onset_within,
        "first_row_positive": first_row_positive, "left_truncated": left_truncated,
        "primary_cohort": primary, "exclusion_reason": ";".join(reasons),
        "eligible_hour_count": int(eligible.sum()), "sens3_evaluable": bool(warning3.any()),
        "sens0_evaluable": bool(warning0.any()), "hmax_truncated": bool(horizon > config.hmax),
    }
    for variable in LONGITUDINAL_COLUMNS:
        count = int(np.isfinite(frame[variable].to_numpy(float)).sum())
        row[f"{variable}_nobs"] = count
        row[f"{variable}_missing_fraction"] = float(1 - count / len(frame)) if len(frame) else np.nan
    return row


def build_patient_manifest(config: RealDataConfig, raw_manifest: pd.DataFrame, *, n_jobs: int = 4) -> pd.DataFrame:
    started = time.perf_counter()
    items=list(raw_manifest.itertuples(index=False))
    def parse(row): return patient_metadata(config.root / row.source_file, row.hospital_set, row.sha256, config)
    with ThreadPoolExecutor(max_workers=n_jobs,thread_name_prefix="cohort-audit") as pool:
        rows=list(pool.map(parse,items,chunksize=16))
    manifest = pd.DataFrame(rows).sort_values(["hospital_set", "patient_id"], kind="mergesort").reset_index(drop=True)
    fatal = manifest.loc[
        (~manifest.baseline_consistent) | (~manifest.label_valid) | (~manifest.label_monotone)
        | (~manifest.ICULOS_integer) | (~manifest.ICULOS_strictly_increasing) | manifest.duplicate_hour
    ]
    if not fatal.empty:
        raise RuntimeError(f"Cohort gate failed for {len(fatal)} patients; see patient manifest")
    processed = config.data_root / "processed"
    qc = config.outputs / "qc"
    atomic_write_csv(manifest, processed / "patient_manifest.csv.gz")
    primary = manifest.loc[manifest.primary_cohort, ["patient_id", "hospital_set", "any_sepsis_label"]]
    atomic_write_csv(primary, processed / "primary_cohort_ids.csv")
    atomic_write_csv(manifest.loc[~manifest.primary_cohort, ["patient_id", "hospital_set", "exclusion_reason"]], processed / "exclusion_log.csv")
    counts = manifest.groupby(["hospital_set", "primary_cohort", "any_sepsis_label"], dropna=False).size().rename("n").reset_index()
    atomic_write_csv(counts, qc / "cohort_counts.csv")
    onset = manifest.loc[manifest.any_sepsis_label, ["hospital_set", "first_positive_ICULOS", "reconstructed_onset", "onset_within_record", "left_truncated"]]
    atomic_write_csv(onset, qc / "event_onset_summary.csv")
    stay = manifest.groupby("hospital_set").agg(n=("patient_id", "size"), median_h=("last_ICULOS", "median"), q1_h=("last_ICULOS", lambda x: x.quantile(.25)), q3_h=("last_ICULOS", lambda x: x.quantile(.75)), max_h=("last_ICULOS", "max")).reset_index()
    atomic_write_csv(stay, qc / "stay_length_summary.csv")
    miss_rows = []
    for variable in LONGITUDINAL_COLUMNS:
        miss_rows.append({"variable": variable, "patient_mean_missing_fraction": manifest[f"{variable}_missing_fraction"].mean(), "total_observations": int(manifest[f"{variable}_nobs"].sum())})
    atomic_write_csv(pd.DataFrame(miss_rows), qc / "variable_missingness.csv")
    exclusions = manifest.loc[~manifest.primary_cohort, "exclusion_reason"].value_counts(dropna=False)
    lines = ["# Cohort QC Report", "", "Gate status: **PASS**", "", f"Primary cohort: {len(primary):,}", f"Excluded: {len(manifest)-len(primary):,}", "", "## Exclusion combinations", ""]
    lines += [f"- `{key or 'none'}`: {value}" for key, value in exclusions.items()]
    lines += ["", f"Elapsed seconds: {time.perf_counter()-started:.3f}"]
    (qc / "COHORT_QC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
