"""Strict stage-gated orchestration and immutable analysis lock."""
from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import pandas as pd

from pafar_sim.io_utils import atomic_write_csv, atomic_write_json, file_checksum
from .cohort import build_patient_manifest
from .feature_cache import build_feature_cache, validate_feature_cache
from .feature_smoke import run_feature_smoke
from .internal_analysis import run_internal_primary
from .raw_io import audit_raw_files, verify_raw_unchanged
from .recovery import (
    create_lock_v2, create_lock_v2_1, load_fixed_internal_split, load_frozen_patient_manifest,
    verify_lock_v2, verify_lock_v2_1, verify_raw_from_frozen_manifest,
)
from .robustness import run_robustness
from .schema import RealDataConfig
from .splitting import seed_registry
from .transfer_analysis import run_cross_hospital
from .utility import OFFICIAL_SHA256
from .windows_stress import run_windows_memmap_stress


STAGES=(
    "raw","cohort","features","internal","transfer","bootstrap","robustness","manuscript",
    "windows-memmap-stress","feature-cache","validate-feature-cache","analysis-after-cache",
)


def tree_manifest(paths: list[Path], root: Path) -> pd.DataFrame:
    rows=[]
    for base in paths:
        if not base.exists():
            rows.append({"path":base.relative_to(root).as_posix(),"exists":False,"size_bytes":0,"mtime_ns":0,"sha256":""}); continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            stat=path.stat(); rows.append({"path":path.relative_to(root).as_posix(),"exists":True,"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":file_checksum(path)})
    return pd.DataFrame(rows)


def aggregate_manifest(frame: pd.DataFrame) -> str:
    text="\n".join(f"{r.path}|{r.exists}|{r.size_bytes}|{r.mtime_ns}|{r.sha256}" for r in frame.sort_values("path").itertuples())
    return sha256(text.encode()).hexdigest()


def source_checksum(config: RealDataConfig) -> str:
    files=sorted((config.root/"src"/"pafar_sim"/"realdata").glob("*.py"))+sorted((config.root/"scripts").glob("*realdata*.py"))+sorted((config.root/"tests"/"realdata").glob("*.py"))
    digest=sha256()
    for path in files: digest.update(path.relative_to(config.root).as_posix().encode()); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def effective_config_checksum(config: RealDataConfig) -> str:
    robustness=config.root/"configs"/"realdata_robustness.yaml"
    return sha256((config.checksum+"\0").encode()+robustness.read_bytes()).hexdigest()


def protected_baseline(config: RealDataConfig) -> dict[str,object]:
    paths=[config.root/"outputs"/"production",config.root/"outputs"/"manuscript_primary_v2",config.root/"outputs"/"manuscript_primary_final"]
    frame=tree_manifest(paths,config.root); target=config.outputs/"manifests"/"simulation_protected_baseline.csv.gz"; atomic_write_csv(frame,target)
    result={"aggregate_sha256":aggregate_manifest(frame),"file_count":int(frame.exists.sum()),"total_bytes":int(frame.size_bytes.sum())}
    atomic_write_json(result,config.outputs/"manifests"/"simulation_protected_baseline.json"); return result


def verify_protected(config: RealDataConfig) -> tuple[bool,dict[str,object]]:
    baseline=pd.read_csv(config.outputs/"manifests"/"simulation_protected_baseline.csv.gz",dtype={"sha256":"string"}).fillna({"sha256":""})
    paths=[config.root/"outputs"/"production",config.root/"outputs"/"manuscript_primary_v2",config.root/"outputs"/"manuscript_primary_final"]
    current=tree_manifest(paths,config.root); ok=aggregate_manifest(baseline)==aggregate_manifest(current)
    result={"unchanged":ok,"before_aggregate_sha256":aggregate_manifest(baseline),"after_aggregate_sha256":aggregate_manifest(current),"before_files":len(baseline),"after_files":len(current)}
    atomic_write_json(result,config.outputs/"manifests"/"simulation_protected_after.json"); return ok,result


def create_lock(config: RealDataConfig, raw_summary: dict[str,object]) -> dict[str,object]:
    packages={}
    for name in ("numpy","pandas","scipy","scikit-learn","xgboost","matplotlib","joblib","pyyaml","psutil"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]="not-installed"
    lock={"timestamp_utc":pd.Timestamp.utcnow().isoformat(),"master_seed":config.master_seed,"split_seeds":seed_registry(config.master_seed),
          "raw_manifest_checksum":raw_summary["aggregate_manifest_sha256"],"raw_file_counts":raw_summary["counts"],
          "source_code_checksum":source_checksum(config),"effective_config_checksum":effective_config_checksum(config),
          "PaFAR_tex_checksum":file_checksum(config.root/"literature"/"PaFAR.tex"),"package_versions":packages,
          "python":sys.version,"platform":platform.platform(),"utility_scorer_checksum":OFFICIAL_SHA256,
          "settings":{"Hmax":config.hmax,"tmin":config.tmin,"L":config.smooth_length,"alpha":config.alphas,"delta":config.delta}}
    atomic_write_json(lock,config.outputs/"REALDATA_LOCK.json"); return lock


def check_lock(config: RealDataConfig) -> dict[str,object]:
    path=config.outputs/"REALDATA_LOCK.json"
    if not path.is_file(): raise RuntimeError("REALDATA_LOCK.json is missing")
    lock=json.loads(path.read_text(encoding="utf-8"))
    if lock["source_code_checksum"]!=source_checksum(config) or lock["effective_config_checksum"]!=effective_config_checksum(config):
        raise RuntimeError("code/config changed after real-data lock; old checkpoints cannot be reused")
    return lock


def run_pipeline(config: RealDataConfig, *, stage: str="all", resume: bool=True, n_jobs: int=4, check_only: bool=False) -> dict[str,object]:
    if stage == "validate-feature-cache":
        if n_jobs != 1: raise ValueError("feature-cache validation requires n_jobs=1")
    elif n_jobs != 4:
        raise ValueError("primary analysis requires n_jobs=4")
    wanted=list(STAGES) if stage=="all" else [stage]
    if any(item not in STAGES for item in wanted): raise ValueError(f"unknown stage {stage}")
    started=time.perf_counter(); state:dict[str,object]={}

    if stage in ("windows-memmap-stress", "feature-cache", "validate-feature-cache", "analysis-after-cache"):
        manifest=load_frozen_patient_manifest(config)
        split=load_fixed_internal_split(config,manifest)
        state["frozen_inputs"]={"patients":int(manifest.primary_cohort.sum()),"internal_split_rows":len(split)}
        if stage=="windows-memmap-stress":
            state["windows_memmap_stress"]=run_windows_memmap_stress(config,manifest,n_jobs=n_jobs)
            return state
        if stage=="feature-cache":
            lock=create_lock_v2(config,manifest,source_checksum=source_checksum(config),effective_config_checksum=effective_config_checksum(config))
            identity={
                "raw_manifest_checksum":lock["raw_manifest_checksum"],
                "cohort_checksum":lock["cohort_manifest_checksum"],
                "split_checksum":lock["internal_split_checksum"],
                "feature_checksum":lock["feature_definition_checksum"],
                "source_checksum":lock["source_checksum"],
                "effective_config_checksum":lock["effective_config_checksum"],
                "master_seed":lock["master_seed"],
            }
            state["lock_v2"]=lock
            state["feature_cache"]=build_feature_cache(config,manifest,cache_id=lock["cache_id"],identity=identity,n_jobs=n_jobs,fresh=False,write_outputs=True)
            return state
        if stage == "analysis-after-cache" and not (config.outputs / "REALDATA_LOCK_V2_1.json").is_file():
            create_lock_v2_1(config, manifest, source_checksum=source_checksum(config), effective_config_checksum=effective_config_checksum(config))
        lock = (
            verify_lock_v2_1(config, source_checksum=source_checksum(config), effective_config_checksum=effective_config_checksum(config))
            if stage == "analysis-after-cache"
            else verify_lock_v2(config, source_checksum=source_checksum(config), effective_config_checksum=effective_config_checksum(config))
        )
        state["lock_v2"]=lock
        cache_id = lock.get("feature_cache_id", lock.get("cache_id"))
        state["feature_cache"]=validate_feature_cache(config,manifest,cache_id=cache_id,write_outputs=True)
        if stage=="validate-feature-cache": return state
        internal=run_internal_primary(config,manifest,resume=resume); state["internal"]=internal
        transfer=run_cross_hospital(config,manifest,resume=resume); state["transfer"]=transfer
        robustness=run_robustness(config,manifest,internal,transfer); state["robustness"]=robustness
        from .manuscript import build_manuscript_outputs
        state["manuscript_outputs"]=build_manuscript_outputs(config)
        unchanged,proof=verify_protected(config)
        if not unchanged: raise RuntimeError("protected simulation outputs changed")
        raw_ok,raw_failures=verify_raw_from_frozen_manifest(config)
        if not raw_ok: raise RuntimeError(f"raw PhysioNet files changed: {raw_failures[:3]}")
        state["integrity"]={"raw_unchanged":raw_ok,"simulation":proof,"elapsed_seconds":time.perf_counter()-started}
        return state

    if not (config.outputs/"manifests"/"simulation_protected_baseline.csv.gz").is_file(): state["simulation_baseline"]=protected_baseline(config)
    raw_result=audit_raw_files(config,write_outputs=True); state["raw"]=raw_result
    if not raw_result.passed: raise RuntimeError("Gate 1 raw data failed")
    if check_only: return {"status":"check_only_passed","raw_counts":raw_result.counts,"config_checksum":config.checksum}
    manifest=build_patient_manifest(config,raw_result.manifest); state["manifest"]=manifest
    smoke=run_feature_smoke(config,manifest); state["feature_smoke"]=smoke
    if not smoke["passed"]: raise RuntimeError("Gate 3 feature smoke failed")
    if stage in ("raw","cohort","features"): return state
    raw_summary=json.loads((config.data_root/"manifests"/"raw_file_manifest.json").read_text(encoding="utf-8"))
    if not (config.outputs/"REALDATA_LOCK.json").is_file(): create_lock(config,raw_summary)
    else: check_lock(config)
    raise RuntimeError("legacy formal path is disabled after WinError 32; use the V2 recovery stages")
    internal=run_internal_primary(config,manifest,resume=resume); state["internal"]=internal
    if stage=="internal": return state
    transfer=run_cross_hospital(config,manifest,resume=resume); state["transfer"]=transfer
    if stage in ("transfer","bootstrap"): return state
    robustness=run_robustness(config,manifest,internal,transfer); state["robustness"]=robustness
    from .manuscript import build_manuscript_outputs
    manuscript_outputs=build_manuscript_outputs(config); state["manuscript_outputs"]=manuscript_outputs
    unchanged,proof=verify_protected(config)
    if not unchanged: raise RuntimeError("protected simulation outputs changed")
    raw_ok,raw_failures=verify_raw_unchanged(config,raw_result.manifest)
    if not raw_ok: raise RuntimeError(f"raw PhysioNet files changed: {raw_failures[:3]}")
    state["integrity"]={"raw_unchanged":raw_ok,"simulation":proof,"elapsed_seconds":time.perf_counter()-started}
    return state
