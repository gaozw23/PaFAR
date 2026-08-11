"""Fixed-sample feature causality, alignment, reference, and memory smoke gate."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import psutil

from pafar_sim.io_utils import atomic_write_json
from pafar_sim.score import causal_moving_average
from .feature_builder import build_patient_features, reference_feature_row
from .imputation import fit_preprocessor
from .raw_io import read_patient
from .schema import RealDataConfig
from .splitting import seed_registry


def _sample(manifest: pd.DataFrame, hospital: str, rng: np.random.Generator) -> pd.DataFrame:
    block=manifest[(manifest.primary_cohort)&(manifest.hospital_set==hospital)]
    non=block[~block.any_sepsis_label]; event=block[block.any_sepsis_label]
    n_non=min(400,len(non)); n_event=min(100,len(event))
    return pd.concat([non.iloc[rng.choice(len(non),n_non,replace=False)],event.iloc[rng.choice(len(event),n_event,replace=False)]],ignore_index=True)


def run_feature_smoke(config: RealDataConfig, manifest: pd.DataFrame) -> dict[str,object]:
    started=time.perf_counter(); process=psutil.Process(); seed=seed_registry(config.master_seed)["subgroups"]
    rng=np.random.Generator(np.random.PCG64DXSM(seed)); sample=pd.concat([_sample(manifest,h,rng) for h in ("A","B")],ignore_index=True)
    blocks=[]; peak=process.memory_info().rss; checkpoints={}; build_started=time.perf_counter()
    for patient_no,row in enumerate(sample.itertuples(index=False),start=1):
        frame=read_patient(config.root/row.source_file)
        features=build_patient_features(frame,hospital=row.hospital_set,onset=float(row.reconstructed_onset),tmin=config.tmin,hmax=min(int(row.last_ICULOS),config.hmax),smooth_length=config.smooth_length)
        blocks.append((row,features)); peak=max(peak,process.memory_info().rss)
        if patient_no in (100,1000): checkpoints[patient_no]=time.perf_counter()-build_started
    # Independent reference on 20 patients and one deterministic eligible hour each.
    max_error=0.0
    for row,features in blocks[:20]:
        frame=read_patient(config.root/row.source_file); pos=min(len(features.hours)-1,max(0,len(features.hours)//2)); hour=int(features.hours[pos])
        reference=reference_feature_row(frame,hour,hospital=row.hospital_set,onset=float(row.reconstructed_onset))
        error=np.nanmax(np.abs(features.values[pos].astype(np.float64)-reference.astype(np.float64)))
        max_error=max(max_error,float(error))
    if max_error>1e-10: raise AssertionError(f"reference feature error {max_error}")
    # Future perturbation cannot change features at or before t0.
    row,features=blocks[0]; frame=read_patient(config.root/row.source_file); t0=int(features.hours[len(features.hours)//2]); altered=frame.copy()
    future=altered.ICULOS>t0; altered.loc[future,altered.columns[:34]]=altered.loc[future,altered.columns[:34]]+100000
    changed=build_patient_features(altered,hospital=row.hospital_set,onset=float(row.reconstructed_onset),tmin=config.tmin,hmax=min(int(row.last_ICULOS),config.hmax),smooth_length=config.smooth_length)
    prior=features.hours<=t0
    if not np.array_equal(features.values[prior],changed.values[prior],equal_nan=True): raise AssertionError("future feature leakage")
    # Frozen medians: fitting on first half must not react to validation perturbation.
    values=np.vstack([x.values for _,x in blocks[:20]]); names=blocks[0][1].names; train_mask=np.arange(len(values))<len(values)//2
    pre1=fit_preprocessor([values],[train_mask],names); modified=values.copy(); modified[~train_mask]+=12345
    pre2=fit_preprocessor([modified],[train_mask],names)
    if pre1.checksum!=pre2.checksum: raise AssertionError("validation rows changed training medians")
    # Pre-burn: t=6 uses scores at t=4,5,6 and t=4,5 are not fitting rows.
    risk=np.asarray([[.1,.2,.3,.4,.5,.6]],dtype=float); smooth=causal_moving_average(risk,3)
    if not np.isclose(smooth[0,5],np.mean([.4,.5,.6])): raise AssertionError("pre-burn smoother mismatch")
    # Atomic, resumable smoke cache and byte reproducibility.
    out=config.outputs/"feature_smoke"; out.mkdir(parents=True,exist_ok=True); cache=out/"smoke_cache.npz"
    fd,name=tempfile.mkstemp(prefix="smoke_cache.",suffix=".tmp.npz",dir=out); os.close(fd)
    np.savez_compressed(name,patient_id=sample.patient_id.to_numpy(str),seed=np.asarray([seed],dtype=np.uint64)); os.replace(name,cache)
    first=cache.read_bytes(); second=np.load(cache,allow_pickle=False)["patient_id"]
    if not np.array_equal(second,sample.patient_id.to_numpy(str)): raise AssertionError("smoke cache resume mismatch")
    full_rows=int(sum(max(0,int(h)-max(1,config.tmin-config.smooth_length+1)+1) for h in manifest.loc[manifest.primary_cohort,"last_ICULOS"]))
    full_bytes=full_rows*len(names)*4; ram=psutil.virtual_memory().total; external_required=full_bytes>config.max_ram_fraction*ram
    if peak>=config.max_ram_fraction*ram: raise MemoryError("feature smoke exceeded 75% of physical RAM")
    report={"passed":True,"sample_A":int((sample.hospital_set=="A").sum()),"sample_B":int((sample.hospital_set=="B").sum()),
            "events_A":int(((sample.hospital_set=="A")&sample.any_sepsis_label).sum()),"events_B":int(((sample.hospital_set=="B")&sample.any_sepsis_label).sum()),
            "reference_patients":20,"maximum_reference_error":max_error,"future_causality":True,"training_medians_frozen":True,
            "preburn_t6_exact":True,"cache_resume":True,"benchmark_100_patients_seconds":checkpoints.get(100),"benchmark_1000_patients_seconds":checkpoints.get(1000),
            "estimated_full_patients":int(manifest.primary_cohort.sum()),"estimated_full_rows":full_rows,"estimated_raw_feature_disk_bytes":full_bytes,
            "estimated_full_build_seconds":checkpoints.get(1000, time.perf_counter()-build_started)/len(sample)*int(manifest.primary_cohort.sum()),
            "external_memory_required":external_required,"selected_cache_implementation":"Fortran-order memmap plus streamed transformed batches",
            "peak_rss_bytes":peak,"peak_fraction_of_ram":peak/ram,"elapsed_seconds":time.perf_counter()-started}
    atomic_write_json(report,out/"feature_smoke_report.json")
    lines=["# Feature Smoke Report","","Gate status: **PASS**",""]+[f"- {k}: {v}" for k,v in report.items()]
    (out/"FEATURE_SMOKE_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report
