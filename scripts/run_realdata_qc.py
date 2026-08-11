from __future__ import annotations
import argparse
from pafar_sim.realdata.cohort import build_patient_manifest
from pafar_sim.realdata.raw_io import audit_raw_files
from pafar_sim.realdata.schema import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/realdata_primary.yaml"); a=p.parse_args(); c=load_config(a.config)
    raw=audit_raw_files(c); assert raw.passed,"raw gate failed"; manifest=build_patient_manifest(c,raw.manifest); print(f"raw={len(raw.manifest)} primary={manifest.primary_cohort.sum()}")
if __name__=="__main__": main()

