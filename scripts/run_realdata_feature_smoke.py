from __future__ import annotations
import argparse, pandas as pd
from pafar_sim.realdata.feature_smoke import run_feature_smoke
from pafar_sim.realdata.schema import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/realdata_primary.yaml"); a=p.parse_args(); c=load_config(a.config)
    m=pd.read_csv(c.data_root/"processed"/"patient_manifest.csv.gz",float_precision="round_trip"); print(run_feature_smoke(c,m))
if __name__=="__main__": main()

