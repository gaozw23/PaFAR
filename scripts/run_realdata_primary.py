from __future__ import annotations
import argparse
from pafar_sim.realdata.orchestrator import run_pipeline
from pafar_sim.realdata.schema import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/realdata_primary.yaml"); p.add_argument("--resume",action="store_true"); p.add_argument("--n-jobs",type=int,default=4); a=p.parse_args()
    run_pipeline(load_config(a.config),stage="transfer",resume=a.resume,n_jobs=a.n_jobs)
if __name__=="__main__": main()

