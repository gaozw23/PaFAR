from __future__ import annotations
import argparse
from pafar_sim.realdata.manuscript import build_manuscript_outputs
from pafar_sim.realdata.schema import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/realdata_primary.yaml"); a=p.parse_args(); print(build_manuscript_outputs(load_config(a.config)))
if __name__=="__main__": main()

