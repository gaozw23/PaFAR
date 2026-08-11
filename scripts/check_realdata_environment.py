from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from pafar_sim.io_utils import file_checksum
from pafar_sim.realdata.schema import load_config
from pafar_sim.realdata.utility import OFFICIAL_SHA256


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/realdata_primary.yaml"); args=parser.parse_args()
    config=load_config(args.config); packages={}
    for name in ("numpy","pandas","scipy","scikit-learn","xgboost","matplotlib","joblib","pyyaml","psutil"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    scorer=config.data_root/"manifests"/"official_evaluation_2019"/"evaluate_sepsis_score.py"
    result={"python":sys.version,"raw_A_exists":config.raw_a.is_dir(),"raw_B_exists":config.raw_b.is_dir(),"packages":packages,
            "official_scorer_exists":scorer.is_file(),"official_scorer_checksum":file_checksum(scorer) if scorer.is_file() else None,
            "official_scorer_valid":scorer.is_file() and file_checksum(scorer)==OFFICIAL_SHA256,"config_checksum":config.checksum}
    print(json.dumps(result,indent=2)); return 0 if all(v is not None for v in packages.values()) and result["official_scorer_valid"] and result["raw_A_exists"] and result["raw_B_exists"] else 1


if __name__=="__main__": raise SystemExit(main())

