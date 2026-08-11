from __future__ import annotations
import argparse, json, sys
from pafar_sim.realdata.orchestrator import run_pipeline, STAGES
from pafar_sim.realdata.schema import load_config

CONFIRM={
    "windows-memmap-stress":"FIX_AND_TEST_MEMMAP_V2",
    "feature-cache":"BUILD_FRESH_REALDATA_CACHE_V2",
    "validate-feature-cache":"VALIDATE_REALDATA_CACHE_V2",
    "analysis-after-cache":"RUN_PAFAR_REALDATA_AFTER_CACHE_V2_1",
}
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--stage",default="all",choices=("all",)+STAGES); p.add_argument("--resume",action="store_true"); p.add_argument("--n-jobs",type=int,default=4); p.add_argument("--config",default="configs/realdata_primary.yaml"); p.add_argument("--confirm"); p.add_argument("--check-only",action="store_true"); a=p.parse_args()
    expected=CONFIRM.get(a.stage,"RUN_PAFAR_REALDATA_PRIMARY")
    if not a.check_only and a.confirm!=expected: p.error(f"stage {a.stage} requires --confirm {expected}")
    result=run_pipeline(load_config(a.config),stage=a.stage,resume=a.resume,n_jobs=a.n_jobs,check_only=a.check_only)
    summary={"status":"complete","stage":a.stage,"keys":list(result)} if not a.check_only else result
    print(json.dumps(summary,indent=2,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
