from pathlib import Path
import json, pytest
from pafar_sim.realdata.schema import load_config
from pafar_sim.realdata.splitting import seed_registry
from pafar_sim.realdata.utility import OFFICIAL_SHA256
from pafar_sim.io_utils import file_checksum
ROOT=Path(__file__).resolve().parents[2]
def test_config_grid_seed_and_official_scorer():
    c=load_config(ROOT/"configs/realdata_primary.yaml"); assert len(c.xgb_grid)==18 and c.master_seed==20260804; assert seed_registry(c.master_seed)==seed_registry(c.master_seed); assert file_checksum(c.data_root/"manifests/official_evaluation_2019/evaluate_sepsis_score.py")==OFFICIAL_SHA256
def test_tables_have_no_placeholders_when_generated():
    paths=[ROOT/"literature/generated_tables/table6_physionet_internal.tex",ROOT/"literature/generated_tables/table7_cross_hospital.tex"]
    if not all(p.exists() for p in paths): pytest.skip("manuscript outputs not generated yet")
    assert all("resultcell" not in p.read_text(encoding="utf-8") and "NaN" not in p.read_text(encoding="utf-8") for p in paths)

