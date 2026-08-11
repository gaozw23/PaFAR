from dataclasses import replace
import numpy as np, pandas as pd
from pafar_sim.realdata.cohort import patient_metadata
from pafar_sim.realdata.schema import EXPECTED_COLUMNS, load_config

def frame(labels, hours):
    x=pd.DataFrame(np.nan,index=range(len(hours)),columns=EXPECTED_COLUMNS); x["Age"]=60;x["Gender"]=1;x["HospAdmTime"]=-2;x["ICULOS"]=hours;x["SepsisLabel"]=labels; return x
def metadata(tmp_path, labels, hours):
    p=tmp_path/"p.psv"; frame(labels,hours).to_csv(p,sep="|",index=False); c=replace(load_config("configs/realdata_primary.yaml"),root=tmp_path); return patient_metadata(p,"A","x",c)
def test_onset_uses_actual_iculos_not_row_index(tmp_path):
    m=metadata(tmp_path,[0,0,1,1],[1,2,5,6]); assert m["reconstructed_onset"]==11 and m["first_positive_row"]==2
def test_first_row_positive_is_left_truncated(tmp_path):
    m=metadata(tmp_path,[1,1],[5,6]); assert m["left_truncated"] and not m["primary_cohort"]
def test_onset_outside_record_excluded(tmp_path):
    m=metadata(tmp_path,[0,1],[5,6]); assert not m["onset_within_record"] and not m["primary_cohort"]

