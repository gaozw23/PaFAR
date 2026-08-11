import pandas as pd
from pafar_sim.realdata.splitting import stratified_patient_split
def patients(): return pd.DataFrame([{"patient_id":f"{h}{e}{i}","hospital_set":h,"any_sepsis_label":bool(e)} for h in "AB" for e in (0,1) for i in range(40)])
def test_disjoint_stratified_reproducible():
    a=stratified_patient_split(patients(),{"train":.5,"validation":.15,"calibration":.15,"test":.2},7); b=stratified_patient_split(patients(),{"train":.5,"validation":.15,"calibration":.15,"test":.2},7)
    assert a.equals(b) and not a.patient_id.duplicated().any(); assert a.groupby(["split","hospital_set","any_sepsis_label"]).size().gt(0).all()

