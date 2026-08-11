import numpy as np, pandas as pd
from pafar_sim.realdata.feature_builder import build_patient_features
from pafar_sim.realdata.schema import EXPECTED_COLUMNS
def make():
    x=pd.DataFrame(np.nan,index=range(12),columns=EXPECTED_COLUMNS); x["Age"]=50;x["Gender"]=0;x["HospAdmTime"]=-1;x["ICULOS"]=np.arange(1,13);x["SepsisLabel"]=0;x["HR"]=[1,2,np.nan,4,5,6,7,8,9,10,11,12];return x
def test_future_perturbation_does_not_change_past():
    x=make(); a=build_patient_features(x,hospital="A",onset=np.inf,hmax=12); x.loc[x.ICULOS>8,"HR"]+=999; b=build_patient_features(x,hospital="A",onset=np.inf,hmax=12); m=a.hours<=8; assert np.array_equal(a.values[m],b.values[m],equal_nan=True)
def test_window_is_open_left_closed_right():
    x=make(); f=build_patient_features(x,hospital="A",onset=np.inf,hmax=12); row=np.where(f.hours==12)[0][0]; col=f.names.index("HR_count_6h"); assert f.values[row,col]==6

