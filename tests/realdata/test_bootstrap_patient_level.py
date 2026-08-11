import numpy as np
from pafar_sim.realdata.bootstrap import stratified_indices
def test_bootstrap_resamples_complete_patient_indices():
    strata=np.array(["A0","A0","A1","A1","B0"]); idx=stratified_indices(strata,np.random.Generator(np.random.PCG64DXSM(1))); assert len(idx)==len(strata); assert set(np.unique(idx)).issubset(set(range(len(strata))))

