import numpy as np
from pafar_sim.realdata.feature_builder import fitting_mask, utility_mask
def test_utility_not_truncated_at_onset_but_main_is():
    h=np.array([6,7,8,9]); onset=np.array([8,8,8,8]); horizon=np.array([10,10,10,10]); assert np.array_equal(fitting_mask(h,onset,6,168),[1,1,0,0]); assert utility_mask(h,horizon,6,168).all()

