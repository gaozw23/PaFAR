import numpy as np
from pafar_sim.score import eligible_mask
def test_tmin_hmax_and_strict_onset():
    mask=eligible_mask(np.array([200]),np.array([10.]),6,168)[0]
    assert mask[5] and mask[8] and not mask[9] and not mask[168-1+1:].any()
def test_warning_sets():
    times=np.arange(1,20); onset=12; eligible=(times>=6)&(times<onset); w0=eligible&(times>=onset-6)&(times<=onset); w3=eligible&(times>=onset-6)&(times<=onset-3)
    assert np.array_equal(times[w0],[6,7,8,9,10,11]); assert np.array_equal(times[w3],[6,7,8,9])

