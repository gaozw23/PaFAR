import numpy as np
from pafar_sim.alerting import first_alert
from pafar_sim.calibration import marginal_index, marginal_threshold, hc_index
def test_order_indices_recomputed():
    k,a=marginal_index(100,.1); assert k==91 and np.isclose(a,10/101); assert hc_index(100,.1,.05)>=k
def test_strict_threshold_ties():
    score=np.array([[1.,2.]]); eligible=np.ones_like(score,dtype=bool); assert np.isinf(first_alert(score,eligible,2.)[0])
def test_infinite_threshold_when_augmented_rank_selected():
    assert np.isinf(marginal_threshold(np.arange(5.),.1).threshold)

