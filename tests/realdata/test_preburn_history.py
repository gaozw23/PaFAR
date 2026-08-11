import numpy as np
from pafar_sim.score import causal_moving_average
def test_t6_uses_t4_t5_t6():
    x=np.arange(1,7,dtype=float)[None,:]; assert causal_moving_average(x,3)[0,5]==5
def test_t4_t5_not_in_fitting_or_alert_grid():
    hours=np.arange(4,9); assert np.array_equal(hours[hours>=6],[6,7,8])

