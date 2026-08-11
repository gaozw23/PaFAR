import numpy as np

from pafar_sim.exp1.dgp import generate_exp1, stationary_ar1


def test_ar_stationarity():
    rng = np.random.default_rng(3)
    process = stationary_ar1(rng, 30_000, 20, .8, .55)
    assert abs(process[:, -1].std() - .55) < .015


def test_exp1_event_onset_and_shapes():
    batch = generate_exp1(np.random.default_rng(4), 20, "S2", True)
    assert batch.risk.shape == (20, 120)
    assert np.all(batch.onset >= 12)
    assert np.all(batch.onset <= batch.horizon)

