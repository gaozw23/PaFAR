import numpy as np

from pafar_sim.rng import child_seed, make_rng, seed_record


def test_stable_child_streams_are_order_independent():
    first = {r: make_rng(17, "E1", r).normal(size=5) for r in [2, 0, 1]}
    second = {r: make_rng(17, "E1", r).normal(size=5) for r in [0, 1, 2]}
    for replicate in first:
        np.testing.assert_array_equal(first[replicate], second[replicate])
    assert child_seed(17, "a") != child_seed(17, "b")
    assert seed_record(17, "I", "S2", 0, "alpha_005") != seed_record(17, "I", "S2", 0, "weak_signal")
