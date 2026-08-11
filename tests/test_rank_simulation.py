import numpy as np

from pafar_sim.calibration import marginal_threshold


def test_exchangeable_rank_with_continuous_and_tied_scores():
    rng = np.random.default_rng(123)
    for tied in (False, True):
        exceeds = []
        for _ in range(2500):
            sample = rng.normal(size=20)
            new = rng.normal()
            if tied:
                sample = np.round(sample, 1); new = np.round(new, 1)
            result = marginal_threshold(sample, .10)
            exceeds.append(new > result.threshold)
        # Monte Carlo tolerance; strict ties can only lower exceedance.
        assert np.mean(exceeds) < result.alpha_m0 + .025

