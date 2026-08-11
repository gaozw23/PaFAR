import numpy as np

from pafar_sim.exp2.dgp import event_probabilities, latent_severity


def test_latent_severity_stationary_variance():
    rng = np.random.default_rng(5)
    n = 30_000
    severity = latent_severity(rng, np.zeros(n), np.zeros(n, bool), np.full(n, np.inf), 15, 1.0)
    expected = .35 / np.sqrt(1 - .85**2)
    assert abs(severity[:, -1].std() - expected) < .02


def test_marginal_prevalence_near_ten_percent():
    rng = np.random.default_rng(7)
    n = 300_000
    probability = event_probabilities(rng.normal(size=n), rng.binomial(1, .5, n), rng.normal(size=n))
    assert abs(probability.mean() - .10) < .005

