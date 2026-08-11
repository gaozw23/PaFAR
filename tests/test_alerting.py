import numpy as np

from pafar_sim.alerting import count_alert_episodes, first_alert
from pafar_sim.score import trajectory_max


def test_strict_ties_and_maximum_equivalence():
    score = np.array([[1., 1., 0.], [0., 2., 3.]])
    eligible = np.ones_like(score, dtype=bool)
    tau = first_alert(score, eligible, 1.)
    assert np.isinf(tau[0]) and tau[1] == 2
    np.testing.assert_array_equal(np.isfinite(tau), trajectory_max(score, eligible) > 1.)


def test_episode_state_machine_requires_return_and_six_hours():
    score = np.array([[2, 2, 0, 0, 0, 0, 2, 2, 0, 2]], dtype=float)
    eligible = np.ones_like(score, dtype=bool)
    assert count_alert_episodes(score, eligible, 1.)[0] == 2

