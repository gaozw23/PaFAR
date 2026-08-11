import numpy as np

from pafar_sim.score import causal_moving_average, clipped_logit, eligible_mask, q_signal, trajectory_max


def test_smoother_is_causal_and_averages_probabilities():
    risk = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    original = causal_moving_average(risk, 3)
    changed = risk.copy(); changed[0, 4] = 0.99
    future_changed = causal_moving_average(changed, 3)
    np.testing.assert_allclose(original[:, :4], future_changed[:, :4])
    np.testing.assert_allclose(original[0], [0.1, 0.15, 0.2, 0.3, 0.4])


def test_signal_q_boundaries():
    np.testing.assert_allclose(q_signal(np.array([7, 6, 4.5, 3, 1, 0, -1])), [0, 0, .5, 1, 1, 0, 0])


def test_eligible_and_empty_maximum():
    mask = eligible_mask(np.array([8, 4]), np.array([7, np.inf]), 6, 10)
    assert np.flatnonzero(mask[0]).tolist() == [5]
    assert not mask[1].any()
    maxima = trajectory_max(np.ones((2, 10)), mask)
    assert maxima[0] == 1 and np.isneginf(maxima[1])


def test_clipped_logit_is_finite():
    assert np.isfinite(clipped_logit(np.array([[0.0, 1.0]]))).all()

