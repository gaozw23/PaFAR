import numpy as np

from pafar_sim.metrics import evaluate_metrics


def test_na_and_infinite_threshold_conventions():
    score = np.zeros((2, 10))
    eligible = np.ones_like(score, dtype=bool)
    event = np.array([False, True])
    onset = np.array([np.inf, 10.])
    horizon = np.array([10, 10])
    result = evaluate_metrics(score, eligible, np.inf, event, onset, horizon)
    assert result.pfa == 0 and result.sens0 == 0 and result.sens3 == 0
    assert np.isnan(result.ppv0_standardized) and np.isnan(result.median_lead)


def test_empty_sensitivity_denominator_is_na():
    result = evaluate_metrics(np.zeros((1, 2)), np.ones((1, 2), bool), np.inf,
                              np.array([False]), np.array([np.inf]), np.array([2]))
    assert np.isnan(result.sens0) and np.isnan(result.sens3)


def test_alert_burden_is_ratio_of_prevalence_mixtures():
    score = np.zeros((4, 12)); eligible = np.zeros((4, 12), dtype=bool)
    eligible[0, :12] = True; score[0, [0, 7]] = 1
    eligible[1, :2] = True; score[1, 0] = 1
    eligible[2, :12] = True; score[2, 0] = 1
    eligible[3, :6] = True
    event = np.array([False, False, True, True])
    onset = np.array([np.inf, np.inf, 13., 13.]); horizon = np.array([12, 2, 12, 6])
    result = evaluate_metrics(score, eligible, .5, event, onset, horizon, prevalence=.10)
    expected = 100 * (.9 * result.mean_episodes_non_event + .1 * result.mean_episodes_event) / (
        .9 * result.mean_exposure_days_non_event + .1 * result.mean_exposure_days_event
    )
    old = 100 * (.9 * result.mean_episodes_non_event / result.mean_exposure_days_non_event +
                 .1 * result.mean_episodes_event / result.mean_exposure_days_event)
    assert result.alert_burden_100d == expected
    assert not np.isclose(expected, old)


def test_ppv_uses_all_events_but_sensitivity_uses_actual_warning_set():
    score = np.zeros((4, 10)); eligible = np.zeros((4, 10), dtype=bool)
    # Valid event: W3 contains t=4..7 and first alert occurs at t=5.
    eligible[0, 3:10] = True; score[0, 4] = 1
    # Non-evaluable event: no eligible hour in [T-6,T].
    eligible[1, 0] = True
    # Evaluable premature event: first alert at t=1, before T-6.
    eligible[2, :10] = True; score[2, 0] = 1
    # Alerting non-event.
    eligible[3, :10] = True; score[3, 1] = 1
    event = np.array([True, True, True, False])
    onset = np.array([10., 10., 10., np.inf]); horizon = np.array([10, 1, 10, 10])
    result = evaluate_metrics(score, eligible, .5, event, onset, horizon, prevalence=.10)
    assert result.n_event_total == 3
    assert result.n_event_evaluable3 == 2 and result.n_event_evaluable0 == 2
    assert result.sens3 == .5 and result.valid3_rate_all_events == 1 / 3
    assert result.sens0 == .5 and result.valid0_rate_all_events == 1 / 3
    expected_ppv = .1 * (1 / 3) / (.1 * (2 / 3) + .9 * 1.0)
    assert np.isclose(result.ppv3_standardized, expected_ppv)

