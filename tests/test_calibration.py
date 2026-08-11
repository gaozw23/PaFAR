import numpy as np
import pytest

from pafar_sim.calibration import (
    hc_index, hc_threshold, marginal_index, marginal_threshold, minimum_marginal_m0,
    weighted_quantile, youden_threshold,
)


def test_marginal_index_and_one_based_statistic():
    k, achieved = marginal_index(500, .10)
    assert k == 451
    assert achieved == pytest.approx(50 / 501)
    result = marginal_threshold(np.arange(1, 501), .10)
    assert result.threshold == 451


def test_minimum_sample_size_and_infinity():
    assert minimum_marginal_m0(.10) == 9
    assert np.isposinf(marginal_threshold(np.arange(8), .10).threshold)
    assert np.isfinite(marginal_threshold(np.arange(9), .10).threshold)


@pytest.mark.parametrize("m0,alpha,expected", [
    (500, .10, 462), (100, .10, 96), (29, .10, 29), (28, .10, 29),
    (59, .05, 59), (58, .05, 59),
])
def test_hc_indices(m0, alpha, expected):
    assert hc_index(m0, alpha, .05) == expected
    result = hc_threshold(np.arange(m0, dtype=float), alpha, .05)
    assert result.infinite == (expected == m0 + 1)


def test_patient_equal_weighted_quantile():
    # Patient 1 has one score, patient 2 has four; each patient carries total weight 1/2.
    values = np.array([0., 10., 11., 12., 13.])
    weights = np.array([.5, .125, .125, .125, .125])
    assert weighted_quantile(values, weights, .5) == 0.


def test_youden_strict_candidates_and_largest_tie_break():
    risk = np.array([.1, .2, .8, .9])
    labels = np.array([0, 0, 1, 1])
    patient = np.arange(4)
    # c=.2 gives perfect separation under strict risk > c.
    assert youden_threshold(risk, labels, patient) == .2
