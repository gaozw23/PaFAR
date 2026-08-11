from dataclasses import replace
import numpy as np

from pafar_sim.exp2.dgp import generate_exp2
from pafar_sim.exp2.features import FeaturePreprocessor, build_raw_features, feature_names


def test_future_measurements_do_not_change_past_features():
    batch = generate_exp2(np.random.default_rng(9), 4, "E1", force_event=False)
    original = build_raw_features(batch)
    changed_measurements = batch.measurements.copy()
    changed_measurements[:, 10:, :] = 999.
    changed = build_raw_features(replace(batch, measurements=changed_measurements))
    past = original.hour <= 10
    np.testing.assert_allclose(original.values[past], changed.values[past], equal_nan=True)


def test_six_hour_window_excludes_t_minus_six():
    batch = generate_exp2(np.random.default_rng(11), 1, "E1", force_event=False)
    measurements = np.full_like(batch.measurements, np.nan)
    measurements[0, 0, 0] = 100  # t=1 must be excluded from the t=7 six-hour window.
    measurements[0, 1:7, 0] = 1
    rows = build_raw_features(replace(batch, measurements=measurements))
    name = feature_names().index("x1_mean_6h")
    row = np.flatnonzero(rows.hour == 7)[0]
    assert rows.values[row, name] == 1


def test_imputation_schema_is_frozen_by_feature_semantics():
    batch = generate_exp2(np.random.default_rng(12), 5, "E1", force_event=False)
    rows = build_raw_features(batch)
    prep = FeaturePreprocessor.fit(rows)
    assert "x1_slope_6h__missing" in prep.output_names
    assert "x1_count_6h__missing" not in prep.output_names
    assert np.isfinite(prep.transform(rows)).all()
