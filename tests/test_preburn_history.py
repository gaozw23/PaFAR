import numpy as np

from pafar_sim.exp2.dgp import generate_exp2
from pafar_sim.exp2.features import build_raw_features, score_history_mask
from pafar_sim.exp2.runner import _prediction_matrix
from pafar_sim.score import causal_moving_average


def test_preburn_rows_are_scored_but_not_fit_or_alert_eligible():
    batch = generate_exp2(np.random.default_rng(101), 12, "E2", hmax=20, tmin=6, force_event=False)
    fitting = build_raw_features(batch, batch.eligible)
    history_mask = score_history_mask(batch, 6, 3)
    history = build_raw_features(batch, history_mask)
    assert fitting.hour.min() == 6
    assert set((4, 5, 6)) <= set(history.hour)
    assert not batch.eligible[:, :5].any()
    prediction = history.hour.astype(float) / 100
    matrix = _prediction_matrix(prediction, history, len(batch.patient_ids), 20)
    patient = np.flatnonzero(batch.eligible[:, 5] & np.all(np.isfinite(matrix[:, 3:6]), axis=1))[0]
    smooth = causal_moving_average(matrix, 3)
    assert smooth[patient, 5] == np.mean(matrix[patient, 3:6])
    assert np.isfinite(matrix[patient, 3:6]).all()
    assert np.isnan(matrix[patient, :3]).all()

