import numpy as np
import pandas as pd
import json

from pafar_sim.aggregation import aggregate_results
from pafar_sim.calibration import fit_time_template, marginal_threshold
from pafar_sim.exp1.dgp import generate_exp1
from pafar_sim.io_utils import checkpoint_complete, write_checkpoint
from pafar_sim.metrics import evaluate_metrics
from pafar_sim.score import causal_moving_average, clipped_logit, trajectory_max


def test_dgp_to_save_and_aggregate(tmp_path):
    val = generate_exp1(np.random.default_rng(1), 60, "S1", False)
    cal = generate_exp1(np.random.default_rng(2), 20, "S1", False)
    test = generate_exp1(np.random.default_rng(3), 30, "S1", False)
    u_val = clipped_logit(causal_moving_average(val.risk))
    template = fit_time_template(u_val, val.eligible, 6, 120)
    u_cal = clipped_logit(causal_moving_average(cal.risk))
    threshold = marginal_threshold(trajectory_max(template.transform(u_cal), cal.eligible), .1)
    u_test = template.transform(clipped_logit(causal_moving_average(test.risk)))
    metrics = evaluate_metrics(u_test, test.eligible, threshold.threshold, test.event, test.onset, test.horizon)
    frame = pd.DataFrame([{"experiment": "I", "scenario": "S1", "site": "A", "method": "PaFAR-T", "alpha": .1,
                           "target_m0": np.nan, "learner_failure": False, **metrics.as_dict()}])
    path = tmp_path / "rep_000001.csv.gz"
    write_checkpoint(frame, path, "abc")
    assert checkpoint_complete(path, "abc")
    sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    assert len(sidecar["implementation_checksum"]) == 64
    assert not aggregate_results(frame).empty


def test_checkpoint_threshold_round_trip_requires_exact_parser(tmp_path):
    value = float.fromhex("0x1.58a6850cd1a90p-4")
    path = tmp_path / "rep_000002.csv.gz"
    write_checkpoint(pd.DataFrame([{"threshold": value}]), path, "roundtrip")
    restored = pd.read_csv(path, float_precision="round_trip").threshold.iloc[0]
    assert restored == value and restored.hex() == value.hex()


def test_uint64_seed_is_serialized_as_text(tmp_path):
    seed = "u64:18446744073709551615"
    path = tmp_path / "rep_000003.csv.gz"
    write_checkpoint(pd.DataFrame([{"replicate_seed": seed, "seed_record": '{"replicate":18446744073709551615}'}]), path, "seed")
    restored = pd.read_csv(path, dtype={"replicate_seed": "string"})
    assert restored.replicate_seed.iloc[0] == seed
