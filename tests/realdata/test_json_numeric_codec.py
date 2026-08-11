from __future__ import annotations

import json

import numpy as np
import pytest

from pafar_sim.alerting import first_alert
from pafar_sim.calibration import ThresholdResult
from pafar_sim.realdata.calibration import MethodThreshold
from pafar_sim.realdata.internal_analysis import _serialize_threshold_records
from pafar_sim.realdata.json_utils import (
    decode_json_numeric,
    dumps_json_numeric,
    encode_json_numeric,
    loads_json_numeric,
)
from pafar_sim.realdata.transfer_analysis import _transfer_threshold_row
from pafar_sim.realdata.scoring import TrajectoryData, evaluate


def test_json_codec_finite_roundtrip():
    assert loads_json_numeric(dumps_json_numeric(1.25)) == 1.25


def test_json_codec_neg_inf_roundtrip():
    assert np.isneginf(loads_json_numeric(dumps_json_numeric(-np.inf)))


def test_json_codec_pos_inf_roundtrip():
    assert np.isposinf(loads_json_numeric(dumps_json_numeric(np.inf)))


def test_json_codec_array_with_inf():
    decoded = np.asarray(loads_json_numeric(dumps_json_numeric(np.array([1.0, -np.inf, np.inf]))))
    assert decoded[0] == 1.0 and np.isneginf(decoded[1]) and np.isposinf(decoded[2])


def test_json_codec_nested_structure():
    source = {"x": (np.float32(2.5), {"boundary": np.array([-np.inf, 4.0])})}
    decoded = decode_json_numeric(json.loads(dumps_json_numeric(source)))
    assert decoded["x"][0] == 2.5 and np.isneginf(decoded["x"][1]["boundary"][0])


def test_json_codec_rejects_unexpected_nan():
    with pytest.raises(ValueError, match="NaN"):
        encode_json_numeric({"threshold": np.nan})


def test_internal_threshold_neg_inf_serialization():
    record = MethodThreshold("Bonferroni", "A", .05, np.array([1.0, -np.inf]), 10, None, None, False, (6, 12, 168))
    rows, diagnostic = _serialize_threshold_records({.05: [record]})
    decoded = np.asarray(loads_json_numeric(rows.loc[0, "threshold"]))
    assert np.isneginf(decoded[-1])
    assert rows.loc[0, "threshold_state"] == "neg_inf"
    assert diagnostic[0]["classification"] == "algorithm_defined_sentinel"


def test_transfer_threshold_pos_inf_serialization():
    threshold = ThresholdResult(np.inf, 101, 100, .01, 0.0)
    row = _transfer_threshold_row(direction="A_to_B", strategy="Local PaFAR-F", target_m0=100, scale="F", threshold=threshold)
    assert np.isposinf(row["threshold"])
    assert row["threshold_state"] == "pos_inf" and row["threshold_has_pos_inf"]


def test_json_is_strict_standard_json():
    text = dumps_json_numeric({"lo": -np.inf, "hi": np.inf})
    parsed = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)))
    assert parsed == {"lo": {"__float__": "neg_inf"}, "hi": {"__float__": "pos_inf"}}
    assert "Infinity" not in text and "NaN" not in text


def test_threshold_semantics_unchanged_after_roundtrip():
    score = np.array([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]])
    eligible = np.ones_like(score, dtype=bool)
    threshold = np.array([-np.inf, 0.5, np.inf])
    decoded = np.asarray(loads_json_numeric(dumps_json_numeric(threshold)))
    assert np.array_equal(decoded, threshold)
    assert np.array_equal(first_alert(score, eligible, threshold), first_alert(score, eligible, decoded))


def test_serialization_only_preserves_alerts_pfa_and_sensitivity():
    score = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, 2.0], [-2.0, -1.0]])
    eligible = np.ones_like(score, dtype=bool)
    event = np.array([False, False, True, True])
    data = TrajectoryData(
        np.array(["n1", "n2", "e1", "e2"]), np.array(["A"] * 4), event,
        np.array([np.inf, np.inf, 7.0, 7.0]), np.array([2] * 4), score, score,
        eligible, eligible, np.zeros_like(score, dtype=np.uint8),
    )
    threshold = -np.inf
    decoded = loads_json_numeric(dumps_json_numeric(threshold))
    before_metric, before_detail = evaluate(data, score, threshold)
    after_metric, after_detail = evaluate(data, score, decoded)
    assert np.isneginf(decoded)
    assert np.array_equal(before_detail["tau"], after_detail["tau"])
    assert np.array_equal(score > threshold, score > decoded)
    assert before_metric.pfa == after_metric.pfa
    assert before_metric.sens3 == after_metric.sens3
