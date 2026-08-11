import numpy as np
import pandas as pd

from pafar_sim.aggregation import aggregate_results
from pafar_sim.output_schema import describe_method


def test_required_method_schema_and_target_strategies():
    direct = describe_method("Direct source transfer", "E3")
    assert direct == {
        "method_family": "pafar_f", "deployment_strategy": "direct_source_transfer",
        "threshold_scale": "transformed_logit", "threshold_origin": "source_calibration_non_events",
        "template_origin": "none", "is_alias": False, "alias_of": "",
    }
    local_t = describe_method("Local PaFAR-T", "E3")
    assert local_t["deployment_strategy"] == "local_target_recalibration"
    assert local_t["threshold_scale"] == "standardized_time_template"
    assert local_t["template_origin"] == "source_validation_non_events"
    assert describe_method("Binwise Bonferroni", "E2")["threshold_scale"] == "vector_transformed_logit"
    assert describe_method("Fixed 0.5", "E1")["threshold_scale"] == "risk_probability"


def test_aggregation_excludes_raw_alias():
    rows = pd.DataFrame([
        {"experiment": "I", "scenario": "S4", "site": "B", "method": "Direct source transfer", "alpha": .1, "target_m0": 0, "pfa": .1, "learner_failure": False, "is_alias": False},
        {"experiment": "I", "scenario": "S4", "site": "B", "method": "PaFAR-F", "alpha": .1, "target_m0": np.nan, "pfa": .1, "learner_failure": False, "is_alias": True},
    ])
    summary = aggregate_results(rows)
    assert set(summary.method) == {"Direct source transfer"}
    assert set(summary.target_m0) == {0}


def test_local_method_metadata_identifies_target_recalibration():
    local = describe_method("Local PaFAR-F", "S4")
    assert local["deployment_strategy"] == "local_target_recalibration"
    assert local["threshold_origin"] == "target_calibration_non_events"
