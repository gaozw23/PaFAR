import numpy as np
import pandas as pd

from pafar_sim.aggregation import aggregate_learner_metrics, aggregate_results
from pafar_sim.calibration import hc_threshold, marginal_threshold
from pafar_sim.plotting import make_tables_and_figures


def test_pafar_thresholds_are_nonincreasing_across_alpha_grid():
    maxima = np.linspace(-2, 2, 100)
    alpha = [.02, .05, .10, .15, .20]
    marginal = [marginal_threshold(maxima, value).threshold for value in alpha]
    hc = [hc_threshold(maxima, value, .05).threshold for value in alpha]
    assert np.all(np.diff(marginal) <= 0)
    assert np.all(np.diff(hc) <= 0)


def test_aggregate_schema_keeps_canonical_direct_and_formula_components():
    rows = []
    for replicate, threshold, exceed in ((0, 1., False), (1, 3., True)):
        rows.append({
            "experiment": "Experiment I", "condition": "primary", "scenario": "S4", "site": "B",
            "method": "Direct source transfer", "alpha": .1, "operating_alpha": .1,
            "operating_point": "primary", "target_m0": 0, "replicate": replicate,
            "pfa": .1 + replicate * .02, "alert_episodes_per_patient": 1 + replicate,
            "any_alert_rate_non_event": .1, "any_alert_rate_event": .4,
            "valid3_rate_all_events": .2, "valid0_rate_all_events": .3,
            "mean_episodes_non_event": 1., "mean_episodes_event": 2.,
            "mean_exposure_days_non_event": 3., "mean_exposure_days_event": 2.,
            "conditional_pfa_gt_alpha": exceed, "threshold": threshold,
            "infinite_threshold": False, "learner_failure": False, "is_alias": False,
        })
    rows.append({**rows[0], "method": "PaFAR-F", "is_alias": True, "alias_of": "Direct source transfer"})
    summary = aggregate_results(pd.DataFrame(rows))
    assert set(summary.method) == {"Direct source transfer"}
    exceed = summary[summary.metric == "conditional_pfa_gt_alpha"].iloc[0]
    assert exceed["mean"] == .5 and exceed.n_total == 2
    threshold = summary[summary.metric == "threshold"].iloc[0]
    assert threshold.threshold_mean == 2 and np.isclose(threshold.threshold_sd, np.sqrt(2))
    assert np.isnan(threshold.mcse)


def test_learner_metrics_are_summarized_once_per_replicate():
    learner = pd.DataFrame({
        "experiment": ["Experiment II"] * 2, "condition": ["primary"] * 2,
        "scenario": ["E1"] * 2, "site": ["A"] * 2, "score_generator": ["xgboost"] * 2,
        "replicate": [0, 1], "auroc_weighted": [.7, .9], "aucpr_weighted": [.2, .4],
        "best_iteration": [10, 20], "learner_failure": [False, False],
    })
    summary = aggregate_learner_metrics(learner)
    assert len(summary) == 1 and summary.iloc[0].n_total == 2
    assert summary.iloc[0].auroc_mean == .8 and summary.iloc[0].best_iteration_mean == 15


def test_manuscript_tables_contain_aggregate_rows_not_replicates(tmp_path):
    rows = []
    for scenario, experiment, method, target_m0 in (
        ("S1", "Experiment I", "PaFAR-F", np.nan),
        ("E1", "Experiment II", "PaFAR-F", np.nan),
        ("S4", "Experiment I", "Direct source transfer", 0),
        ("E3", "Experiment II", "Direct source transfer", 0),
    ):
        for replicate in (0, 1):
            rows.append({
                "experiment": experiment, "condition": "primary", "scenario": scenario, "site": "B" if scenario in {"S4", "E3"} else "A",
                "method": method, "alpha": .1, "operating_alpha": .1, "operating_point": "primary",
                "target_m0": target_m0, "calibration_m0": 500, "replicate": replicate,
                "pfa": .08 + .01 * replicate, "sens3": .4, "sens0": .5, "premature": .1,
                "median_lead": 4., "ppv3_standardized": .3, "alert_burden_100d": 2.,
                "threshold": 1. + replicate, "conditional_pfa_oracle": .09,
                "cumulative_false_alert_curve": "[0.0,0.1]", "infinite_threshold": False,
                "learner_failure": False, "is_alias": False,
            })
    source = tmp_path / "raw.csv.gz"; pd.DataFrame(rows).to_csv(source, index=False)
    make_tables_and_figures(source, tmp_path)
    for name in ("table3_experiment1.csv", "table4_experiment2_e1_e2.csv", "table5_target_shift.csv"):
        table = pd.read_csv(tmp_path / "tables" / name)
        assert "replicate" not in table.columns
    target = pd.read_csv(tmp_path / "tables" / "table5_target_shift.csv")
    assert set(target.method) == {"Direct source transfer"}
    assert set(target.target_m0) == {0}
