"""Run deterministic diagnostics from saved smoke checkpoint seed records."""
from pathlib import Path
from pafar_sim.config import load_config
from pafar_sim.diagnostics import run_existing_smoke_audit

config = load_config("configs/smoke.yaml").data
artifacts = run_existing_smoke_audit(config, Path("outputs/smoke"), Path("outputs/diagnostics"))
bad = artifacts["independent"].loc[lambda d: ~(d.index_match & d.finite_status_match & d.threshold_match & d.exceedance_match)]
preburn_ok = bool(artifacts["preburn"].exact_equal.all() and
                  (~artifacts["preburn"].preburn_in_fitting_mask).all() and
                  (~artifacts["preburn"].preburn_in_first_alert_eligibility).all())
reservoir_ok = bool(artifacts["reservoir"].all_reservoir_event_false.all() and
                    artifacts["reservoir"].all_reservoir_onset_infinite.all() and
                    artifacts["reservoir"].available_prefixes_nested.all())
alpha_ok = bool((artifacts["alpha_grid"].n_alpha == 5).all() and
                (artifacts["alpha_grid"].seed_record_unique == 1).all() and
                (artifacts["alpha_grid"].best_iteration_unique == 1).all() and
                artifacts["alpha_grid"].threshold_nonincreasing.all())
metric_ok = bool(artifacts["metric_formula"].formulas_differ.all())
ppv_ok = bool(artifacts["ppv_denominator"].stored_ppv_matches_all_event_formula.all() and
              artifacts["ppv_denominator"].stored_ppv_differs_from_evaluable_formula.all())
print(f"distribution rows={len(artifacts['distribution'])}")
print(f"threshold audit rows={len(artifacts['independent'])}, mismatches={len(bad)}")
print(f"preburn={preburn_ok}, conditional_reservoir={reservoir_ok}, metric_formula={metric_ok}, ppv_denominator={ppv_ok}, alpha_grid={alpha_ok}")
print(artifacts["learner"][["scenario", "best_iteration", "raw_prediction_n_unique", "smoothed_prediction_n_unique"]].to_string(index=False))
raise SystemExit(1 if len(bad) or not all((preburn_ok, reservoir_ok, metric_ok, ppv_ok, alpha_ok)) else 0)
