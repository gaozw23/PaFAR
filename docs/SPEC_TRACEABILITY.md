# PaFAR specification traceability

The statistical source is `PaFAR.pdf` (working manuscript, 2026-08-02). `v40i08.pdf` informs only the engineering choices documented in `PERFORMANCE_NOTES.md`.

| Source item | Python implementation | Tests | Principal output fields |
|---|---|---|---|
| Eq. (2), eligible monitoring set | `score.eligible_mask` | `test_score.py`, `test_features_causality.py` | denominator counts, `n_non_events`, `n_events` |
| Eq. (4), causal three-score mean | `score.causal_moving_average` | `test_score.py` | all method scores (not persisted patient-wise) |
| Eqs. (5)-(7), clipped logit, standardized score, maximum | `score.clipped_logit`, `calibration.TimeTemplate.transform`, `score.trajectory_max` | `test_score.py`, `test_time_template.py` | `threshold`, template fields |
| Eq. (8), first formal alert | `alerting.first_alert` | `test_alerting.py` | PFA/sensitivity/burden metrics, cumulative curve |
| Eqs. (9)-(13), PFA, Sens3, Sens0, premature, PPV | `metrics.evaluate_metrics` | `test_metrics.py` | `pfa`, `sens3`, `sens0`, `premature`, `ppv*_standardized` |
| Eq. (14), marginal PaFAR | `calibration.marginal_index`, `marginal_threshold` | `test_calibration.py`, `test_rank_simulation.py` | `m0`, `threshold_index`, `alpha_m0`, `infinite_threshold` |
| Eqs. (17)-(22), time bins, weighted median/MAD | `calibration.initial_boundaries`, `merge_sparse_bins`, `weighted_quantile`, `fit_time_template` | `test_time_template.py`, `test_calibration.py` | `template_boundaries`, `template_locations`, `template_scales`, `raw_bin_boundaries` |
| Eqs. (24)-(25), PaFAR-HC | `calibration.hc_index`, `hc_threshold` | `test_calibration.py` | `delta`, `threshold_index`, `conditional_pfa_*` |
| Eqs. (30)-(32), labels and XGBoost weights | `features.build_raw_features`, `learner.training_weights`, `validation_weights` | `test_training_weights.py`, `test_features_causality.py` | `pi_y_hat`, `best_iteration`, learner fields |
| Eqs. (33)-(39), Experiment I DGP | `exp1.dgp.generate_exp1`, `stationary_ar1`, `scenario_parameters` | `test_exp1_dgp.py` | scenario/site/signal and timing fields |
| Eqs. (40)-(46), Experiment II DGP | `exp2.dgp.generate_exp2`, `latent_severity`, `event_probabilities` | `test_exp2_dgp.py`, `test_features_causality.py` | scenario/site/realized calibration m0 |
| Eqs. (47)-(48), standardized PPV and burden | `metrics._standardized_ppv`, `evaluate_metrics` | `test_metrics.py` | `ppv3_standardized`, `ppv0_standardized`, `alert_burden_100d` |

## Appendix C checklist (items 1-17)

1. Software/RNG/seeds/configuration: `rng.py`, manifests; `test_rng_reproducibility.py`; seed and checksum fields.
2. Frozen learner/smoother/template/scalar threshold split: experiment runners; integration test; template and threshold fields.
3. Event reconstruction/onset/eligible windows: both DGP modules and `eligible_mask`; DGP/feature tests; denominator fields.
4. Calibration unit is patient trajectory maximum: `trajectory_max`; calibration tests; `m0` and index.
5. Patient-level split and patient-weighted training: `learner.py`; training-weight test; `pi_y_hat`.
6. Feature windows, missingness, imputation and column freeze: `features.py`; causality/window tests; feature time.
7. Locked model, best iteration and explicit prediction range: `learner.FittedLearner`; smoke validation; best-iteration fields.
8. Frozen smoother/template/bin merge/MAD floor: `score.py`, `calibration.py`; score/template tests; template metadata.
9. Non-event calibration and strict inequality: runners and `first_alert`; alert/calibration tests; threshold/PFA fields.
10. Tied-score rank correction: `marginal_threshold`; rank simulation; `alpha_m0`.
11. HC binomial tail via `scipy.stats.binom.sf`: `hc_index`; exact-index tests; delta/conditional-risk fields.
12. Stable alternative index checks: unit tests cover analytic marginal and HC boundaries; output indices.
13. Undefined-value conventions: `metrics.py`; metric tests; NA plus denominator counts.
14. Eligible exposure/episodes/standardized burden: `metrics.py`, `alerting.py`; alert/metric tests; burden fields.
15. Direct source transfer and local target recalibration: S4/E3 runners; smoke nested-prefix checks; `target_m0`.
16. Oracle-F and conditional-risk check: `exp1.oracle`; integration/smoke checks; `conditional_pfa_oracle`.
17. Saved-output-only tables/figures: `plotting.py`; smoke artifact checks; tables, figures, JSON sidecars.

