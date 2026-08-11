# Metric formula audit

## Scope

This audit covers Equations (10)--(13), (47), and (48). It changes no DGP parameter, alert threshold, strict `>` crossing rule, signal strength, or learner hyperparameter. Regression evidence is saved in `outputs/diagnostics/metric_formula_regression_check.csv` and `tests/test_metrics.py`.

## Evaluable warning sets

For patient `i`, the implementation now constructs the warning sets directly from the actual Boolean eligibility matrix and integer hour coordinates:

- `W_i,0 = I_i ∩ [T_i-6, T_i]`
- `W_i,3 = I_i ∩ [T_i-6, T_i-3]`
- `eval0_i = any(W_i,0)`
- `eval3_i = any(W_i,3)`

No horizon/onset shortcut is used. Pre-burn score-history hours can be finite but remain absent from `I_i`, so they cannot make a patient evaluable and cannot become a first alert.

`Sens0` is the fraction of `eval0` patients whose first eligible strict alert lies in `[T-6,T]`. `Sens3` is the corresponding fraction among `eval3` patients with a first alert in `[T-6,T-3]`. Undefined denominators remain `NA`.

## Standardized PPV

The stored components are now:

- `a0 = any_alert_rate_non_event`
- `a1 = any_alert_rate_event`
- `v3 = valid3_rate_all_events`
- `v0 = valid0_rate_all_events`

Crucially, `v3` and `v0` divide by all event patients, not by only evaluable event patients. The implementation evaluates

`PPV_l(pi) = pi*v_l / (pi*a1 + (1-pi)*a0)`.

`Sens3`/`Sens0` still use their evaluable denominators, so sensitivity and standardized PPV intentionally use different denominators. The three-event regression is implemented in `tests/test_metrics.py::test_ppv_uses_all_events_but_sensitivity_uses_actual_warning_set` and persisted separately as `outputs/diagnostics/ppv_denominator_regression_check.csv`. It contains one valid/evaluable event, one non-evaluable event, one evaluable premature event, and a non-event. It verifies `n_event_total=3`, `Sens3=1/2`, `v3=1/3`, and computes PPV from `1/3`, not `1/2`.

## Alert burden

The previous implementation first calculated two class-specific episode rates and then averaged the rates:

`100 * [(1-pi)*(mu_N0/mu_E0) + pi*(mu_N1/mu_E1)]`.

That is not Equation (48). The corrected implementation uses a prevalence-mixture numerator divided by a prevalence-mixture exposure denominator:

`AB(pi) = 100 * [(1-pi)*mu_N0 + pi*mu_N1] / [(1-pi)*mu_E0 + pi*mu_E1]`.

The separate `outputs/diagnostics/metric_formula_regression_check.csv` is the two-event artificial example for alert burden only; it is not the PPV-denominator example. In that burden example, the old expression is `468.923077`, while the correct ratio of mixtures is `466.019417`; the test requires these to differ and requires exact agreement with the latter construction.

## Saved composition fields and reweighting

Every successful method/replicate row now saves:

- `any_alert_rate_non_event`, `any_alert_rate_event`
- `valid3_rate_all_events`, `valid0_rate_all_events`
- `mean_episodes_non_event`, `mean_episodes_event`
- `mean_exposure_days_non_event`, `mean_exposure_days_event`
- `n_event_total`, `n_event_evaluable3`, `n_event_evaluable0`

The primary columns use prevalence 0.10. The same alert outcomes are reweighted at prevalence 0.05 into `ppv3_standardized_pi050`, `ppv0_standardized_pi050`, and `alert_burden_100d_pi050`. No DGP generation, learner fitting, calibration, or alert detection is repeated for prevalence reweighting.

## Verification

- `tests/test_metrics.py` verifies ratio-of-mixtures burden, the all-event PPV numerator, actual eligible warning sets, undefined denominators, and infinite-threshold conventions.
- All 49 tests pass.
- Cold-start smoke reports 16/16 checks, legal PFA/Sens/PPV ranges, nonnegative burden, and zero learner failures.
- No `NA` result is replaced by zero.
