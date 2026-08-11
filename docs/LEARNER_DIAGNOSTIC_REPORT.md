# XGBoost learner diagnostic report

## Scope

This report audits the existing single-replicate E1/E2/E3 smoke checkpoints. It reconstructs the deterministic smoke replicate, refits the configured learner without changing the DGP, feature definitions, signal strength, smoothing, XGBoost hyperparameters, number of boosting rounds, early-stopping rule, or threshold rule. The row-level evidence is in `outputs/diagnostics/learner_diagnostics.csv`.

## Learner summary

| Scenario | Train patients / rows | Validation patients / rows | Positive train patients / hours | Positive validation patients / hours | pi_y_hat | Kfit min / median / max | Best iteration | Trees used | Best score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 400 / 19,326 | 150 / 7,129 | 34 / 204 | 18 / 108 | 0.028626 | 6 / 39 / 91 | 28 | 29 | 0.195420 |
| E2 | 400 / 18,814 | 150 / 7,036 | 37 / 222 | 13 / 78 | 0.027010 | 6 / 38 / 93 | 0 | 1 | 0.051429 |
| E3 | 400 / 18,811 | 150 / 7,144 | 36 / 216 | 13 / 78 | 0.024222 | 9 / 39 / 92 | 0 | 1 | 0.057092 |

`prediction_iteration_end` is respectively 29, 1, and 1, so it is exactly `best_iteration + 1`. E2 and E3 stop at the first tree; this is the immediate source of their very discrete smoke predictions. It is an observed consequence of the configured learner and small smoke sample, not an iteration-range or checkpoint bug.

## Discrimination and score resolution

| Scenario | Weighted train AUROC | Weighted validation AUROC | Weighted train PR-AUC | Weighted validation PR-AUC | Raw prediction unique values | Smoothed prediction unique values |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.977388 | 0.750305 | 0.780421 | 0.180878 | 12,200 | 14,311 |
| E2 | 0.803652 | 0.599396 | 0.235241 | 0.090780 | 8 | 110 |
| E3 | 0.828210 | 0.645413 | 0.358129 | 0.179394 | 8 | 111 |

The training matrices contain 543 columns: E1 `(19,326, 543)`, E2 `(18,814, 543)`, and E3 `(18,811, 543)`. Constant/zero-variance columns account for 6.629834% in E1, 0% in E2, and 6.629834% in E3. These percentages do not support the hypothesis that imputation made most dynamic features constant.

Missingness is recorded as actual percentages by feature family. The largest family-level values are the rolling `change`, `sd`, and `slope` families: 26.9079% in E1, 46.7585% in E2, and 23.5709% in E3. The `mean`, `min`, and `max` family percentages are 9.9575%, 23.6520%, and 8.3400%, respectively. Baseline/current/count/never families are 0%. Full family values and the top 20 mapped feature importances are preserved in the CSV rather than reported as anonymous `f#` indices.

## Integrity checks

All three scenarios passed every implementation-integrity check:

- train, validation, and prediction matrices have identical feature-column order;
- DMatrix feature-name state is identical (all matrices intentionally use the same unnamed-column representation);
- prediction with `iteration_range=(0, best_iteration + 1)` exactly reconstructs the stored prediction;
- prediction without `iteration_range` differs, confirming that the selected early-stopping range matters;
- target predictions are newly computed and do not reuse source prediction memory;
- feature rows remain aligned with labels, patient IDs, and time indices;
- validation is the sole and final early-stopping evaluation set;
- causal score smoothing is applied after prediction, not before training.

## Conclusions

1. E2/E3 `best_iteration=0` is genuinely very early and yields only eight raw probability values in each smoke replicate.
2. The moving average expands the score support but cannot remove the endpoint atoms inherited from the one-tree learner.
3. No source/target reuse, feature-order mismatch, row misalignment, incorrect early-stopping eval order, or misplaced smoother was found.
4. The E2/E3 smoke threshold collapse is therefore XGBoost score discreteness under the specified smoke settings. It is not an implementation failure.
5. The production-size one-replicate pilots selected iterations 38, 54, and 23 for E1/E2/E3, and their patient maxima were essentially continuous; this supports treating the smoke degeneracy as a small-sample/early-stopping diagnostic finding, not as a reason to modify the registered learner.

