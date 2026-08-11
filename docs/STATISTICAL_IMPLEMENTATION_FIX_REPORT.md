# Statistical implementation fix report

## Scope and immutable choices

The implementation was corrected and revalidated without changing `PaFAR.pdf`, `v40i08.pdf`, the statistical method definitions, DGP parameters, signal strengths, strict `>` rule, smoothing length, or primary XGBoost parameters. No formal 500/250/100/50-replicate production simulation was run.

## 1. How many scores did the old Experiment II t=6 smoother average?

It averaged **one score, `R(6)`**. `build_raw_features` was hard-wired to `batch.eligible`; because eligibility starts at `tmin=6`, the prediction matrix contained `NaN` at t=4 and t=5. The moving-average function correctly ignored missing values, but its only available value in the t=4:6 window was `R(6)`.

## 2. Does corrected t=6 strictly use t=4,5,6?

Yes. `build_raw_features` now accepts an explicit `row_mask`. Model fitting and validation early stopping still use `batch.eligible`, while post-fit score generation uses

`t <= horizon`, `t < onset`, and `t >= max(1, tmin-L+1)`.

For primary `tmin=6,L=3`, this begins at t=4. Across 90 persisted validation/calibration/test checks, every t=6 score with complete history satisfies exact floating-point equality

`smoothed_R(6) == mean(R(4),R(5),R(6))`.

t=4,5 are absent from fitting and first-alert eligibility in every check. Template fitting, Youden, threshold-free metrics, patient maxima, PFA, sensitivity, and first alerts continue to read eligible hours only.

## 3. Why was the former target reservoir not conditional D=0?

`force_event=False` sampled `A`, `Q`, and `b` from their unconditional baseline laws and then overwrote the event label. Under Equation (40), event probability depends on those variables. Relabeling an unconditional patient does not draw `(A,Q,b,trajectory | D=0)` and therefore also gives the wrong conditional severity, biomarkers, and observation process.

## 4. How does corrected rejection/filter sampling work?

`generate_exp2_non_events` repeatedly generates candidate blocks from the complete natural target-site DGP with no event override. It retains only naturally generated `D=0` patients, continues until enough have been accepted, applies an independent `target_reservoir_order` child RNG, returns exactly the requested number, and renumbers patient IDs.

The retained patients all have `event=False` and `onset=inf`. Their baseline variables, frailty, severity, biomarkers, missingness, and observation process are generated under the same mechanism as naturally occurring target-test non-events. Candidate total and acceptance rate are saved. A single seed-fixed reservoir supplies nested prefixes; the production configuration uses 100/250/500.

The final smoke reservoir accepted 234 natural non-events from 256 candidates (`0.9140625`) and returned 100 ordered patients. Tests additionally compare a large helper draw with an independently generated natural-D=0 sample and verify the expected Equation (40) selection shift relative to the unconditional baseline population.

## 5. What did the old and corrected Equation (48) code compute?

The old code computed a prevalence-weighted mean of class-specific rates:

`100 * [(1-pi)*(mu_N0/mu_E0) + pi*(mu_N1/mu_E1)]`.

The corrected code computes Equation (48), a ratio of prevalence mixtures:

`100 * [(1-pi)*mu_N0 + pi*mu_N1] / [(1-pi)*mu_E0 + pi*mu_E1]`.

The artificial regression example gives `468.923077` under the old expression and `466.019417` under Equation (48). PPV was separately corrected so `v3/v0` divide by all event patients, while Sens3/Sens0 divide by their actual eligible warning-set populations. Details are in `docs/METRIC_FORMULA_AUDIT.md`.

## 6. Which smoke and pilot values changed?

The corrected smoke has **511 operating-point rows**, rather than 109 single-alpha rows, because each locked replicate now contains alpha 0.02/0.05/0.10/0.15/0.20. Fixed 0.5 and Youden still appear only once per replicate. Alpha rows share the same seed record, learner, test sample, and template.

For 95 directly aligned old/new canonical primary rows, the number of changed defined values was:

| Field | Changed rows | Maximum absolute change |
|---|---:|---:|
| threshold | 7 | 0.056543 |
| PFA | 8 | 0.011111 |
| Sens3 | 3 | 0.033333 |
| Sens0 | 2 | 0.029412 |
| premature | 8 | 0.033333 |
| median lead | 3 | 1 hour |
| standardized PPV3 | 10 | 0.100000 |
| standardized PPV0 | 11 | 0.042857 |
| alert burden | 76 | 3.082371 per 100 patient-days |

Threshold/alert changes in Experiment II are expected from restoring t=4,5 score history. E3 local changes also reflect the conditional target reservoir. Alert burden changes broadly because Equation (48) affects every method with unequal class exposures. Evaluability/PPV corrections account for the relevant sensitivity and PPV differences. Undefined values were not converted to zero.

The corrected production-size operating results are:

| Scenario/method | Threshold | PFA | Sens3 | Sens0 |
|---|---:|---:|---:|---:|
| E1 PaFAR-F | 1.004066 | 0.106746 | 0.033113 | 0.079470 |
| E1 PaFAR-T | 2.955814 | 0.111935 | 0.033113 | 0.052980 |
| E1 PaFAR-HC | 1.041735 | 0.086731 | 0.039735 | 0.079470 |
| E2 PaFAR-F | 1.053763 | 0.117647 | 0.022989 | 0.045977 |
| E2 PaFAR-T | 2.981718 | 0.122172 | 0.022989 | 0.068966 |
| E2 PaFAR-HC | 1.095309 | 0.108597 | 0.017241 | 0.040230 |
| E3 Direct | 0.646199 | 0.136013 | 0.059783 | 0.168478 |
| E3 source PaFAR-T | 3.595602 | 0.121145 | 0.016304 | 0.130435 |
| E3 source PaFAR-HC | 0.709901 | 0.091410 | 0.032609 | 0.135870 |
| E3 local PaFAR-F, m0=500 | 0.687717 | 0.106828 | 0.043478 | 0.146739 |
| E3 local PaFAR-T, m0=500 | 3.610809 | 0.117291 | 0.016304 | 0.125000 |
| E3 local PaFAR-HC, m0=500 | 0.737663 | 0.070485 | 0.032609 | 0.125000 |

These are diagnostic single replicates and are excluded from manuscript tables.

## 7. Is threshold arithmetic still completely correct?

Yes. The independent implementation still matches all **16/16** audited thresholds:

- PaFAR and HC integer indices match exactly;
- finite/infinite status matches exactly;
- threshold maximum absolute error is zero;
- strict exceedance counts match exactly.

The alpha-grid check covers 101 replicate/method groups. Every group contains all five alpha values, one seed record, one learner iteration value, one template, and nonincreasing PaFAR thresholds as alpha increases.

## 8. New pilot best iterations

- E1: **38**
- E2: **54**
- E3: **23**

They are unchanged from the earlier production-size pilots. All learners succeeded.

## 9. Pilot score resolution, PFA/Sens3, time, and peak RSS

| Scenario | Calibration fixed/time maxima | Test fixed/time maxima | Total time | Peak RSS |
|---|---|---|---:|---:|
| E1 | 553/553 unique; 553/553 unique | 1349/1349; 1349/1349 | 40.848 s | 1.561 GiB |
| E2 | 551/551 unique; 551/551 unique | 1326/1326; 1326/1326 | 44.133 s | 1.909 GiB |
| E3 | 542/542 unique; 542/542 unique | 1816/1816; 1816/1816 | 44.353 s | 2.119 GiB |

The largest maxima atom is one patient in every listed distribution. Production-size E2/E3 therefore show no smoke-style endpoint collapse. Method-specific PFA and Sens3 are listed in the preceding table and the complete pilot checkpoint.

## 10. Recommended n_jobs

Use **`n_jobs=4`** on this approximately 15.69-GiB host. The largest observed process RSS was 2.119 GiB; four simultaneous peaks imply roughly 8.48 GiB before parent/OS/cache overhead. `n_jobs=2` is the conservative shared-host setting. Eight simultaneous peaks would already imply about 16.95 GiB and are unsafe.

Using measured one-replicate times and E1=100, E2=100, E3=50 gives an idealized 2.977 serial hours, 1.488 hours at two workers, 0.744 hours at four, and 0.372 hours at eight. These are arithmetic lower bounds, not measured parallel runtimes.

## 11. Production status

**The formal production simulation is still not run.** `outputs/production/` contains only its pre-existing `.gitkeep`; it has zero production result files. The production orchestrator was executed only in check-only mode and explicitly reported that no oracle or production simulation ran.

## Aggregation, table, figure, and safety corrections

- Raw rows save all Equation (47)--(48) composition quantities and prevalence-0.05 reweights.
- `summary_long.csv` includes defined/undefined counts, failure/infinite-threshold frequencies, alert episodes, component means/rates, and conditional-PFA exceedance proportions.
- Thresholds use `threshold_mean` and `threshold_sd`; SD is not labeled MCSE.
- `learner_metrics_summary.csv` is generated from one independent learner row per replicate.
- S4 `Direct source transfer, target_m0=0` is canonical; it is not discarded as an alias.
- Tables 3--5 are built from aggregate summaries and contain no replicate column.
- Figures 2--6 do not average unexplained scenario mixtures; their input CSV and JSON sidecars record filters and scenarios.
- Effective config checksums include condition and master-seed overrides.
- Checkpoint sidecars include the implementation checksum.
- Oracle filename/metadata includes scenario, site, tmin, hmax, smoothing length, master seed, and Nref; stale or checksum-mismatched files are rejected.
- `scripts/production_orchestrator.py` serializes oracle preparation before parallel replicates and requires an explicit production confirmation token.

## Validation sequence

1. Full pytest: **49 passed, 0 failed**.
2. Final cold-start smoke: **511 rows, 16/16 checks, 0 learner failures**, 88.25 seconds.
3. Preproduction audit: 96 score-distribution rows; threshold mismatches 0/16; pre-burn, conditional reservoir, metric formula, and alpha-grid checks all passed.
4. Production-size one-replicate pilots: E1/E2/E3 all succeeded.
5. Resume check: completed checkpoint mtimes remained unchanged; 16/16 checks passed.
6. Table/figure dry run: completed successfully; all Table 3--5 files contain aggregate rows and no `replicate` column.

## Modified and added files

Configuration:

- `configs/smoke.yaml`
- `configs/exp1_primary.yaml`
- `configs/exp2_primary.yaml`
- `configs/exp1_sensitivity.yaml`
- `configs/exp2_sensitivity.yaml`

Implementation:

- `src/pafar_sim/config.py`
- `src/pafar_sim/rng.py`
- `src/pafar_sim/io_utils.py`
- `src/pafar_sim/metrics.py`
- `src/pafar_sim/aggregation.py`
- `src/pafar_sim/plotting.py`
- `src/pafar_sim/diagnostics.py`
- `src/pafar_sim/exp1/oracle.py`
- `src/pafar_sim/exp1/runner.py`
- `src/pafar_sim/exp2/dgp.py`
- `src/pafar_sim/exp2/features.py`
- `src/pafar_sim/exp2/runner.py`

Scripts:

- `scripts/run_smoke.py`
- `scripts/run_preproduction_audit.py`
- `scripts/run_production_pilot.py`
- `scripts/production_orchestrator.py`

Tests:

- `tests/test_metrics.py`
- `tests/test_config.py`
- `tests/test_output_schema.py`
- `tests/test_pipeline_integration.py`
- `tests/test_preburn_history.py`
- `tests/test_conditional_reservoir.py`
- `tests/test_alpha_aggregation.py`
- `tests/test_oracle_safety.py`

Reports:

- `docs/METRIC_FORMULA_AUDIT.md`
- `docs/STATISTICAL_IMPLEMENTATION_FIX_REPORT.md`

Required/new evidence:

- `outputs/diagnostics/preburn_score_history_check.csv`
- `outputs/diagnostics/conditional_non_event_reservoir_check.csv`
- `outputs/diagnostics/metric_formula_regression_check.csv`
- `outputs/diagnostics/alpha_grid_check.csv`
- `outputs/production_pilot/production_one_rep_timing_after_fix.csv`
- `outputs/production_pilot/production_one_rep_results_after_fix.csv.gz`
- `outputs/production_pilot/production_one_rep_score_diagnostics_after_fix.csv`
- regenerated `outputs/smoke/` aggregate/table/figure/checkpoint artifacts
- `outputs/table_figure_dry_run/` aggregate/table/figure dry-run artifacts

`PaFAR.pdf` and `v40i08.pdf` were not modified. `outputs/production/` remains empty apart from `.gitkeep`.
