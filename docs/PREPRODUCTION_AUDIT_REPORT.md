# Pre-production diagnostic audit

## Scope and guardrails

The audit began from the existing smoke checkpoints. It did not modify `PaFAR.pdf` or `v40i08.pdf`, the statistical methods, DGP, signal strength, strict `>` alert rule, primary smoothing setting, or primary XGBoost hyperparameters. No jitter, random tie breaking, extra boosting rounds, or post-hoc tuning was introduced. The only production-size runs were one diagnostic replicate each for E1, E2, and E3, stored under `outputs/production_pilot/`; no formal production replicate was run.

## Required findings

### 1. Were code errors found?

Yes, three engineering/output defects were found. None was an error in the statistical threshold formulas or the in-memory simulation results.

### 2. Location, impact, and correction

1. **CSV floating-point readback:** pandas' default parser read some upper-endpoint thresholds one ULP below the exact value written to the checkpoint. For example, E2's stored double is `0x1.58a6850cd1a90p-4`, while default readback could produce the adjacent lower double. This created false exceedances in a checkpoint-based independent audit, even though the original raw CSV and in-memory calibration were correct. Checkpoint collection, plotting, diagnostics, and smoke learner aggregation now use `float_precision="round_trip"`. A regression test writes an adjacent-double-sensitive threshold and requires exact recovery.
2. **Output schema:** E3 direct transfer did not consistently use `target_m0=0`; method metadata did not separate method family, deployment strategy, threshold scale, threshold origin, and template origin; S4's direct-transfer alias could be counted twice; and threshold-free learner metrics were repeated on method rows. A centralized output schema now supplies the required fields plus `is_alias`/`alias_of`; aggregation excludes aliases; threshold-method AUC/PR fields are empty; and learner metrics are emitted once per `scenario × replicate × site × score_generator` in a separate `learner_metrics.csv`. Dedicated schema tests cover these invariants.
3. **64-bit seed persistence:** an unsigned 64-bit replicate seed could be inferred as a floating number when CSV was read, losing exact identity. Replicate seeds are now serialized as `u64:<decimal>` strings, with an exact round-trip regression test.

The first defect affected downstream checkpoint interpretation and could make a re-audit report false exceedances; it did not change already computed PFA/sensitivity. The latter two affected provenance, aggregation semantics, and reproducibility metadata. The before/after statistical comparison confirms that none of the corrections changed the statistical outputs.

### 3. Pytest before and after

- Before audit fixes: **33 passed, 0 failed**.
- After fixes and added regressions: **37 passed, 0 failed**. The cold-smoke preflight recorded **2.83 s**; the final standalone verification run also passed all 37 tests in **30.05 s** (wall time varies with process/cache state).

The new tests cover exact threshold float round-trip, unsigned seed round-trip, required schema fields/values, alias exclusion, and uniqueness of threshold-free learner metrics.

### 4. Cold-start smoke before and after

- Before: 109 raw method rows; all **16/16** engineering checks passed; approximately **62.08 s**.
- After: 109 raw method rows; all **16/16** engineering checks passed; **61.86 s**; no learner failures; peak Python allocation diagnostic 183.8 MiB.
- Exact comparison: threshold, threshold index, `m0`, `alpha_m0`, PFA, long-stay PFA, Sens3, Sens0, premature-alert rate, median lead, both standardized PPVs, and alert burden are all exactly equal over all 109 rows; maximum absolute difference is **0** for every field.

The after-smoke contains three infinite thresholds and 39 undefined selected metrics. These are expected finite-sample/definition outcomes already handled by the pipeline, not crashes or silent substitutions.

### 5. Exact cause of E2/E3 threshold collapse

The smoke learners for E2 and E3 stop at `best_iteration=0`, use one tree, and produce only eight distinct raw probabilities. Causal smoothing raises the number of distinct values to 110 and 111, but a large fraction of patient maxima still lands on the learner's exact upper endpoint.

- E2 Naive maximum, PaFAR-F (`k=145`), and PaFAR-HC (`k=151`) all select the same endpoint double, `0x1.58a6850cd1a90p-4` (`0.08414318058667392`). Their different quantile/order-statistic locations fall inside the same 25-observation endpoint atom.
- E3 local PaFAR-F for target `m0=25,50,100` selects the same endpoint double, `0x1.0d7ea08f3b5e0p-4` (`0.06579458921961345`). The relevant nested-reservoir order statistics all fall inside the endpoint atom: 16/25, 29/50, and 66/100 values equal it.
- These values are exactly equal in double precision; the equality is not caused by six-decimal printing.
- PaFAR-T in E2 uses a different standardized threshold (`1.2840697423575651`) with 12 calibration maxima above it, so it retains nonzero alerts. In E3, its threshold (`1.3943648883275972`) is itself the standardized-score endpoint, leaving no maxima above it and therefore zero strict alerts.

The collapse is thus score discreteness caused by the configured learner stopping after one tree in the small smoke sample, not a defect in the threshold implementation.

### 6. Largest calibration-maxima atom fractions

| Scenario / calibration source | Fixed maximum | Time-standardized maximum |
|---|---:|---:|
| E1 source non-events | 1/159 = 0.006289 | 1/159 = 0.006289 |
| E2 source non-events | 25/160 = **0.156250** | 3/160 = 0.018750 |
| E3 source non-events | 83/160 = **0.518750** | 74/160 = **0.462500** |
| E3 target reservoir | 66/100 = **0.660000** | 63/100 = **0.630000** |

For E2 PaFAR-T specifically, seven calibration maxima equal its interior threshold and 12 are greater; its threshold atom is not the largest time-standardized atom.

### 7. Role of strict `>` in zero PFA/sensitivity

Strict `>` is the direct alert-count mechanism: when the selected threshold equals the maximum attainable score, no observation can exceed it. It is not the cause of score collapse, but it converts the endpoint tie into zero alerts exactly as the prespecified rule requires.

- E2 fixed methods: strict `>` gives 0 test alerts; diagnostic-only `>=` would give 55 (47 non-events and 8 events).
- E3 fixed endpoint methods: strict `>` gives 0; diagnostic-only `>=` would give 170 (149 non-events and 21 events).
- E3 PaFAR-T endpoint methods: strict `>` gives 0; diagnostic-only `>=` would give 159 (140 non-events and 19 events).
- E2 PaFAR-T is not at the attainable endpoint: strict `>` gives 16 test alerts and hence nonzero PFA.

The `>=` counts are explanatory counterfactuals only. No formal result uses `>=`, and the strict rule was not changed.

### 8. Did the production-size pilots retain threshold collapse?

No. Under the formal sample sizes, 400-round cap, and 25-round early stopping, E1/E2/E3 selected best iterations 38/54/23. E1 and E2 fixed/time calibration maxima were all unique. E3 fixed source calibration maxima had 541 unique values among 542 patients (largest atom 2/542 = 0.003690), while its time maxima were all unique. E3 test fixed maxima had only a two-patient atom among 1,816 (0.001101), and time maxima were all unique.

All pilot thresholds are distinct where the strategies differ, all reported PFA/Sens3/Sens0 values are finite, and no learner failed. The smoke endpoint collapse therefore did not persist in these three one-replicate production-size diagnostics. A single replicate is diagnostic evidence, not a manuscript estimate.

### 9. Measured one-replicate wall time and native peak RSS

| Scenario | DGP | Features | Fit | Calibration | Metrics | Write | Total | Peak RSS | Best iteration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.327 s | 18.191 s | 12.275 s | 0.200 s | 0.065 s | 0.008 s | **31.853 s** | **1.222 GiB** | 38 |
| E2 | 0.548 s | 18.307 s | 14.216 s | 0.208 s | 0.071 s | 0.008 s | **34.018 s** | **1.556 GiB** | 54 |
| E3 | 0.676 s | 20.836 s | 9.913 s | 0.165 s | 0.176 s | 0.008 s | **32.503 s** | **1.759 GiB** | 23 |

RSS was sampled with `psutil` and includes native allocations; it is not a `tracemalloc` estimate. Each pilot used `n_jobs=1` and the requested formal sizes/settings.

### 10. Production runtime estimates

Using the measured scenario times and the planned primary counts E1=100, E2=100, E3=50 gives **8,212.25 s = 2.281 h** of serial work.

| Execution | Idealized elapsed time |
|---|---:|
| Serial | 2.281 h |
| `n_jobs=2` | 1.141 h |
| `n_jobs=4` | 0.570 h |
| `n_jobs=8` | 0.285 h |

These are arithmetic lower-bound estimates assuming perfect scheduling and no memory-bandwidth, serialization, startup, or I/O penalty. They are not measured parallel runtimes.

### 11. Recommended production parallelism

Use **`n_jobs=4`** as the default on this host. The machine has about 15.69 GiB physical RAM, while the largest measured single-worker peak is 1.759 GiB. Four workers leave useful margin for the parent process, filesystem cache, transient overlap, and the OS. `n_jobs=2` is the conservative fallback if other memory-heavy work shares the host. `n_jobs=8` could nominally consume about 14.1 GiB from worker peaks alone and is not recommended without a measured parallel memory test.

### 12. Formal production status

**The formal 500/250/100/50-replicate production simulation has not been run.** `outputs/production/` was not populated by this audit. The files in `outputs/production_pilot/` are isolated one-replicate diagnostics and must not be used in manuscript tables.

## Independent threshold audit

The independent implementation does not call the production threshold functions. It independently recomputes NumPy type-7 naive quantiles, marginal PaFAR index/threshold, PaFAR-HC index/threshold, strict exceedance counts, and `alpha_m0`. After exact round-trip checkpoint reading, all 16 audited methods satisfy:

- integer index exact match;
- finite/infinite status exact match;
- threshold exact equality (maximum absolute error 0);
- strict exceedance-count exact match.

There is therefore no evidence of a statistical calibration-code error.

## Added or modified files

Source and scripts:

- `src/pafar_sim/diagnostics.py` — deterministic checkpoint reconstruction, distribution/tie/learner diagnostics, independent threshold audit, and figures.
- `src/pafar_sim/output_schema.py` — centralized method metadata and alias semantics.
- `src/pafar_sim/aggregation.py` — round-trip float loading and alias exclusion.
- `src/pafar_sim/plotting.py` — round-trip float loading.
- `src/pafar_sim/exp1/runner.py` — schema application, separate learner metrics, exact seed serialization.
- `src/pafar_sim/exp2/runner.py` — schema application, E3 `target_m0`, alias handling, separate learner metrics, exact seed serialization.
- `scripts/run_preproduction_audit.py` — diagnostic audit entry point.
- `scripts/run_production_pilot.py` — isolated one-replicate production-size pilot with native RSS polling.
- `scripts/run_smoke.py` — learner-metric aggregation.
- `pyproject.toml` and `requirements.txt` — add `psutil` for native RSS measurement.

Tests and reports:

- `tests/test_pipeline_integration.py` — threshold and uint64 seed round-trip regressions.
- `tests/test_output_schema.py` — schema, E3 target size, alias, and learner-metric tests.
- `docs/LEARNER_DIAGNOSTIC_REPORT.md` — learner audit.
- `docs/PREPRODUCTION_AUDIT_REPORT.md` — this report.

Generated diagnostic evidence:

- `outputs/diagnostics/score_distribution_diagnostics.csv`
- `outputs/diagnostics/calibration_maxima_top_counts.csv`
- `outputs/diagnostics/threshold_tie_audit.csv`
- `outputs/diagnostics/independent_threshold_audit.csv`
- `outputs/diagnostics/learner_diagnostics.csv`
- `outputs/diagnostics/smoke_before_schema_fix.csv.gz`
- `outputs/diagnostics/smoke_after_schema_fix.csv.gz`
- `outputs/diagnostics/smoke_before_after_comparison.csv`
- `outputs/diagnostics/figures/e1_fixed_maxima_diagnostics.png` and `.json`
- `outputs/diagnostics/figures/e1_time_maxima_diagnostics.png` and `.json`
- `outputs/diagnostics/figures/e2_fixed_maxima_diagnostics.png` and `.json`
- `outputs/diagnostics/figures/e2_time_maxima_diagnostics.png` and `.json`
- `outputs/diagnostics/figures/e3_fixed_maxima_diagnostics.png` and `.json`
- `outputs/diagnostics/figures/e3_time_maxima_diagnostics.png` and `.json`
- `outputs/production_pilot/production_one_rep_timing.csv`
- `outputs/production_pilot/production_one_rep_results.csv.gz`
- `outputs/production_pilot/production_one_rep_score_diagnostics.csv`
- `outputs/production_pilot/pilot_manifest.json`

The cold-smoke outputs were regenerated under `outputs/smoke/`, including the separate aggregated `outputs/smoke/learner_metrics.csv`.
