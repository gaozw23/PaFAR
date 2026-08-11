# Primary Results Manuscript Revision Report

Finalized: 2026-08-03 (Asia/Shanghai)

## 1. Scope and completion status

The locked `condition=primary`, `is_alias=False` production results were postprocessed into manuscript-ready Tables 3--5, Figures 2--6, supplementary audit tables, and paired-replicate contrasts. `literature/PaFAR.tex` was revised in place and compiled successfully as a 44-page PDF. No simulation was rerun. No file under `outputs/production/` was written, replaced, touched, or timestamp-modified.

## 2. Manuscript checksum

- Before SHA-256, `literature/PaFAR.tex`: `b2a352df06faab768f9036e80ba89f4b3d9c96c2fa5014103a29d1d7c57157f3`
- After SHA-256, `literature/PaFAR.tex`: `8316f1cec629a514c575a7604dfc5099eeea03a0dd7cc0a62a497228c6ebce9b`
- Bibliography before and after: `d857eb33093b8cd9d63205f561aed17de210cefe0de9fc5728672027977f8113` (unchanged)
- Full source diff: `literature/PaFAR_primary_results.diff`

## 3. Locked-production checksum proof

| Locked file | Before SHA-256 | After SHA-256 | Result |
|---|---|---|---|
| `PRODUCTION_LOCK.json` | `ebe39073bb8fcfd0bd416307cc3f93e4730514180701524c45ddaf768ee10f86` | same | unchanged |
| `all_replicate_results.csv.gz` | `93f01b67fdd6277ccac5e9876137ed6468356a983c466a95d78a55f3dfeefcf2` | same | unchanged |
| `summary_long.csv` | `f0a076fdebf4e04b801f7a4bc1da0e76d2f274808a2cac40c63261da3f9a7d30` | same | unchanged |
| `learner_metrics.csv` | `9e5613f66e2e3ee3cbea414c147fa76d6ddf9a59118ce03be7cddde5a6f43bfa` | same | unchanged |

The recursive `outputs/production/raw/` audit also matched before and after: 2,250 checkpoint files, 2,250 JSON sidecars, 34,165,024 total bytes, and latest modification time `1785754970663634200` ns since the Unix epoch (`2026-08-03T11:02:50.663634Z`). This independently verifies that raw production output was not modified.

## 4. Table outputs and row counts

| Manuscript item | Rows | CSV | LaTeX |
|---|---:|---|---|
| Table 3, Experiment I | 12 | `literature/generated_tables/table3_experiment1.csv` | `literature/generated_tables/table3_experiment1.tex` |
| Table 4, Experiment II | 8 | `literature/generated_tables/table4_experiment2.csv` | `literature/generated_tables/table4_experiment2.tex` |
| Table 5, target shift | 12 | `literature/generated_tables/table5_target_shift.csv` | `literature/generated_tables/table5_target_shift.tex` |
| Table S1, learner metrics | 3 | `literature/supplementary/tableS1_learner_metrics.csv` | `literature/supplementary/tableS1_learner_metrics.tex` |
| Table S2, HC exceedance | 5 | `literature/supplementary/tableS2_hc_exceedance.csv` | `literature/supplementary/tableS2_hc_exceedance.tex` |
| Table S3, alpha grid | 50 | `literature/supplementary/tableS3_alpha_grid.csv` | `literature/supplementary/tableS3_alpha_grid.tex` |
| Table S4, definedness | 7 | `literature/supplementary/tableS4_definedness.csv` | `literature/supplementary/tableS4_definedness.tex` |
| Paired contrasts | 96 | `literature/supplementary/paired_contrasts.csv` | `literature/supplementary/paired_contrasts.tex` |

Structural missingness is retained as missing/`null`, never converted to zero. Main-table cells contain no placeholder macro or dash-only numerical substitute.

## 5. Figure outputs

- Figure 2: `literature/figures/figure2_cumulative_false_alert.{pdf,png,csv,json}`
- Figure 3: `literature/figures/figure3_pfa_reliability.{pdf,png,csv,json}`
- Figure 4: `literature/figures/figure4_sens3_vs_pfa.{pdf,png,csv,json}`
- Figure 5: `literature/figures/figure5_conditional_pfa.{pdf,png,csv,json}`
- Figure 6: `literature/figures/figure6_target_recalibration.{pdf,png,csv,json}`

The target-recalibration display uses target `m0=500` as specified. The S4 local HC audit frequency of 0.038 in Table S2 aggregates the prespecified `m0=100,250,500` calibration prefixes (1,500 replicate rows); it is intentionally distinct from the `m0=500`-only value shown in Figure 5.

## 6. Main numerical findings

- At nominal alpha 0.10, PaFAR-F patient-level PFA was 0.099 in S1, 0.099 in S2, and 0.101 in S3. Experiment II gave 0.098 in E1 and 0.101 in E2.
- Pointwise-alpha thresholds accumulated severe patient-level false-alarm risk: PFA 0.431, 0.474, and 0.481 in S1--S3, and 0.615 and 0.595 in E1--E2.
- Under unequal monitoring length in S3, PaFAR-T lowered long-stay PFA relative to PaFAR-F by 0.0348 (paired Monte Carlo SE 0.0008; Monte Carlo interval `[-0.0363,-0.0333]`) while changing overall PFA by only 0.0002 and Sens3 by 0.0046.
- Fixed 0.5 thresholds had poor patient-level PFA in E1/E2 (0.751/0.733). Youden had 0.876/0.868. PaFAR-F reduced these to 0.098/0.101, with mean Sens3 0.035/0.038.
- Learner discrimination was stable: mean AUROC 0.716, 0.707, and 0.723 and trapezoidal PR-AUC 0.097, 0.096, and 0.105 in E1--E3; learner-failure frequency was zero.
- Direct source-threshold transfer failed under both site shifts: target PFA 0.280 in S4 and 0.136 in E3. Re-estimating only the scalar PaFAR-F threshold with 500 target non-events reduced PFA to 0.100 in both experiments. Paired reductions were 0.1802 (MCSE 0.0013) in S4 and 0.0363 (MCSE 0.0035) in E3.
- PaFAR-HC conditional exceedance frequencies were 0.040, 0.032, 0.048, and 0.038 in exchangeable S1, S2, S3, and local S4 audits, near the prespecified delta 0.05. Transferred source PaFAR-HC in S4 exceeded in every replicate, as expected outside the exchangeability guarantee.

## 7. Abstract revision

The abstract now reports the 2,250-replicate primary simulation, finite-sample PFA behavior, pointwise-alpha failure, site-shift transfer failure and local recalibration, the PaFAR-HC reliability/efficiency tradeoff, and the explicitly planned status of the PhysioNet analysis. It does not claim clinical validation.

## 8. Section 6.6 revision

Section 6.6 is now a completed `Primary simulation results` section. It integrates operating characteristics, end-to-end irregular-EHR performance, alpha-grid reliability, high-confidence calibration, target-site recalibration, and definedness/learner-stability findings, with Tables 3--5 and Figures 2--6 placed near the relevant text.

## 9. Discussion and conclusion revision

The discussion and conclusion now distinguish finite-sample patient-level control from discrimination, describe PaFAR-T as a reliability/efficiency tradeoff rather than a uniform improvement, document failure of direct threshold transfer, and avoid equating paired Monte Carlo intervals with patient-level confidence intervals or hypothesis tests. Clinical efficacy and real-data performance are not asserted.

## 10. Remaining planned work

Only two substantive empirical items remain explicitly planned: the PhysioNet 2019 analysis and the one-factor sensitivity analyses that were not part of the locked primary production run. Tables 6--7 and the real-data figure placeholders are retained solely for the planned PhysioNet analysis.

## 11. Compilation and visual QA

- Baseline command: `latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=PaFAR_original_before_primary PaFAR.tex`
- Final command: `latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=PaFAR_results_filled PaFAR.tex`
- Final PDF: `literature/PaFAR_results_filled.pdf`
- Final PDF SHA-256: `200cf8d8280aae7eb87f38b00f5a50f2249648d7567baac80ee0505274322873`
- Final page count: 44
- LaTeX errors: 0
- Undefined controls/citations/references: 0/0/0
- Duplicate labels: 0
- Overfull/underfull boxes: 0/0
- Other LaTeX, package, or pdfTeX warnings: 0

All 44 pages were rendered and reviewed. The abstract, Tables 3--5, Figures 2--6, supplementary tables, proofs, and bibliography were additionally inspected at page level; no clipping, overlap, blank page, or displaced end-of-document float was found.

## 12. Automated tests

Command: `.\\.venv\\Scripts\\python.exe -m pytest -q`

Result: **55 passed in 6.79 seconds**. Tests cover primary filters, table keys and dimensions, figure filters, stale placeholder language, main-table placeholder exclusion, production hashes/counts/timestamps, manuscript modification, and clean PDF/log output.

## 13. Files created or modified

Core implementation and audit files:

- Modified: `literature/PaFAR.tex`
- Added: `scripts/build_primary_manuscript_outputs.py`
- Added: `tests/test_manuscript_outputs.py`
- Added/updated: `literature/manuscript_data_manifest.json`
- Added: `literature/PaFAR_primary_results.diff`
- Added: `literature/PRIMARY_RESULTS_REVISION_REPORT.md`
- Added: `literature/PaFAR_results_filled.pdf` and standard LaTeX build byproducts for job `PaFAR_results_filled`
- Added backups: `literature/backups/PaFAR_before_primary_results_20260803T131112Z.tex` and `literature/backups/PaFAR_references_before_primary_results_20260803T131112Z.bib`
- Added baseline build: `literature/PaFAR_original_before_primary.pdf` and standard LaTeX build byproducts for job `PaFAR_original_before_primary`

Generated manuscript data files:

- `literature/generated_tables/table3_experiment1.{csv,tex}`
- `literature/generated_tables/table4_experiment2.{csv,tex}`
- `literature/generated_tables/table5_target_shift.{csv,tex}`
- `literature/figures/figure2_cumulative_false_alert.{csv,json,pdf,png}`
- `literature/figures/figure3_pfa_reliability.{csv,json,pdf,png}`
- `literature/figures/figure4_sens3_vs_pfa.{csv,json,pdf,png}`
- `literature/figures/figure5_conditional_pfa.{csv,json,pdf,png}`
- `literature/figures/figure6_target_recalibration.{csv,json,pdf,png}`
- `literature/supplementary/tableS1_learner_metrics.{csv,tex}`
- `literature/supplementary/tableS2_hc_exceedance.{csv,tex}`
- `literature/supplementary/tableS3_alpha_grid.{csv,tex}`
- `literature/supplementary/tableS4_definedness.{csv,tex}`
- `literature/supplementary/paired_contrasts.{csv,tex}`

The same tables, figures, supplementary artifacts, and `primary_numerical_summary.json` were mirrored under `outputs/manuscript_primary_final/`. Temporary rendered QA pages are under `tmp/pdfs/final/` and are not manuscript inputs.

## 14. Exceptions encountered and resolved

- The first baseline TeX invocation could not write the user-level font cache inside the sandbox; the same compilation was rerun with the required permission and succeeded. This was an environment permission issue, not a manuscript error.
- Matplotlib initially selected an interactive backend; the postprocessor was made deterministic with the noninteractive `Agg` backend and a project-local config directory.
- Structural missing values in JSON were explicitly converted to `null`.
- An initial three-part-table layout caused an overfull Table 5; the generated table wrapper was changed to a width-bounded layout. The final log has zero overfull or underfull boxes.
- The S4 local high-confidence audit denominator was checked against the locked design and correctly uses all three target calibration prefixes for the reported 0.038 aggregate frequency.

No unresolved production, numerical, compilation, or layout exception remains.

## 15. Explicit non-rerun statement

No formal or smoke simulation command was executed during this manuscript pass. The postprocessor only read locked production CSV/JSON artifacts; it did not import or invoke a simulation runner. Source methods, DGPs, signal settings, method definitions, and production configurations were not changed.
