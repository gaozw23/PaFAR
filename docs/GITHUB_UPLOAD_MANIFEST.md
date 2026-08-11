# GitHub upload manifest

Generated for the private `PaFAR` repository preparation on 2026-08-12. The candidate set is restricted to the explicitly approved root files and the `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, `literature/`, `data/README.md`, and `outputs/README.md` paths after applying `.gitignore`.

## Candidate summary

- Candidate file count: 199
- Candidate total size: 4,563,413 bytes (4.352 MiB)
- Files over 10 MiB: none
- Files over 50 MiB: none

### File counts by directory

| Directory | Files | Size (MiB) |
| --- | ---: | ---: |
| Root | 6 | 0.746 |
| `configs/` | 7 | 0.004 |
| `data/` | 1 | <0.001 |
| `docs/` | 13 | 0.054 |
| `literature/` | 64 | 3.072 |
| `outputs/` | 1 | <0.001 |
| `scripts/` | 23 | 0.075 |
| `src/` | 49 | 0.351 |
| `tests/` | 35 | 0.048 |

## Excluded categories

- Python environments and caches: `.venv/`, cache directories, bytecode.
- Raw and processed clinical data: everything under `data/` except `data/README.md`, including PhysioNet PSV files, manifests, feature caches, and the locally downloaded official scorer.
- Generated run artifacts: everything under `outputs/` except `outputs/README.md`, including simulation raw outputs, real-data outputs, checkpoints, models, bootstrap objects, logs, and failure caches.
- Archives and temporary files: `archives/`, `tmp/`, compressed archives, and manuscript backup directories.
- Credentials and secrets: `.env` files, credential/secret/token files, private-key formats, and editor-local configuration.
- LaTeX build products and superseded manuscript renderings.
- Downloaded third-party reference `v40i08.pdf`.

## Largest 30 candidate files

| Rank | Path | Bytes | MiB |
| ---: | --- | ---: | ---: |
| 1 | `PaFAR.pdf` | 775612 | 0.740 |
| 2 | `literature/PaFAR.pdf` | 775612 | 0.740 |
| 3 | `literature/figures/figure3_pfa_reliability.png` | 240610 | 0.229 |
| 4 | `literature/figures/figure5_conditional_pfa.csv` | 227436 | 0.217 |
| 5 | `literature/figures/figure2_cumulative_false_alert.png` | 212564 | 0.203 |
| 6 | `literature/figures/figure8_realdata_false_alert.csv` | 199414 | 0.190 |
| 7 | `literature/figures/figure10_cross_hospital.png` | 196362 | 0.187 |
| 8 | `literature/figures/figure6_target_recalibration.png` | 177387 | 0.169 |
| 9 | `literature/figures/figure8_realdata_false_alert.png` | 165381 | 0.158 |
| 10 | `literature/figures/figure7_cohort_flow.png` | 131606 | 0.126 |
| 11 | `literature/PaFAR.tex` | 110473 | 0.105 |
| 12 | `literature/figures/figure2_cumulative_false_alert.csv` | 100585 | 0.096 |
| 13 | `literature/figures/figure4_sens3_vs_pfa.png` | 100166 | 0.096 |
| 14 | `literature/figures/figure5_conditional_pfa.png` | 94284 | 0.090 |
| 15 | `literature/figures/figure9_realdata_tradeoff.png` | 82625 | 0.079 |
| 16 | `literature/PaFAR_primary_results.diff` | 57831 | 0.055 |
| 17 | `literature/figures/figure8_realdata_false_alert.pdf` | 45629 | 0.044 |
| 18 | `scripts/build_primary_manuscript_outputs.py` | 40019 | 0.038 |
| 19 | `literature/figures/figure2_cumulative_false_alert.pdf` | 36983 | 0.035 |
| 20 | `src/pafar_sim/diagnostics.py` | 35035 | 0.033 |
| 21 | `literature/figures/figure3_pfa_reliability.pdf` | 31007 | 0.030 |
| 22 | `src/pafar_sim/realdata/feature_cache.py` | 25723 | 0.025 |
| 23 | `literature/figures/figure10_cross_hospital.pdf` | 23268 | 0.022 |
| 24 | `literature/figures/figure6_target_recalibration.pdf` | 22981 | 0.022 |
| 25 | `src/pafar_sim/exp2/runner.py` | 19978 | 0.019 |
| 26 | `src/pafar_sim/production.py` | 19598 | 0.019 |
| 27 | `literature/figures/figure4_sens3_vs_pfa.pdf` | 18844 | 0.018 |
| 28 | `literature/figures/figure7_cohort_flow.pdf` | 18729 | 0.018 |
| 29 | `literature/figures/figure5_conditional_pfa.pdf` | 17746 | 0.017 |
| 30 | `literature/figures/figure9_realdata_tradeoff.pdf` | 17565 | 0.017 |

## Audit results

- Forbidden-path audit: passed; no candidate path matched the prohibited data, output, cache, model, checkpoint, archive, secret, or large-object patterns.
- Secret-pattern audit: passed; no candidate text matched the configured token, credential, private-key, authorization-header, or secret-assignment patterns.
- Local absolute-path audit: passed after manual review. One `literature/PaFAR.tex` match was the LaTeX expression `u:\widehat`, not a filesystem path. No source module or formal script depends on a local absolute path.
- Patient-identifier audit: passed after manual review. Two pattern-shaped strings in `tests/realdata/test_raw_schema.py` are expected PhysioNet record filenames used by a schema test, not a patient-ID list; no patient contents are included.
- Third-party-file audit: passed. `v40i08.pdf` is ignored. Raw PhysioNet data and the local official scorer are ignored under `data/`; temporary scorer downloads are ignored under `tmp/`. Candidate code contains only an adapter that validates and dynamically loads a separately obtained local scorer. No candidate third-party source file was found.

## Hard-stop assessment

No candidate file exceeds 50 MiB, no forbidden data/cache/model artifact is in the candidate set, and no unresolved sensitive or third-party file was detected. The candidate set is eligible for testing and explicit staging.
