# Implementation report

Implemented on 2026-08-03 against the complete relevant pages of `PaFAR.pdf` and the engineering guidance in `v40i08.pdf`.

## Delivered scope

- Experiment I S1-S4, including class-specific score DGPs, independent chunked Oracle-F references, conditional deployment PFA, source transfer, and nested target recalibration.
- Experiment II E1-E3, including latent severity, 12 irregular biomarkers, informative observation, causal window features, training-frozen imputation/schema, native XGBoost early stopping, and target transfer/recalibration.
- Fixed 0.5, Youden, pointwise-alpha, binwise Bonferroni conformal, naive maximum, PaFAR-F, PaFAR-T, PaFAR-HC, and Oracle-F where prescribed.
- All requested patient-level estimands, denominator counts, threshold metadata, seed/config/PDF checksums, failure retention, timing, atomic checkpoints, parallel execution, and resume.
- Production primary and one-factor sensitivity configurations. Named sensitivity conditions use independent deterministic seed streams and separate checkpoint paths.
- Saved-output aggregation, Tables 3-5 extracts, Figures 2-6 with JSON sidecars, unit tests, smoke checks, and benchmark.

## Verification status

See `docs/TEST_REPORT.md` and `outputs/smoke/SMOKE_TEST_REPORT.md`. No production simulation and no real-data analysis were run.

