# Performance notes

- Experiment I uses patient-by-hour NumPy matrices. AR recurrences loop only over hour while vectorizing patients.
- Experiment II latent severity, biomarker noise, and observation draws vectorize patient/biomarker dimensions. The feature kernel loops over hour, biomarker, and three fixed windows while operating on all patients at once; it never invokes pandas rolling inside the hotspot.
- XGBoost matrices are `float32`; thresholds and calibration maxima are `float64`.
- Oracle generation is chunked and retains only one maximum per patient. Conditional PFA uses `numpy.searchsorted` on the sorted reference.
- Replicate parallelism uses joblib `loky`; XGBoost uses `nthread=1` to prevent nested oversubscription.
- Sorting/cumulative operations are reused for weighted quantiles and Youden threshold selection.
- Measured on the local Python 3.12 environment: 1,000 Experiment I trajectories in 0.0066 s, the causal feature kernel for 100 E2 patients in 0.389 s, and a 2,000-patient chunked oracle in 0.0350 s. Machine-readable values are in `outputs/benchmarks/benchmark.json`.
- The final cold smoke (including four 20,000-trajectory oracle builds) took 62.08 s; scenario timings are in `outputs/smoke/timing_summary.csv`. No unmeasured speedup claim is made.
