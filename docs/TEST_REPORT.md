# Test report

- Interpreter: project-local Python 3.12.13.
- Unit suite: 33 passed, 0 failed.
- Final cold smoke: S1-S4 (2 replicates each), E1-E3 (1 replicate each), 109 method rows.
- Smoke automated validation: 16 passed, 0 failed.
- Learner failures: 0; failures would remain as explicit method rows without resampling.
- Resume: all complete checkpoint modification times remained unchanged.
- Smoke wall time: 62.08 s; peak Python-tracked allocation: 183.8 MiB (native allocations excluded).
- Production simulation: not run.

Detailed package versions, operating points, undefined metrics, infinite thresholds, output tree, and limitations are in `outputs/smoke/SMOKE_TEST_REPORT.md` and `outputs/smoke/run_manifest.json`.
