# Windows PowerShell runbook

All commands run from the project root and use the project-local Python 3.12 environment.

```powershell
# Environment and installation
& '.\.venv\Scripts\python.exe' scripts\check_environment.py
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'

# Tests and smoke only
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' scripts\run_smoke.py --n-jobs 1

# Build a production oracle (repeat for scenarios/sites required)
& '.\.venv\Scripts\python.exe' scripts\build_exp1_oracle.py --config configs\exp1_primary.yaml --scenario S1 --site A --output-root outputs\oracle

# Selected Experiment I scenarios/range
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_primary.yaml --scenario S1 --scenario S2 --replicate-start 0 --replicate-end 9 --n-jobs 4 --master-seed 20260802 --output-root outputs\production

# Experiment I sensitivities
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_sensitivity.yaml --condition calibration_size --replicate-start 0 --replicate-end 249 --n-jobs 4 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_sensitivity.yaml --condition alpha_005 --replicate-start 0 --replicate-end 249 --n-jobs 4 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_sensitivity.yaml --condition weak_signal --replicate-start 0 --replicate-end 249 --n-jobs 4 --output-root outputs\production

# Experiment II primary counts: E1/E2 have 100 replicates; E3 has 50
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_primary.yaml --scenario E1 --scenario E2 --replicate-start 0 --replicate-end 99 --n-jobs 4 --master-seed 20260802 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_primary.yaml --scenario E3 --replicate-start 0 --replicate-end 49 --n-jobs 4 --master-seed 20260802 --output-root outputs\production

# Experiment II one-factor sensitivities
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_sensitivity.yaml --condition alpha_005 --replicate-start 0 --replicate-end 49 --n-jobs 4 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_sensitivity.yaml --condition weak_signal --replicate-start 0 --replicate-end 49 --n-jobs 4 --output-root outputs\production

# Resume the exact same range/checksum; complete checkpoints are skipped
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_primary.yaml --scenario E3 --replicate-start 0 --replicate-end 49 --n-jobs 4 --output-root outputs\production --resume

# Saved-output aggregation and reporting (never reruns simulations)
& '.\.venv\Scripts\python.exe' scripts\summarize_results.py --raw-root outputs\production\raw --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\make_figures.py --results outputs\production\all_replicate_results.csv.gz --output-dir outputs\production
```

Production commands are examples only and were not executed during implementation.
