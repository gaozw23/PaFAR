# Reproducing the PaFAR analyses

This directory documents how the code produces the analyses and manuscript-ready artifacts. It does not contain manuscript source, compiled papers, raw data, or generated results.

Run all commands from the repository root after installing the package and its test dependencies into a project-local environment.

## Primary simulations

The versioned primary configurations are `configs/exp1_primary.yaml` and `configs/exp2_primary.yaml`. The complete prespecified command record, including scenario and replicate ranges, is maintained in `docs/RUNBOOK.md`.

Representative entry points are:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_primary.yaml --scenario S1 --replicate-start 0 --replicate-end 9 --n-jobs 4 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_primary.yaml --scenario E1 --replicate-start 0 --replicate-end 99 --n-jobs 4 --output-root outputs\production
```

Use the exact configuration, seed, scenario, and replicate range required by the analysis plan. Generated outputs are written under `outputs/` and are not version controlled.

## Real-data analysis

The retrospective pipeline uses PhysioNet/Computing in Cardiology Challenge 2019 data with `configs/realdata_primary.yaml` and `configs/realdata_robustness.yaml`. Validate the local resources before running any analysis:

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_realdata_environment.py --config configs\realdata_primary.yaml
& '.\.venv\Scripts\python.exe' scripts\run_realdata_pipeline.py --config configs\realdata_primary.yaml --check-only
```

The full pipeline is guarded by explicit confirmation values implemented in `scripts/run_realdata_pipeline.py`. Consult that entry point and the project documentation before selecting a stage. Do not treat `--check-only` as an analysis run.

PhysioNet raw data are not included. Authorized users must obtain the data and official evaluation resources from PhysioNet and place them at the relative locations declared in the configuration. Patient files, processed datasets, feature caches, and patient-level outputs remain local.

## Manuscript-ready tables and figures

The existing small entry-point scripts remain under `scripts/` so there is a single maintained implementation:

```powershell
& '.\.venv\Scripts\python.exe' scripts\build_primary_manuscript_outputs.py --check-only
& '.\.venv\Scripts\python.exe' scripts\build_realdata_manuscript_outputs.py --config configs\realdata_primary.yaml
```

The first command validates the available primary simulation summaries without rerunning simulations. The real-data builder consumes the locked local summaries produced by the retrospective pipeline. The scripts write manuscript-ready artifacts into the locally retained manuscript/output locations; those generated files are not tracked in this code repository.

No entry-point script is copied into this directory because duplicating executable code would create maintenance risk. The authoritative scripts are referenced above.

## Excluded research artifacts

- PhysioNet raw data and official evaluation downloads are not included.
- Large simulation and real-data outputs are not included.
- Feature caches, fitted models, checkpoints, bootstrap objects, logs, and failure archives are not included.
- Manuscript source and compiled paper files are not included in the code repository.

These exclusions affect Git tracking only; they do not prescribe deletion of a researcher's authorized local data or generated analysis artifacts.
