# PaFAR

PaFAR is a finite-sample patient-level false-alarm calibration layer for dynamic clinical early-warning systems.

## Overview

PaFAR calibrates alerting rules against patient-level false-alarm risk rather than treating repeated measurements as independent alarms. This repository contains the method implementation, simulation and retrospective-analysis entry points, versioned configurations, tests, and reproducibility documentation.

The repository is intentionally code-focused. Manuscript source, compiled papers, raw clinical data, generated outputs, model objects, and checkpoints remain outside Git tracking.

## Repository structure

- `src/`: installable Python package implementing PaFAR, simulation methods, and the retrospective-analysis pipeline.
- `scripts/`: command-line entry points for experiments, validation, aggregation, and manuscript-ready outputs.
- `configs/`: versioned simulation and PhysioNet analysis configurations.
- `tests/`: unit and integration tests.
- `docs/`: design decisions, implementation notes, audits, and operational guidance.
- `paper_repro/`: concise instructions for reproducing the analyses and manuscript-ready tables and figures.

## Installation

Python 3.10 or later is required. From the repository root:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e .
```

Install test dependencies with `python -m pip install -e ".[test]"`. On POSIX systems, activate `.venv/bin/activate` and run the equivalent commands with `python`.

## Reproducing simulations

Simulation commands use relative paths and write generated artifacts under the ignored `outputs/` directory. Representative primary runs are:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_primary.yaml --scenario S1 --replicate-start 0 --replicate-end 9 --n-jobs 4 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_primary.yaml --scenario E1 --replicate-start 0 --replicate-end 99 --n-jobs 4 --output-root outputs\production
```

Review `docs/RUNBOOK.md`, the selected YAML configuration, replicate ranges, and computational requirements before launching a production run. Smoke tests and validation commands are not substitutes for the prespecified production configurations.

## Real-data analysis

The retrospective analysis uses PhysioNet/Computing in Cardiology Challenge 2019 data. Raw data and the official evaluation resources are not distributed in this repository. After obtaining authorized copies and placing them at the locations defined by `configs/realdata_primary.yaml`, check the local setup with:

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_realdata_environment.py --config configs\realdata_primary.yaml
& '.\.venv\Scripts\python.exe' scripts\run_realdata_pipeline.py --config configs\realdata_primary.yaml --check-only
```

The guarded pipeline requires explicit confirmation values for analysis stages. Raw records, derived patient-level data, feature caches, fitted models, checkpoints, and bootstrap objects must remain local.

## Reproducing manuscript tables and figures

See [`paper_repro/README.md`](paper_repro/README.md) for the analysis-to-artifact workflow and the existing entry points under `scripts/`. Manuscript source and compiled paper files are not included in this code repository.

## Data availability

Obtain PhysioNet 2019 data and official evaluation resources directly from authorized PhysioNet sources and comply with the applicable access and redistribution terms. Do not commit patient files, identifiers, processed clinical data, credentials, or local caches.

## Tests

Run the test suite from the repository root:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Some real-data integration checks require the separately obtained local PhysioNet resources.

## Citation

A formal manuscript citation will be added when available. Until then, cite the project as “PaFAR: finite-sample patient-level false-alarm control for dynamic clinical early-warning systems.”

## License

No open-source license has been granted. This private research repository does not grant permission to use, copy, modify, or redistribute its contents beyond separately authorized collaboration.
