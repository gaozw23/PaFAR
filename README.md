# PaFAR

Code and reproducibility materials accompanying the manuscript *PaFAR: Finite-Sample Patient-Level False-Alarm Calibration for Repeated Risk Monitoring* by Zhuowei Gao and Weipeng Sun, School of Mathematics, Jilin University, Changchun, Jilin 130012, China.

## Overview

PaFAR calibrates repeated risk monitoring at the level of a complete monitoring episode rather than at individual decision times. Each non-event calibration trajectory contributes one maximum score, and a finite-sample corrected order statistic is used to select the deployment threshold.

PaFAR is a calibration layer for a frozen score trajectory. It does not refit the underlying risk model, improve discrimination by itself, or imply clinical effectiveness, universal robustness, or conditional subgroup control.

## Repository scope

This repository contains:

- analysis and simulation code;
- frozen configuration files;
- regression and implementation tests;
- environment checks and reproduction scripts.

The repository does not redistribute the raw PhysioNet records, manuscript submission files, or the large generated output directory. Machine-readable manuscript results are not included in the current repository tree.

## Manuscript code version

The code version audited for the manuscript is:

```text
c4a50d8b303a3d9994d8c9355724e5388f6ef613
```

That commit is preserved in the repository history. Subsequent commits, if any, are limited to public-release documentation and repository presentation and do not alter the audited statistical implementation.

## Repository structure

```text
.
├── configs/          Frozen simulation and retrospective-analysis configurations
├── data/             Public data instructions only; clinical records remain local
├── docs/             Design decisions and reproducibility guidance
├── paper_repro/      Guarded, user-facing reproduction entry points
├── scripts/          Maintained simulation, analysis, aggregation, and output builders
├── src/pafar_sim/    PaFAR implementation
├── tests/            Regression and implementation tests
├── pyproject.toml    Package metadata and dependency specification
└── requirements.txt  Flat dependency reference
```

## Data

The retrospective application uses the public training component of the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/), version 1.0.0, DOI [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857).

Raw clinical records are not redistributed in this repository. Obtain the data through PhysioNet and place the two training sets in the repository-relative locations documented in [`data/README.md`](data/README.md).

## Environment and installation

Python 3.10 or later is required by `pyproject.toml`. From the repository root:

```bash
git clone https://github.com/gaozw23/PaFAR.git
cd PaFAR
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python paper_repro/setup_environment.py --check-import
```

`pyproject.toml` is the authoritative package definition. `requirements.txt` is retained as a flat dependency reference.

## Reproduction

### 1. Environment and lightweight verification

Confirm that the package is importable and run the test suite:

```bash
python paper_repro/setup_environment.py --check-import
python -m pytest -q
```

These checks do not run the production simulations or the retrospective analysis.

### 2. Primary simulations

The guarded wrappers use the frozen primary configurations and require an explicit confirmation value:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

These are computationally intensive production workflows. Review [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), the selected configuration, and `python paper_repro/run_simulation.py --help` before execution. Generated results are written beneath the ignored `outputs/` directory.

### 3. PhysioNet workflow

After obtaining the required PhysioNet files, validate the local layout without running the analysis:

```bash
python paper_repro/run_realdata.py --check-only
```

Run the maintained retrospective-analysis pipeline only when the required local data are present:

```bash
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

The raw clinical records are not downloaded by these commands and are not included in Git.

### 4. Tables and figures

Existing saved simulation and retrospective-analysis summaries are required before manuscript-ready artifacts can be regenerated. Check that the required summaries are present without writing outputs:

```bash
python paper_repro/build_paper_outputs.py --component all --check-only
```

Then build the artifacts with:

```bash
python paper_repro/build_paper_outputs.py --component all --confirm BUILD_PAFAR_PAPER_OUTPUTS
```

The large saved summaries and generated artifacts are not distributed in this repository, so this step is not a one-command workflow from a fresh clone. See [`paper_repro/README.md`](paper_repro/README.md) for the maintained entry-point details.

## Tests

The test suite covers calibration, alerting, patient-level splitting, causal feature construction, reproducible random-number streams, output schemas, and saved-output generation:

```bash
python -m pytest -q
```

## Citation

If you use this software, please cite the accompanying manuscript:

> Gao, Z. and Sun, W. *PaFAR: Finite-Sample Patient-Level False-Alarm Calibration for Repeated Risk Monitoring.* Publication details will be updated after publication.
