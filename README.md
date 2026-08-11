# PaFAR

PaFAR develops finite-sample patient-level false-alarm control for dynamic clinical early-warning systems.

## Status

- Primary simulation experiments are complete.
- The retrospective PhysioNet analysis is complete.
- The manuscript is under development.

This private repository contains research code, configurations, tests, and manuscript-ready artifacts. Generated run directories, model objects, caches, and raw clinical data are intentionally excluded.

## Repository structure

- `src/`: Python package implementing simulation and real-data methods.
- `scripts/`: command-line entry points for validation, experiments, aggregation, and manuscript outputs.
- `configs/`: versioned simulation and retrospective-analysis configurations.
- `tests/`: unit and integration tests.
- `docs/`: design, implementation, audit, and reproducibility notes.
- `literature/`: manuscript source, bibliography, final PDF, figures, and generated tables.
- `data/README.md`: instructions for supplying local PhysioNet data; no patient data are included.
- `outputs/README.md`: policy for local generated outputs.

## Installation

Python 3.10 or later is required. From the repository root:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'
```

On POSIX systems, activate `.venv/bin/activate` and run the equivalent `python -m pip` commands.

## Simulations

The full command record and production ranges are documented in `docs/RUNBOOK.md`. Representative commands are:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_exp1.py --config configs\exp1_primary.yaml --scenario S1 --replicate-start 0 --replicate-end 9 --n-jobs 4 --output-root outputs\production
& '.\.venv\Scripts\python.exe' scripts\run_exp2.py --config configs\exp2_primary.yaml --scenario E1 --replicate-start 0 --replicate-end 99 --n-jobs 4 --output-root outputs\production
```

These commands create local generated outputs that are not version controlled. Review the configuration and requested replicate ranges before launching a production run.

## Retrospective PhysioNet analysis

Place the authorized PhysioNet/CinC 2019 training data at the locations described in `data/README.md`, then validate the environment and use the guarded pipeline entry point:

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_realdata_environment.py --config configs\realdata_primary.yaml
& '.\.venv\Scripts\python.exe' scripts\run_realdata_pipeline.py --config configs\realdata_primary.yaml --check-only
```

The full pipeline requires the explicit confirmation values enforced by `scripts/run_realdata_pipeline.py`. Raw patient files, processed data, feature caches, fitted models, checkpoints, and bootstrap objects must remain local.

## Reproducing manuscript tables and figures

Manuscript-ready tables and figures are retained under `literature/generated_tables/` and `literature/figures/`. To rebuild them from existing local saved outputs without rerunning simulations or the retrospective analysis:

```powershell
& '.\.venv\Scripts\python.exe' scripts\build_primary_manuscript_outputs.py --check-only
& '.\.venv\Scripts\python.exe' scripts\build_realdata_manuscript_outputs.py --config configs\realdata_primary.yaml
```

The second command requires the corresponding locked local real-data summaries under the ignored output tree.

## Tests

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Tests are run from the repository root. Some real-data integration checks require the locally supplied PhysioNet files or official evaluation resources and do not make those resources redistributable.

## Data availability, privacy, and third-party material

PhysioNet/CinC 2019 raw data are not included. Obtain data and the official evaluation resources directly from their authorized upstream sources and comply with their terms. Do not commit patient files, identifiers, derived patient-level data, caches, credentials, or other restricted artifacts. A downloaded third-party reference PDF is also excluded from this repository.

## Citation

A formal citation will be added when the manuscript is released. Until then, cite the project as “PaFAR: finite-sample patient-level false-alarm control for dynamic clinical early-warning systems” and contact the authors for the current manuscript citation.

## License

No open-source license has been granted. This private repository contains research material and does not grant permission to use, copy, modify, or redistribute the code or manuscript beyond separately authorized collaboration.
