# PaFAR

PaFAR is a finite-sample patient-level false-alarm calibration framework for dynamic clinical early-warning systems.

## Overview

Clinical early-warning scores are repeatedly evaluated along each patient trajectory. PaFAR calibrates trajectory-level alert thresholds to control patient-level false-alarm probability in finite samples. The implementation separates model fitting, validation, calibration, and testing; supports time-standardized monitoring; and evaluates scalar target-site recalibration without test-based tuning.

## Repository structure

- `src/`: PaFAR algorithms, simulation models, metrics, and real-data implementation.
- `scripts/`: maintained simulation, real-data, aggregation, and output-building entry points.
- `configs/`: frozen simulation and PhysioNet analysis configurations.
- `tests/`: unit and integration tests.
- `docs/`: method implementation decisions and reproducibility details.
- `paper_repro/`: safe, documented entry points for reproducing paper analyses.

## Installation

Python 3.10 or later is required.

```bash
git clone <repository-url>
cd PaFAR
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Primary simulation

The paper reproduction wrapper uses the frozen primary configurations and requires an explicit confirmation string before starting a production run:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

Review `python paper_repro/run_simulation.py --help` and `docs/REPRODUCIBILITY.md` before launching these computationally intensive jobs.

## Real-data analysis

The retrospective analysis uses PhysioNet/Computing in Cardiology Challenge 2019 data. Raw patient records are not included. Place authorized data at:

```text
data/physionet2019/raw/training_setA/
data/physionet2019/raw/training_setB/
```

Validate the setup before explicitly confirming the analysis:

```bash
python paper_repro/run_realdata.py --check-only
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

## Paper reproduction

See [`paper_repro/README.md`](paper_repro/README.md) for the complete simulation, real-data, and saved-result table/figure workflow.

## Data availability

Raw PhysioNet records and official evaluation resources must be obtained separately from authorized PhysioNet sources. Patient data, feature caches, models, checkpoints, and generated outputs are excluded from version control.

## Tests

```bash
python -m pytest -q
```

## Citation

Citation information will be added after publication.

## License

No open-source license has been granted. This private repository contains research material for authorized collaboration.
