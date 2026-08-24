# PaFAR

Code and reproducibility materials for *PaFAR: Finite-Sample Patient-Level False-Alarm Calibration for Repeated Risk Monitoring* by Zhuowei Gao and Weipeng Sun, School of Mathematics, Jilin University.

## Overview

PaFAR calibrates repeated risk monitoring at the level of a complete monitoring episode. Each eligible non-event calibration trajectory contributes one maximum score, and a finite-sample corrected order statistic determines the alert threshold.

PaFAR is a calibration layer applied to frozen score trajectories. It does not refit the underlying risk model or change its discrimination.

## Installation

PaFAR requires Python 3.10 or later.

```bash
git clone https://github.com/gaozw23/PaFAR.git
cd PaFAR
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Data

The retrospective analysis uses version 1.0.0 of the public training data from the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) (DOI: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)). Raw clinical records are not distributed with this repository. See [`data/README.md`](data/README.md) for the expected local layout.

## Reproducing the paper

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for configuration details, data layout, computational requirements, and the calibration rules used in the analyses.

### Simulations

The primary simulations use the configurations under `configs/`:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

### Retrospective analysis

After placing the PhysioNet files in the documented locations, check the inputs and run the analysis with:

```bash
python paper_repro/run_realdata.py --check-only
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

### Tables and figures

Saved simulation and retrospective-analysis summaries are required to generate the manuscript tables and figures:

```bash
python paper_repro/build_paper_outputs.py --component all --check-only
python paper_repro/build_paper_outputs.py --component all --confirm BUILD_PAFAR_PAPER_OUTPUTS
```

The saved summaries and generated artifacts are not distributed with the repository. Additional entry-point information is available in [`paper_repro/README.md`](paper_repro/README.md).

## Code version

The numerical results reported in the manuscript correspond to commit `c4a50d8b303a3d9994d8c9355724e5388f6ef613`.

Later documentation-only changes do not alter the implementation used for the reported results.

## Tests

```bash
python -m pytest -q
```

The tests cover the principal calibration, alerting, patient-level splitting, causal feature construction, reproducibility, and output-generation rules.

## Citation

If you use this software, please cite the accompanying manuscript:

> Gao, Z. and Sun, W. *PaFAR: Finite-Sample Patient-Level False-Alarm Calibration for Repeated Risk Monitoring.* Publication details will be updated after publication.

## License

The source code in this repository is released under the [MIT License](LICENSE).
