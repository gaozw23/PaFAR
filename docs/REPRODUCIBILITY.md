# Reproducibility

Run all commands from the repository root. The user-facing entry points are under `paper_repro/`, and the lower-level analysis scripts are under `scripts/`.

## Environment

PaFAR requires Python 3.10 or later. Create an isolated environment and install the package with its test dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python paper_repro/setup_environment.py --check-import
```

The package definition is in `pyproject.toml`; `requirements.txt` provides a flat dependency reference.

## Data

The retrospective analysis uses version 1.0.0 of the public training data from the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) (DOI: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)). Obtain the records from PhysioNet and place them at:

```text
data/physionet2019/raw/training_setA/
data/physionet2019/raw/training_setB/
```

Raw patient records are not distributed with this repository. The pipeline reads the source PSV files without modifying them. See [`data/README.md`](../data/README.md) for details.

## Simulations

The primary configurations are:

- `configs/exp1_primary.yaml`: Experiment I scenarios S1–S4, 500 replicates per scenario;
- `configs/exp2_primary.yaml`: Experiment II scenarios E1–E3, with 100 replicates for E1 and E2 and 50 replicates for E3;
- `configs/exp1_sensitivity.yaml` and `configs/exp2_sensitivity.yaml`: prespecified simulation sensitivity analyses.

Run the primary experiments with:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

These production-scale simulations are computationally intensive. The commands support resuming completed replicates and write generated files under the ignored `outputs/production/` directory. Lower-level options are documented by `python paper_repro/run_simulation.py --help`.

## Retrospective analysis

The primary and robustness configurations are `configs/realdata_primary.yaml` and `configs/realdata_robustness.yaml`. After installing the dependencies and placing the PhysioNet files in the expected directories, check the inputs and run the analysis with:

```bash
python paper_repro/run_realdata.py --check-only
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

The workflow validates the input files, constructs the cohort and backward-looking features, creates patient-level data partitions, fits the score model, performs internal and cross-hospital analyses, computes bootstrap and robustness summaries, and prepares the saved outputs used for manuscript tables and figures. Generated files are written under the ignored `outputs/realdata/` directory.

## Calibration details

- Training, validation, calibration, and test partitions are disjoint at the patient level. Learners and preprocessing parameters are fit on training patients, and model selection uses validation patients.
- Features at each evaluation time use only information observed by that time; moving windows and missingness summaries are backward-looking.
- Each eligible non-event calibration patient contributes one trajectory-level maximum score.
- Marginal PaFAR uses the finite-sample order index `ceil((m0 + 1) * (1 - alpha))`. If that index exceeds the calibration sample size, the threshold is positive infinity.
- PaFAR-HC uses its prespecified binomial-tail index rather than an asymptotic replacement.
- An alert occurs at the first eligible time for which the score is strictly greater than the threshold; a tie does not trigger an alert.
- For PaFAR-T, time-bin boundaries and robust location and scale values are estimated from validation non-event trajectories. Sparse adjacent bins are merged before the template is fixed and applied unchanged to calibration and test trajectories.
- Direct transfer retains the source learner, features, preprocessing, smoother, and time template. Target-site recalibration changes only the scalar threshold using target-site non-event calibration trajectories; target test outcomes are not used for fitting or tuning.

## Tables and figures

Once the required saved summaries are available, check their presence and generate the manuscript artifacts with:

```bash
python paper_repro/build_paper_outputs.py --component all --check-only
python paper_repro/build_paper_outputs.py --component all --confirm BUILD_PAFAR_PAPER_OUTPUTS
```

Simulation artifacts are derived from saved simulation summaries, and retrospective-analysis artifacts are derived from saved real-data summaries. Generated tables and figures remain under ignored local output directories.

## Randomness

Simulation and analysis randomness is controlled by the prespecified seeds stored in the configuration files.
