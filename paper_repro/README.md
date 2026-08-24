# Paper reproduction

This directory contains the entry points for reproducing the PaFAR analyses. Method implementations are under `src/pafar_sim/`, and lower-level commands are under `scripts/`.

## Environment

From the repository root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python paper_repro/setup_environment.py --check-import
```

## Simulations

The simulation entry point uses `configs/exp1_primary.yaml` and `configs/exp2_primary.yaml`:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

The runs are computationally intensive. Existing compatible results are reused only when `--resume` is supplied, and generated files are written below `outputs/production/`.

## Retrospective analysis

Obtain the PhysioNet/Computing in Cardiology Challenge 2019 data separately and place the training sets at:

```text
data/physionet2019/raw/training_setA/
data/physionet2019/raw/training_setB/
```

Check the inputs and run the analysis with:

```bash
python paper_repro/run_realdata.py --check-only
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

Raw records are not downloaded or modified by these commands. Generated analysis files are written below `outputs/realdata/`.

## Tables and figures

After the saved simulation and retrospective-analysis summaries are available, run:

```bash
python paper_repro/build_paper_outputs.py --component all --check-only
python paper_repro/build_paper_outputs.py --component all --confirm BUILD_PAFAR_PAPER_OUTPUTS
```

The entry point calls `scripts/build_primary_manuscript_outputs.py` and `scripts/build_realdata_manuscript_outputs.py`. It builds tables and figures from saved summaries without rerunning simulations or fitting models.

See [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the data, configuration, calibration, and randomness details.
