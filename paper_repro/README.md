# Paper reproduction

The files in this directory are small, transparent entry points for reproducing the PaFAR analyses. Algorithm implementations remain in `src/pafar_sim/`; maintained lower-level commands remain in `scripts/`.

## 1. Environment

Python 3.10 or later is recommended. From the repository root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python paper_repro/setup_environment.py --check-import
```

The helper only reports environment readiness and recommended commands; it does not install software or modify the system.

## 2. Primary simulation

The wrapper uses `configs/exp1_primary.yaml` and `configs/exp2_primary.yaml`, delegates to the maintained simulation runners, and requires explicit confirmation:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

These are production-scale analyses. Review `--help`, the frozen configuration, expected replicate counts, and compute requirements before execution. Existing production output is never reused unless `--resume` is supplied.

## 3. Real-data analysis

The retrospective analysis uses PhysioNet/Computing in Cardiology Challenge 2019 data. Raw data are not included and are never downloaded by these helpers. Place authorized files at:

```text
data/physionet2019/raw/training_setA/
data/physionet2019/raw/training_setB/
```

Check the setup, then explicitly confirm the formal pipeline:

```bash
python paper_repro/run_realdata.py --check-only
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

The wrapper does not modify raw PSV files and does not copy the real-data implementation into this directory.

## 4. Reproducing manuscript tables and figures

Build manuscript-ready artifacts only after saved simulation and real-data summaries exist:

```bash
python paper_repro/build_paper_outputs.py --component all --confirm BUILD_PAFAR_PAPER_OUTPUTS
```

- Simulation tables and figures come from saved simulation results.
- Retrospective-analysis tables and figures come from saved real-data results.

The wrapper calls `scripts/build_primary_manuscript_outputs.py` and `scripts/build_realdata_manuscript_outputs.py`. It does not run simulations, train XGBoost, tune on the test set, or change statistical results.

## 5. Data availability

Raw PhysioNet patient records and official evaluation resources must be obtained separately from authorized PhysioNet sources and used under their applicable terms.

## 6. Generated outputs

Large checkpoints, models, feature caches, bootstrap objects, raw simulation output, real-data output, and generated figures are excluded from Git version control. They remain under ignored local `outputs/`, `data/`, and manuscript artifact paths.

## 7. Manuscript

Manuscript `.tex` and `.pdf` files are not included in the code repository. The reproduction workflow generates local tables and figures for integration into separately maintained manuscript source.
