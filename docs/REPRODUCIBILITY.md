# Reproducibility guide

All commands below are run from the repository root. The canonical user-facing entry points are under `paper_repro/`; maintained lower-level commands remain under `scripts/`.

## Environment and installation

PaFAR requires Python 3.10 or later. Create an isolated environment and install the package with test dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python paper_repro/setup_environment.py --check-import
```

The package definition in `pyproject.toml` is authoritative. `requirements.txt` is retained as a flat dependency reference.

## Frozen configurations and seeds

- `configs/exp1_primary.yaml`: Experiment I primary scenarios S1–S4; master seed `20260802`.
- `configs/exp2_primary.yaml`: Experiment II primary scenarios E1–E3; master seed `20260802`.
- `configs/exp1_sensitivity.yaml` and `configs/exp2_sensitivity.yaml`: prespecified one-factor simulation sensitivities.
- `configs/realdata_primary.yaml`: primary PhysioNet analysis; master seed `20260804`.
- `configs/realdata_robustness.yaml`: prespecified retrospective robustness analyses.
- `configs/smoke.yaml`: small test-only configuration, not a substitute for a primary analysis.

Do not change seeds, scenario definitions, replicate ranges, or analysis grids when reproducing the reported analyses.

## Simulation entry points

Inspect the safe wrapper first:

```bash
python paper_repro/run_simulation.py --help
```

Run the primary experiments with explicit confirmation and resume semantics:

```bash
python paper_repro/run_simulation.py --experiment exp1 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
python paper_repro/run_simulation.py --experiment exp2 --resume --confirm RUN_PAFAR_PRIMARY_SIMULATION
```

Experiment I runs S1–S4 for 500 replicates each. Experiment II runs E1 and E2 for 100 replicates each and E3 for 50 replicates. The wrapper delegates to `scripts/run_exp1.py` and `scripts/run_exp2.py`; statistical implementation remains in `src/pafar_sim/`.

Lower-level runs can select explicit scenarios and replicate ranges, for example:

```bash
python scripts/run_exp1.py --config configs/exp1_primary.yaml --scenario S1 --replicate-start 0 --replicate-end 499 --n-jobs 4 --output-root outputs/production --resume
python scripts/run_exp2.py --config configs/exp2_primary.yaml --scenario E1 --replicate-start 0 --replicate-end 99 --n-jobs 4 --output-root outputs/production --resume
```

## Real-data entry points

Obtain PhysioNet/CinC 2019 data and the official evaluation resource from authorized upstream sources. Expected raw-data directories are:

```text
data/physionet2019/raw/training_setA/
data/physionet2019/raw/training_setB/
```

The pipeline never downloads these records. Check inputs, then run with explicit confirmation:

```bash
python paper_repro/run_realdata.py --check-only
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

The wrapper delegates to `scripts/run_realdata_pipeline.py`. The implementation in `src/pafar_sim/realdata/` performs raw-file audit, cohort construction, causal feature generation, patient-level splitting, internal analysis, cross-hospital transfer, bootstrap summaries, robustness analyses, and manuscript-output preparation.

## Expected local output structure

Generated content is intentionally ignored by Git:

```text
outputs/
├── production/        # simulation checkpoints, aggregates, and saved summaries
└── realdata/          # real-data locks, caches, summaries, tables, and figures

literature/
├── generated_tables/  # local manuscript-ready tables
└── figures/           # local manuscript-ready figures
```

Raw simulation checkpoints, fitted learners, feature caches, bootstrap objects, and patient-level outputs are not version controlled.

## Manuscript tables and figures

After the corresponding saved summaries exist, regenerate manuscript-ready artifacts without rerunning simulations or fitting models:

```bash
python paper_repro/build_paper_outputs.py --component all --confirm BUILD_PAFAR_PAPER_OUTPUTS
```

- Simulation tables and figures are derived from saved simulation summaries.
- Retrospective-analysis tables and figures are derived from saved real-data summaries.
- Outputs are written under ignored local `literature/` and `outputs/` paths.

## Tests and reproducibility checks

Run the full test suite:

```bash
python -m pytest -q
```

The tests cover patient-level splitting, causal feature construction, strict threshold crossing, finite-sample order statistics, time-template freezing, target-site recalibration, seed reproducibility, result schemas, and saved-output generation. Before using `--resume`, the pipeline checks configuration/checkpoint compatibility rather than silently combining incompatible runs.

## Implementation traceability

| Method component | Main implementation | Representative tests |
| --- | --- | --- |
| Eligible monitoring and trajectory scores | `src/pafar_sim/score.py` | `tests/test_score.py` |
| Strict first alert and alert summaries | `src/pafar_sim/alerting.py` | `tests/test_alerting.py` |
| Marginal, time-template, and HC calibration | `src/pafar_sim/calibration.py` | `tests/test_calibration.py`, `tests/test_time_template.py` |
| Patient-level operating metrics | `src/pafar_sim/metrics.py` | `tests/test_metrics.py` |
| Experiment I and II generators/runners | `src/pafar_sim/exp1/`, `src/pafar_sim/exp2/` | `tests/test_exp1_dgp.py`, `tests/test_exp2_dgp.py` |
| Retrospective PhysioNet pipeline | `src/pafar_sim/realdata/` | `tests/realdata/` |
| Saved-result tables and figures | maintained build scripts | `tests/test_manuscript_outputs.py` |

## Data availability

Raw PhysioNet patient records are not distributed with this repository. Users are responsible for obtaining authorized data and official evaluation resources and for complying with their access and redistribution terms. The pipeline does not modify raw PSV files.
