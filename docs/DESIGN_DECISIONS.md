# Method implementation decisions

This document records implementation choices that affect interpretation or reproducibility of PaFAR. Operational debugging history is intentionally excluded.

## Patient-level separation

- Train, validation, calibration, and test partitions are disjoint at the patient level.
- Learners and preprocessing parameters are fit only on training patients.
- Model selection and early stopping use validation patients; the calibration and test partitions do not tune the learner.
- Calibration operates on one trajectory-level maximum per eligible non-event patient, preserving the patient as the statistical unit.

## Causal score construction

- Features at an evaluation time use only information observed by that time.
- Moving windows and missingness summaries are backward-looking.
- The formal alert is the first eligible time at which the monitored score is strictly greater than the frozen threshold; ties do not trigger an alert.

## Finite-sample calibration

- Marginal PaFAR uses the finite-sample order index `ceil((m0 + 1) * (1 - alpha))` on non-event patient maxima.
- When the required order statistic lies above the available calibration sample, the threshold is represented as positive infinity rather than silently relaxed.
- PaFAR-HC uses the prespecified binomial-tail index and does not replace the exact finite-sample calculation with an asymptotic approximation.
- Undefined metrics remain undefined; they are not filled with zero.

## PaFAR-T template freezing

- Time-template boundaries and robust location/scale values are fit using validation non-event trajectories.
- Sparse adjacent time bins are merged before the template is frozen.
- Weighted medians and median absolute deviations use patient-equalized contributions, with the prespecified scale floor.
- The frozen template is applied unchanged to calibration and test trajectories.

## Target-site recalibration

- Direct transfer freezes the source learner, feature definition, preprocessing, smoother, and template.
- Local target recalibration changes only the scalar alert threshold using target-site non-event calibration trajectories.
- Nested target calibration sizes use fixed prefixes of one seed-defined reservoir.
- Target test outcomes are not used for learner selection, template fitting, threshold selection, or method tuning.

## Reproducibility principles

- Randomness is derived from versioned master seeds and deterministic seed streams.
- Effective configurations and result checkpoints carry checksums needed for resume validation.
- Completed compatible checkpoints may be resumed; seeds are not replaced after a failed replicate.
- Manuscript tables and figures are built from saved summaries rather than by silently rerunning analyses.
