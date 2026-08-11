# Design decisions

- No statistical conflict was found between the user specification and the inspected equations/Appendix B of `PaFAR.pdf`. The PDF therefore required no conflict override.
- Longest-quartile PFA uses decreasing eligible exposure, stable patient-id order for ties, and the first `ceil(n0/4)` patients because the manuscript does not prescribe exposure-tie handling.
- Experiment II includes only generated `A`, `Q`, elapsed ICU hour, and causal biomarker features. ICU-unit and admission-offset variables are not fabricated.
- The E3 source-only learner has no target-site category, because such a column has no source variation.
- Experiment II monitoring lengths use the common 0.70/0.30 mixed distribution at both sites; the S4-only 0.60/0.40 target shift is not transferred to E3.
- A final single empty PaFAR-T bin uses location 0 and scale 1. All other scales use `max(1.4826*MAD, 0.10)`.
- Undefined metrics remain IEEE NaN in CSV. No blanket fill-to-zero is used. Valid denominators with an infinite threshold yield zero PFA/sensitivity, while PPV and median lead remain undefined.
- Failure checkpoints contain all expected method rows with `learner_failure=True`; no seed replacement or automatic retry occurs.

