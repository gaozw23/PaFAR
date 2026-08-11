# Non-finite threshold serialization fix

## Diagnosis

The failing values are the fifth components of the prespecified Bonferroni
threshold vectors for hospitals A and B at both operating points, alpha 0.05
and 0.10.  Each vector has shape `(5,)`, uses the fixed-logit score scale, and
corresponds to boundaries `[6, 12, 24, 48, 72, 168]`.  Only the final
`(72, 168]` bin is `-inf`.

This is an algorithm-defined sentinel, not NaN propagation.  The locked
`trajectory_max` definition maps an empty eligible set to `-inf`; the
prespecified binwise Bonferroni order statistic legitimately retains that
value when the late-time support is sparse.  The diagnostic is recorded in
`outputs/realdata/logs/nonfinite_threshold_diagnostic.json`.

## Patch scope

`encode_json_numeric` and `decode_json_numeric` recursively support Python and
NumPy floating values, ndarrays, lists, tuples, and dictionaries.  Finite
values remain numeric.  Positive and negative infinity use explicit
`{"__float__":"pos_inf"}` and `{"__float__":"neg_inf"}` tags.  Unexpected
NaN raises `ValueError`, and every JSON writer uses strict `allow_nan=False`.

Internal threshold JSON-in-CSV fields now use the codec and include
`threshold_state`, `threshold_nonfinite_count`, `threshold_has_neg_inf`, and
`threshold_has_pos_inf`.  Transfer CSV thresholds remain numeric and receive
the same state columns.  JSON sidecars written through `atomic_write_json`
also use the codec.

## Statistical invariance

Regression tests compare the numeric threshold before serialization with the
decoded value, the strict `score > threshold` crossing matrix, first alerts,
PFA, and sensitivity.  They are identical, including `np.isneginf` and
`np.isposinf` after round-trip.

This patch changes serialization only and does not alter threshold
computation, alerting, calibration, or statistical results.
