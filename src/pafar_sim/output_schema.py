"""Stable method metadata for raw results, summaries, and diagnostics."""
from __future__ import annotations

from typing import Any


def describe_method(method: str, scenario: str, *, alias: bool = False) -> dict[str, Any]:
    """Return non-statistical method/deployment metadata without changing results."""
    if method == "Fixed 0.5":
        family, scale = "fixed_probability", "risk_probability"
    elif method == "Youden":
        family, scale = "youden", "risk_probability"
    elif method == "Binwise Bonferroni":
        family, scale = "binwise_bonferroni", "vector_transformed_logit"
    elif "PaFAR-T" in method:
        family, scale = "pafar_t", "standardized_time_template"
    elif "PaFAR-HC" in method:
        family, scale = "pafar_hc", "transformed_logit"
    elif "PaFAR-F" in method or method == "Direct source transfer":
        family, scale = "pafar_f", "transformed_logit"
    elif method == "Naive maximum":
        family, scale = "naive_maximum", "transformed_logit"
    elif method == "Pointwise-alpha":
        family, scale = "pointwise_alpha", "transformed_logit"
    elif method == "Oracle-F":
        family, scale = "oracle_f", "transformed_logit"
    else:
        family, scale = method.lower().replace(" ", "_"), "unknown"
    shifted = scenario in {"S4", "E3"}
    if method.startswith("Local"):
        strategy, threshold_origin = "local_target_recalibration", "target_calibration_non_events"
    elif shifted:
        strategy, threshold_origin = "direct_source_transfer", "source_calibration_non_events"
    else:
        strategy, threshold_origin = "within_site_calibration", "calibration_non_events"
    if method == "Fixed 0.5":
        threshold_origin = "prespecified_constant"
    elif method == "Youden":
        threshold_origin = "source_validation" if shifted else "validation"
    elif method == "Oracle-F":
        threshold_origin = "independent_oracle_reference"
    template_origin = "source_validation_non_events" if shifted and "PaFAR-T" in method else (
        "validation_non_events" if "PaFAR-T" in method else "none"
    )
    return {
        "method_family": family, "deployment_strategy": strategy, "threshold_scale": scale,
        "threshold_origin": threshold_origin, "template_origin": template_origin,
        "is_alias": bool(alias), "alias_of": "PaFAR-F" if alias else "",
    }

