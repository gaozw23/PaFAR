"""Build manuscript-only tables and figures from locked PRIMARY outputs.

This script is intentionally post-processing only.  It reads the completed
production CSV files and never imports or calls a simulation, learner, oracle,
or calibration runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "manuscript_primary_final" / ".mplconfig")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "outputs" / "production"
LITERATURE = ROOT / "literature"
TABLE_DIR = LITERATURE / "generated_tables"
FIGURE_DIR = LITERATURE / "figures"
SUPPLEMENT_DIR = LITERATURE / "supplementary"
POSTPROCESS_DIR = LITERATURE / "postprocess"
FINAL_DIR = ROOT / "outputs" / "manuscript_primary_final"
SOURCE_RESULTS = PRODUCTION / "all_replicate_results.csv.gz"

ALPHA = 0.10
METRICS = [
    "pfa",
    "long_stay_pfa",
    "sens3",
    "sens0",
    "premature",
    "median_lead",
    "ppv3_standardized",
    "alert_burden_100d",
]

METHOD_COLORS = {
    "Pointwise-alpha": "#8c564b",
    "Binwise Bonferroni": "#1f77b4",
    "Naive maximum": "#ff7f0e",
    "PaFAR-F": "#2ca02c",
    "PaFAR-T": "#9467bd",
    "PaFAR-HC": "#d62728",
    "Direct source transfer": "#4d4d4d",
    "Local PaFAR-F": "#2ca02c",
    "Local PaFAR-T": "#9467bd",
    "Local PaFAR-HC": "#d62728",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_results() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_RESULTS, float_precision="round_trip")
    if data["is_alias"].dtype == object:
        alias = data["is_alias"].astype(str).str.lower().eq("true")
    else:
        alias = data["is_alias"].fillna(False).astype(bool)
    data = data.loc[(data["condition"] == "primary") & ~alias].copy()
    return data


def select(
    data: pd.DataFrame,
    *,
    scenario: str,
    method: str,
    alpha: float = ALPHA,
    target_m0: int | None = None,
) -> pd.DataFrame:
    mask = (
        data["scenario"].eq(scenario)
        & data["method"].eq(method)
        & np.isclose(data["operating_alpha"], alpha)
    )
    if target_m0 is None:
        mask &= data["target_m0"].isna()
    else:
        mask &= data["target_m0"].fillna(-1).eq(float(target_m0))
    out = data.loc[mask].copy()
    if out.empty:
        raise AssertionError(
            f"No rows for scenario={scenario}, method={method}, "
            f"alpha={alpha}, target_m0={target_m0}"
        )
    if out["replicate"].duplicated().any():
        raise AssertionError(f"Duplicate replicate rows for {scenario}/{method}/{target_m0}")
    return out


def mean_mcse(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    if not n:
        return math.nan, math.nan, 0
    mean = float(np.mean(values))
    mcse = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, mcse, n


def mean_sd(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna().to_numpy(float)
    n = len(values)
    if not n:
        return math.nan, math.nan, 0
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return mean, sd, n


def fmt_mean_mcse(series: pd.Series, digits: int) -> str:
    mean, mcse, n = mean_mcse(series)
    if n == 0:
        raise AssertionError("Selected manuscript cell has no defined values")
    return f"{mean:.{digits}f} ({mcse:.{digits}f})"


def assert_close(data: pd.DataFrame, scenario: str, method: str, column: str,
                 expected: float, target_m0: int | None = None) -> float:
    value = float(select(data, scenario=scenario, method=method,
                         target_m0=target_m0)[column].mean())
    if not math.isclose(value, expected, abs_tol=5e-6, rel_tol=0.0):
        raise AssertionError(
            f"Sanity check failed for {scenario}/{method}/{column}/m0={target_m0}: "
            f"got {value:.12f}, expected {expected:.12f}"
        )
    return value


def sanity_checks(data: pd.DataFrame) -> dict[str, float]:
    checked: dict[str, float] = {}
    for scenario, expected in {
        "S1": 0.099291,
        "S2": 0.098773,
        "S3": 0.100658,
        "E1": 0.098241,
        "E2": 0.101329,
    }.items():
        checked[f"{scenario}_PaFAR-F_PFA"] = assert_close(
            data, scenario, "PaFAR-F", "pfa", expected
        )
    for scenario, method, m0, expected in [
        ("S4", "Direct source transfer", 0, 0.280479),
        ("S4", "Local PaFAR-F", 500, 0.100270),
        ("E3", "Direct source transfer", 0, 0.135885),
        ("E3", "Local PaFAR-F", 500, 0.099602),
    ]:
        checked[f"{scenario}_{method}_{m0}_PFA"] = assert_close(
            data, scenario, method, "pfa", expected, m0
        )
    for method, column, expected in [
        ("PaFAR-F", "long_stay_pfa", 0.1813),
        ("PaFAR-T", "long_stay_pfa", 0.1465),
        ("PaFAR-F", "sens3", 0.178736),
        ("PaFAR-T", "sens3", 0.183344),
    ]:
        value = float(select(data, scenario="S3", method=method)[column].mean())
        tolerance = 5e-6 if column == "sens3" else 5e-4
        if not math.isclose(value, expected, abs_tol=tolerance, rel_tol=0.0):
            raise AssertionError(
                f"S3 time-adaptation sanity check failed for {method}/{column}: "
                f"got {value:.12f}, expected approximately {expected:.12f}"
            )
        checked[f"S3_{method}_{column}"] = value

    learner = pd.read_csv(PRODUCTION / "learner_metrics.csv", float_precision="round_trip")
    for scenario, expected in {"E1": 0.715680, "E2": 0.707041, "E3": 0.723448}.items():
        value = float(learner.loc[learner["scenario"].eq(scenario), "auroc_weighted"].mean())
        if not math.isclose(value, expected, abs_tol=5e-6, rel_tol=0.0):
            raise AssertionError(f"Learner AUROC sanity check failed for {scenario}: {value}")
        checked[f"{scenario}_AUROC"] = value

    for scenario, method, m0, expected in [
        ("S1", "PaFAR-HC", None, 0.040),
        ("S2", "PaFAR-HC", None, 0.032),
        ("S3", "PaFAR-HC", None, 0.048),
        ("S4", "Local PaFAR-HC", "all", 0.038),
    ]:
        if m0 == "all":
            rows = data.loc[
                data["scenario"].eq(scenario)
                & data["method"].eq(method)
                & np.isclose(data["operating_alpha"], ALPHA)
                & data["target_m0"].isin([100.0, 250.0, 500.0])
            ]
        else:
            rows = select(data, scenario=scenario, method=method, target_m0=m0)
        value = float(rows["conditional_pfa_gt_alpha"].mean())
        if not math.isclose(value, expected, abs_tol=5e-6, rel_tol=0.0):
            raise AssertionError(f"HC exceedance sanity check failed for {scenario}/{method}: {value}")
        checked[f"{scenario}_{method}_HC_exceedance"] = value
    return checked


def tex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        text = text.replace(old, new)
    return text


def write_tex_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    notes: Sequence[str],
    column_format: str,
    resize: bool = True,
) -> None:
    headers = " & ".join(tex_escape(c) for c in frame.columns) + r" \\" 
    rows = [" & ".join(tex_escape(v) for v in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    tabular = [rf"\begin{{tabular}}{{{column_format}}}", r"\toprule", headers,
               r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    if resize:
        body = [r"\resizebox{\linewidth}{!}{%", *tabular, "}"]
    else:
        body = tabular
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        *body,
        r"\vspace{0.35em}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        *[rf"\noindent {note}\par" for note in notes],
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def table3(data: pd.DataFrame) -> pd.DataFrame:
    order = [
        ("S1", "Pointwise-alpha", "Pointwise-alpha"),
        ("S1", "Binwise Bonferroni", "Bonferroni"),
        ("S1", "PaFAR-F", "PaFAR-F"),
        ("S1", "PaFAR-HC", "PaFAR-HC"),
        ("S2", "Pointwise-alpha", "Pointwise-alpha"),
        ("S2", "Binwise Bonferroni", "Bonferroni"),
        ("S2", "PaFAR-F", "PaFAR-F"),
        ("S2", "PaFAR-HC", "PaFAR-HC"),
        ("S3", "Pointwise-alpha", "Pointwise-alpha"),
        ("S3", "PaFAR-F", "PaFAR-F"),
        ("S3", "PaFAR-T", "PaFAR-T"),
        ("S3", "PaFAR-HC", "PaFAR-HC"),
    ]
    rows = []
    for scenario, method, display in order:
        d = select(data, scenario=scenario, method=method)
        rows.append({
            "Scenario": scenario,
            "Method": display,
            "PFA": fmt_mean_mcse(d["pfa"], 3),
            "Long-stay PFA": fmt_mean_mcse(d["long_stay_pfa"], 3),
            "Sens3": fmt_mean_mcse(d["sens3"], 3),
            "Sens0": fmt_mean_mcse(d["sens0"], 3),
            "Premature": fmt_mean_mcse(d["premature"], 3),
            "Alerts/100 patient-days": fmt_mean_mcse(d["alert_burden_100d"], 2),
        })
    out = pd.DataFrame(rows)
    assert len(out) == 12 and not out[["Scenario", "Method"]].duplicated().any()
    assert not out.drop(columns=["Scenario", "Method"]).isna().all(axis=1).any()
    assert "S4" not in set(out["Scenario"])
    return out


def table4(data: pd.DataFrame) -> pd.DataFrame:
    order = [
        ("E1", "Youden"),
        ("E1", "Pointwise-alpha"),
        ("E1", "PaFAR-F"),
        ("E1", "PaFAR-T"),
        ("E2", "Pointwise-alpha"),
        ("E2", "PaFAR-F"),
        ("E2", "PaFAR-T"),
        ("E2", "PaFAR-HC"),
    ]
    rows = []
    for scenario, method in order:
        d = select(data, scenario=scenario, method=method)
        _, threshold_sd, n = mean_sd(d["threshold"])
        if not n:
            raise AssertionError(f"No finite thresholds for Table 4 {scenario}/{method}")
        rows.append({
            "Scenario": scenario,
            "Method": method,
            "PFA": fmt_mean_mcse(d["pfa"], 3),
            "Sens3": fmt_mean_mcse(d["sens3"], 3),
            "Premature": fmt_mean_mcse(d["premature"], 3),
            "Median lead": fmt_mean_mcse(d["median_lead"], 2),
            "PPV3": fmt_mean_mcse(d["ppv3_standardized"], 3),
            "Threshold SD": f"{threshold_sd:.3f}",
        })
    out = pd.DataFrame(rows)
    assert len(out) == 8 and not out[["Scenario", "Method"]].duplicated().any()
    assert not out.drop(columns=["Scenario", "Method"]).isna().all(axis=1).any()
    forbidden = {"Fixed 0.5", "Naive maximum", "Binwise Bonferroni"}
    assert not set(out["Method"]) & forbidden
    assert not ((out["Scenario"] == "E1") & (out["Method"] == "PaFAR-HC")).any()
    return out


def table5(data: pd.DataFrame) -> pd.DataFrame:
    spec = []
    for experiment, scenario in [("Experiment I", "S4"), ("Experiment II", "E3")]:
        spec.extend([
            (experiment, scenario, "Direct source transfer", "Direct source transfer", 0),
            (experiment, scenario, "Local PaFAR-F", "Local PaFAR-F", 100),
            (experiment, scenario, "Local PaFAR-F", "Local PaFAR-F", 250),
            (experiment, scenario, "Local PaFAR-F", "Local PaFAR-F", 500),
            (experiment, scenario, "Local PaFAR-T", "Local PaFAR-T (source template)", 500),
            (experiment, scenario, "Local PaFAR-HC", "Local PaFAR-HC", 500),
        ])
    rows = []
    for experiment, scenario, method, strategy, m0 in spec:
        d = select(data, scenario=scenario, method=method, target_m0=m0)
        threshold_mean, threshold_sd, n = mean_sd(d["threshold"])
        if not n:
            raise AssertionError(f"No finite Table 5 threshold for {scenario}/{method}/{m0}")
        rows.append({
            "Experiment": experiment,
            "Strategy": strategy,
            "Target m0": str(m0),
            "Target PFA": fmt_mean_mcse(d["pfa"], 3),
            "Sens3": fmt_mean_mcse(d["sens3"], 3),
            "PPV3": fmt_mean_mcse(d["ppv3_standardized"], 3),
            "Threshold mean (SD)": f"{threshold_mean:.3f} ({threshold_sd:.3f})",
        })
    out = pd.DataFrame(rows)
    assert len(out) == 12
    assert out.groupby("Experiment").size().eq(6).all()
    assert out.groupby("Experiment")["Target m0"].apply(list).apply(
        lambda x: x == ["0", "100", "250", "500", "500", "500"]
    ).all()
    assert not out[["Experiment", "Strategy", "Target m0"]].duplicated().any()
    assert not out.drop(columns=["Experiment", "Strategy", "Target m0"]).isna().all(axis=1).any()
    return out


def supplementary_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    learner = pd.read_csv(PRODUCTION / "learner_metrics.csv", float_precision="round_trip")
    learner_rows = []
    for scenario in ["E1", "E2", "E3"]:
        d = learner.loc[learner["scenario"].eq(scenario)]
        bi_mean, bi_sd, _ = mean_sd(d["best_iteration"])
        learner_rows.append({
            "Scenario": scenario,
            "AUROC": fmt_mean_mcse(d["auroc_weighted"], 3),
            "Trapezoidal PR-AUC": fmt_mean_mcse(d["aucpr_weighted"], 3),
            "Best iteration": f"{bi_mean:.1f} ({bi_sd:.1f})",
            "Learner failure frequency": f"{d['learner_failure'].fillna(False).astype(bool).mean():.3f}",
        })
    s1 = pd.DataFrame(learner_rows)

    hc_spec = [
        ("S1", "PaFAR-HC", None, "S1 PaFAR-HC"),
        ("S2", "PaFAR-HC", None, "S2 PaFAR-HC"),
        ("S3", "PaFAR-HC", None, "S3 PaFAR-HC"),
        ("S4", "Local PaFAR-HC", "all", "S4 Local PaFAR-HC"),
        ("S4", "PaFAR-HC", 0, "S4 source PaFAR-HC"),
    ]
    hc_rows = []
    for scenario, method, m0, label in hc_spec:
        if m0 == "all":
            d = data.loc[
                data["scenario"].eq(scenario)
                & data["method"].eq(method)
                & np.isclose(data["operating_alpha"], ALPHA)
                & data["target_m0"].isin([100.0, 250.0, 500.0])
            ].copy()
        else:
            d = select(data, scenario=scenario, method=method, target_m0=m0)
        hc_rows.append({
            "Setting": label,
            "Alpha": "0.10",
            "Delta": "0.05",
            "Exceedance frequency": f"{d['conditional_pfa_gt_alpha'].mean():.3f}",
            "Replicates": str(len(d)),
        })
    s2 = pd.DataFrame(hc_rows)

    grid_rows = []
    for scenario in ["S1", "S2", "S3", "E1", "E2"]:
        for method in ["PaFAR-F", "PaFAR-T"]:
            for alpha in [0.02, 0.05, 0.10, 0.15, 0.20]:
                d = select(data, scenario=scenario, method=method, alpha=alpha)
                pfa_mean, pfa_mcse, n = mean_mcse(d["pfa"])
                alpha_mean, alpha_mcse, _ = mean_mcse(d["alpha_m0"])
                grid_rows.append({
                    "Scenario": scenario,
                    "Method": method,
                    "Nominal alpha": alpha,
                    "Mean PFA": pfa_mean,
                    "PFA MCSE": pfa_mcse,
                    "Mean alpha_m0": alpha_mean,
                    "alpha_m0 MCSE": alpha_mcse,
                    "n defined": n,
                })
    s3 = pd.DataFrame(grid_rows)

    defined_rows = []
    defined_metrics = ["pfa", "sens3", "ppv3_standardized", "median_lead", "threshold"]
    for scenario, d in data.groupby("scenario", sort=False):
        row: dict[str, object] = {"Scenario": scenario, "Operating rows": len(d)}
        for metric in defined_metrics:
            finite = pd.to_numeric(d[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
            row[f"{metric} defined"] = int(finite.sum())
            row[f"{metric} undefined"] = int((~finite).sum())
        row["No-alert rows"] = int((d["n_alert_non_events"].fillna(0) + d["n_alert_events"].fillna(0)).eq(0).sum())
        row["No-valid-detection rows"] = int(d["n_valid3"].fillna(0).eq(0).sum())
        row["No-evaluable-denominator rows"] = int(d["n_sens3"].fillna(0).eq(0).sum())
        row["Infinite certified threshold rows"] = int(d["infinite_threshold"].fillna(False).astype(bool).sum())
        row["Learner failure rows"] = int(d["learner_failure"].fillna(False).astype(bool).sum())
        row["Structurally non-scalar threshold rows"] = int(
            (d["threshold"].isna() & d["threshold_vector"].notna()).sum()
        )
        defined_rows.append(row)
    s4 = pd.DataFrame(defined_rows)

    inf = data.loc[data["infinite_threshold"].fillna(False).astype(bool)]
    expected_inf = (
        inf["method"].eq("Local PaFAR-HC")
        & inf["target_m0"].eq(100.0)
        & np.isclose(inf["operating_alpha"], 0.02)
        & inf["scenario"].isin(["S4", "E3"])
    )
    assert len(inf) == 550 and expected_inf.all()
    assert inf.groupby("scenario").size().to_dict() == {"E3": 50, "S4": 500}
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}


def paired_contrasts(data: pd.DataFrame) -> pd.DataFrame:
    specs: list[tuple[str, str, str, int | None, str, int | None]] = []
    for scenario in ["S3", "E1", "E2"]:
        specs.append((scenario, "PaFAR-T", "PaFAR-F", None, "PaFAR-T minus PaFAR-F", None))
    for scenario in ["S1", "S2", "S3"]:
        specs.append((scenario, "PaFAR-HC", "PaFAR-F", None, "PaFAR-HC minus PaFAR-F", None))
    for scenario in ["S4", "E3"]:
        specs.extend([
            (scenario, "Local PaFAR-F", "Direct source transfer", 500, "Local PaFAR-F m0=500 minus direct transfer", 0),
            (scenario, "Local PaFAR-T", "Direct source transfer", 500, "Local PaFAR-T m0=500 minus direct transfer", 0),
            (scenario, "Local PaFAR-HC", "Local PaFAR-F", 500, "Local PaFAR-HC m0=500 minus Local PaFAR-F m0=500", 500),
        ])
    rows = []
    for scenario, method_a, method_b, m0_a, contrast, m0_b in specs:
        a = select(data, scenario=scenario, method=method_a, target_m0=m0_a)
        b = select(data, scenario=scenario, method=method_b, target_m0=m0_b)
        merged = a[["replicate", *METRICS]].merge(
            b[["replicate", *METRICS]], on="replicate", suffixes=("_a", "_b"), validate="one_to_one"
        )
        for metric in METRICS:
            diff = merged[f"{metric}_a"] - merged[f"{metric}_b"]
            diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
            n = len(diff)
            if not n:
                continue
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1)) if n > 1 else 0.0
            mcse = sd / math.sqrt(n) if n else math.nan
            rows.append({
                "scenario": scenario,
                "contrast": contrast,
                "metric": metric,
                "mean_paired_difference": mean,
                "paired_sd": sd,
                "paired_mcse": mcse,
                "mc_interval_lower": mean - 1.96 * mcse,
                "mc_interval_upper": mean + 1.96 * mcse,
                "n_pairs": n,
            })
    out = pd.DataFrame(rows)
    assert not out[["scenario", "contrast", "metric"]].duplicated().any()
    return out


def save_figure(fig: plt.Figure, stem: str, input_data: pd.DataFrame,
                sidecar: dict[str, object]) -> None:
    fig.tight_layout(pad=1.2, w_pad=1.0, h_pad=1.4)
    for directory in [FIGURE_DIR, FINAL_DIR / "figures"]:
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
        input_data.to_csv(directory / f"{stem}.csv", index=False)
        (directory / f"{stem}.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    plt.close(fig)


def figure2(data: pd.DataFrame) -> None:
    scenarios = ["S2", "S3", "E1", "E2"]
    methods = ["Pointwise-alpha", "Binwise Bonferroni", "PaFAR-F", "PaFAR-T", "PaFAR-HC"]
    records = []
    for scenario in scenarios:
        for method in methods:
            try:
                d = select(data, scenario=scenario, method=method)
            except AssertionError:
                continue
            curves = []
            for value in d["cumulative_false_alert_curve"].dropna():
                curves.append(np.asarray(json.loads(value), dtype=float))
            if not curves:
                continue
            lengths = {len(x) for x in curves}
            if len(lengths) != 1:
                raise AssertionError(f"Unequal curve lengths for {scenario}/{method}")
            arr = np.vstack(curves)
            for hour, value in enumerate(arr.mean(axis=0)):
                records.append({"scenario": scenario, "method": method, "hour": hour,
                                "mean_cumulative_false_alert_probability": value,
                                "n_defined": len(arr)})
    inp = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=False)
    for index, (ax, scenario) in enumerate(zip(axes.flat, scenarios)):
        for method in methods:
            d = inp.loc[(inp["scenario"] == scenario) & (inp["method"] == method)]
            if d.empty:
                continue
            ax.plot(d["hour"], d["mean_cumulative_false_alert_probability"],
                    label=method.replace("Binwise ", ""), color=METHOD_COLORS[method], linewidth=1.6)
        ax.set_title(scenario)
        if index >= 2:
            ax.set_xlabel("Elapsed monitoring hour")
        if index % 2 == 0:
            ax.set_ylabel("Cumulative probability of a first false alert")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    save_figure(fig, "figure2_cumulative_false_alert", inp, {
        "source": "outputs/production/all_replicate_results.csv.gz",
        "filters": {"condition": "primary", "is_alias": False, "scenarios": scenarios,
                    "operating_alpha": ALPHA},
        "methods": methods, "target_m0": None, "operating_alpha": ALPHA,
    })


def figure3(data: pd.DataFrame) -> None:
    scenarios = ["S1", "S3", "E1", "E2"]
    methods = ["Pointwise-alpha", "Binwise Bonferroni", "Naive maximum", "PaFAR-F", "PaFAR-T", "PaFAR-HC"]
    records = []
    for scenario in scenarios:
        for method in methods:
            for alpha in [0.02, 0.05, 0.10, 0.15, 0.20]:
                try:
                    d = select(data, scenario=scenario, method=method, alpha=alpha)
                except AssertionError:
                    continue
                mean, mcse, n = mean_mcse(d["pfa"])
                records.append({"scenario": scenario, "method": method, "nominal_alpha": alpha,
                                "mean_pfa": mean, "pfa_mcse": mcse, "n_defined": n})
    inp = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    for index, (ax, scenario) in enumerate(zip(axes.flat, scenarios)):
        ax.plot([0.02, 0.20], [0.02, 0.20], "--", color="black", linewidth=1.0, label="Identity")
        for method in methods:
            d = inp.loc[(inp["scenario"] == scenario) & (inp["method"] == method)].sort_values("nominal_alpha")
            if d.empty:
                continue
            ax.plot(d["nominal_alpha"], d["mean_pfa"], marker="o", markersize=3.5,
                    label=method.replace("Binwise ", ""), color=METHOD_COLORS[method], linewidth=1.4)
        ax.set_title(scenario)
        ax.set_xticks([0.02, 0.05, 0.10, 0.15, 0.20])
        ax.set_xticklabels(["0.02", "0.05", "0.10", "0.15", "0.20"], fontsize=8)
        if index >= 2:
            ax.set_xlabel("Nominal alpha")
        if index % 2 == 0:
            ax.set_ylabel("Mean patient-level PFA")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=6.8, frameon=False, ncol=2)
    save_figure(fig, "figure3_pfa_reliability", inp, {
        "source": "outputs/production/all_replicate_results.csv.gz",
        "filters": {"condition": "primary", "is_alias": False, "scenarios": scenarios,
                    "alpha_grid": [0.02, 0.05, 0.10, 0.15, 0.20]},
        "methods": methods, "target_m0": None,
        "operating_alpha": [0.02, 0.05, 0.10, 0.15, 0.20],
    })


def figure4(data: pd.DataFrame) -> None:
    scenarios = ["S3", "E2"]
    methods = ["Binwise Bonferroni", "Pointwise-alpha", "PaFAR-F", "PaFAR-T", "PaFAR-HC"]
    records = []
    for scenario in scenarios:
        for method in methods:
            for alpha in [0.05, 0.10]:
                d = select(data, scenario=scenario, method=method, alpha=alpha)
                pfa_mean, pfa_mcse, n = mean_mcse(d["pfa"])
                sens_mean, sens_mcse, _ = mean_mcse(d["sens3"])
                records.append({"scenario": scenario, "method": method, "alpha": alpha,
                                "mean_pfa": pfa_mean, "pfa_mcse": pfa_mcse,
                                "mean_sens3": sens_mean, "sens3_mcse": sens_mcse,
                                "n_defined": n})
    inp = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    markers = {0.05: "o", 0.10: "s"}
    for ax, scenario in zip(axes, scenarios):
        for method in methods:
            d = inp.loc[(inp["scenario"] == scenario) & (inp["method"] == method)].sort_values("alpha")
            ax.plot(d["mean_pfa"], d["mean_sens3"], color=METHOD_COLORS[method],
                    linewidth=1.0, label=method.replace("Binwise ", ""))
            for row in d.itertuples(index=False):
                ax.scatter(row.mean_pfa, row.mean_sens3, marker=markers[row.alpha], s=38,
                           color=METHOD_COLORS[method])
        ax.set_title(scenario)
        ax.set_xlabel("Realized patient-level PFA")
        ax.set_ylabel("Mean Sens3")
        ax.grid(alpha=0.2, linewidth=0.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([
            Line2D([], [], color="black", marker="o", linestyle="None", label="alpha=0.05"),
            Line2D([], [], color="black", marker="s", linestyle="None", label="alpha=0.10"),
        ])
        labels.extend(["alpha=0.05", "alpha=0.10"])
        ax.legend(handles, labels, fontsize=6.8, frameon=False, ncol=2)
    save_figure(fig, "figure4_sens3_vs_pfa", inp, {
        "source": "outputs/production/all_replicate_results.csv.gz",
        "filters": {"condition": "primary", "is_alias": False, "scenarios": scenarios,
                    "operating_alpha": [0.05, 0.10]},
        "methods": methods, "target_m0": None, "operating_alpha": [0.05, 0.10],
    })


def figure5(data: pd.DataFrame) -> None:
    panels = [
        [
            ("S1 PaFAR-F", "S1", "PaFAR-F", None),
            ("S1 PaFAR-HC", "S1", "PaFAR-HC", None),
            ("S2 PaFAR-F", "S2", "PaFAR-F", None),
            ("S2 PaFAR-HC", "S2", "PaFAR-HC", None),
            ("S3 PaFAR-F", "S3", "PaFAR-F", None),
            ("S3 PaFAR-HC", "S3", "PaFAR-HC", None),
        ],
        [
            ("Direct transfer", "S4", "Direct source transfer", 0),
            ("Local PaFAR-F", "S4", "Local PaFAR-F", 500),
            ("Source PaFAR-HC", "S4", "PaFAR-HC", 0),
            ("Local PaFAR-HC", "S4", "Local PaFAR-HC", 500),
        ],
    ]
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for panel_index, (ax, specs) in enumerate(zip(axes, panels), start=1):
        values = []
        labels = []
        for label, scenario, method, m0 in specs:
            d = select(data, scenario=scenario, method=method, target_m0=m0)
            v = d["conditional_pfa_oracle"].dropna().to_numpy(float)
            values.append(v)
            labels.append(label)
            records.extend({"panel": panel_index, "setting": label, "scenario": scenario,
                            "method": method, "target_m0": m0, "replicate": int(rep),
                            "conditional_pfa_oracle": float(value)}
                           for rep, value in zip(d.loc[d["conditional_pfa_oracle"].notna(), "replicate"], v))
        ax.boxplot(values, tick_labels=labels, showfliers=False,
                   medianprops={"color": "#d95f02"})
        ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("A. Exchangeable settings" if panel_index == 1 else "B. S4 target shift")
        ax.set_ylabel("Independent-oracle conditional PFA")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    inp = pd.DataFrame(records)
    assert set(inp.loc[inp["method"].str.startswith("Local"), "target_m0"].dropna()) == {500}
    save_figure(fig, "figure5_conditional_pfa", inp, {
        "source": "outputs/production/all_replicate_results.csv.gz",
        "filters": {"experiment": "Experiment I", "condition": "primary", "is_alias": False,
                    "operating_alpha": ALPHA},
        "methods": sorted(inp["method"].unique().tolist()),
        "target_m0": {"source_or_direct": 0, "local": 500}, "operating_alpha": ALPHA,
    })


def figure6(data: pd.DataFrame) -> None:
    scenarios = ["S4", "E3"]
    methods = ["Direct source transfer", "Local PaFAR-F", "Local PaFAR-T", "Local PaFAR-HC"]
    records = []
    for scenario in scenarios:
        for method in methods:
            m0s = [0] if method == "Direct source transfer" else [100, 250, 500]
            for m0 in m0s:
                d = select(data, scenario=scenario, method=method, target_m0=m0)
                for metric in ["pfa", "sens3"]:
                    mean, mcse, n = mean_mcse(d[metric])
                    records.append({"scenario": scenario, "strategy": method,
                                    "target_m0": m0, "metric": metric, "mean": mean,
                                    "mcse": mcse, "n_defined": n})
    inp = pd.DataFrame(records)
    assert set(inp["strategy"]) == set(methods)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex="col")
    for col, scenario in enumerate(scenarios):
        for row, metric in enumerate(["pfa", "sens3"]):
            ax = axes[row, col]
            for method in methods:
                d = inp.loc[(inp["scenario"] == scenario) & (inp["metric"] == metric)
                            & (inp["strategy"] == method)].sort_values("target_m0")
                marker = "*" if method == "Direct source transfer" else "o"
                ax.plot(d["target_m0"], d["mean"], marker=marker,
                        linestyle="None" if method == "Direct source transfer" else "-",
                        label=method.replace("Local ", ""), color=METHOD_COLORS[method], linewidth=1.5)
            if metric == "pfa":
                ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1.0)
            ax.set_title(scenario if row == 0 else "")
            ax.set_xlabel("Target non-event calibration size")
            ax.set_ylabel("Target PFA" if metric == "pfa" else "Mean Sens3")
            ax.grid(alpha=0.2, linewidth=0.5)
            ax.legend(fontsize=7, frameon=False)
    save_figure(fig, "figure6_target_recalibration", inp, {
        "source": "outputs/production/all_replicate_results.csv.gz",
        "filters": {"condition": "primary", "is_alias": False, "scenarios": scenarios,
                    "operating_alpha": ALPHA},
        "methods": methods, "target_m0": [0, 100, 250, 500], "operating_alpha": ALPHA,
    })


def numerical_summary(data: pd.DataFrame, checks: dict[str, float]) -> dict[str, object]:
    summary: dict[str, object] = {"sanity_checks": checks}
    def finite_or_none(value: float) -> float | None:
        return value if math.isfinite(value) else None
    for scenario in ["S1", "S2", "S3", "E1", "E2"]:
        for method in ["Pointwise-alpha", "PaFAR-F", "PaFAR-T", "PaFAR-HC"]:
            try:
                d = select(data, scenario=scenario, method=method)
            except AssertionError:
                continue
            summary[f"{scenario}_{method}"] = {
                metric: finite_or_none(mean_mcse(d[metric])[0]) for metric in METRICS
            } | {"alpha_m0": finite_or_none(mean_mcse(d["alpha_m0"])[0])}
    summary["production_sha256"] = sha256(SOURCE_RESULTS)
    summary["sensitivity_simulations_run"] = False
    return summary


def mirror(path: Path, relative: Path) -> None:
    destination = FINAL_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def build(data: pd.DataFrame, checks: dict[str, float]) -> None:
    for directory in [TABLE_DIR, FIGURE_DIR, SUPPLEMENT_DIR, POSTPROCESS_DIR,
                      FINAL_DIR / "tables", FINAL_DIR / "figures", FINAL_DIR / "supplementary"]:
        directory.mkdir(parents=True, exist_ok=True)

    t3, t4, t5 = table3(data), table4(data), table5(data)
    tables = [
        (t3, "table3_experiment1", "Experiment I primary results at $\\alpha=0.10$ and $m_0=500$.",
         "tab:exp1-primary", [
             "Entries are Monte Carlo means with Monte Carlo standard errors in parentheses.",
             "Long-stay PFA is descriptive and is not a separately guaranteed target.",
         ], "llrrrrrr"),
        (t4, "table4_experiment2", "Experiment II end-to-end XGBoost results at $\\alpha=0.10$.",
         "tab:exp2-primary", [
             "Entries are Monte Carlo means with Monte Carlo standard errors in parentheses; threshold SD is the across-replicate standard deviation.",
             "Threshold-free AUROC and trapezoidal PR-AUC are reported once per learner in a supplementary table.",
             "Fixed-boundary thresholds and PaFAR-T standardized thresholds are on different numerical scales and are not directly comparable.",
         ], "llrrrrrr"),
        (t5, "table5_target_shift", "Target-site shift and local threshold recalibration.",
         "tab:target-recalibration", [
             "PFA, Sens3, and PPV3 are Monte Carlo means with Monte Carlo standard errors in parentheses; thresholds are means with across-replicate SDs in parentheses.",
             "Direct source-transfer, PaFAR-F, and PaFAR-HC thresholds are on the transformed-logit scale.",
             "Local PaFAR-T thresholds are on the standardized time-template scale. Threshold values on different scales are not directly comparable.",
         ], "llrrrrr"),
    ]
    for frame, stem, caption, label, notes, fmt in tables:
        csv_path = TABLE_DIR / f"{stem}.csv"
        tex_path = TABLE_DIR / f"{stem}.tex"
        frame.to_csv(csv_path, index=False)
        write_tex_table(frame, tex_path, caption=caption, label=label, notes=notes,
                        column_format=fmt, resize=True)
        mirror(csv_path, Path("tables") / csv_path.name)
        mirror(tex_path, Path("tables") / tex_path.name)

    supplements = supplementary_tables(data)
    supplement_specs = {
        "S1": ("tableS1_learner_metrics", "Experiment II learner metrics.", "tab:learner-metrics",
               ["AUROC and trapezoidal PR-AUC are threshold-free learner metrics. Entries in parentheses are Monte Carlo SEs, except best iteration, which reports across-replicate SD."], "lrrrr"),
        "S2": ("tableS2_hc_exceedance", "PaFAR-HC conditional PFA exceedance frequency at $\\alpha=0.10$.", "tab:hc-exceedance",
               ["The tolerance-limit setting is $\\delta=0.05$. The source PaFAR-HC threshold transferred to S4 is not protected by the exchangeability guarantee."], "lrrrr"),
        "S3": ("tableS3_alpha_grid", "PaFAR-F/T reliability over the prespecified alpha grid.", "tab:alpha-grid",
               ["The five operating points reuse the same primary replicate trajectories, learner, template, calibration sample, and test sample; only the threshold is recomputed."], "llrrrrrr"),
        "S4": ("tableS4_definedness", "Defined, undefined, infinite-threshold, and learner-failure diagnostics.", "tab:definedness",
               ["Diagnostic categories are descriptive and may overlap. Structural non-applicability, no alerts, no valid detection, no evaluable denominator, an infinite certified threshold, and learner failure are distinct states; undefined cells are not failures.",
                "All 550 infinite-threshold operating rows are Local PaFAR-HC at target $m_0=100$ and $\\alpha=0.02$: 500 S4 rows and 50 E3 rows."], "l" + "r" * (len(supplements["S4"].columns) - 1)),
    }
    for key, frame in supplements.items():
        stem, caption, label, notes, fmt = supplement_specs[key]
        csv_path = SUPPLEMENT_DIR / f"{stem}.csv"
        tex_path = SUPPLEMENT_DIR / f"{stem}.tex"
        frame.to_csv(csv_path, index=False)
        display = frame.copy()
        if key == "S3":
            for col in ["Nominal alpha", "Mean PFA", "PFA MCSE", "Mean alpha_m0", "alpha_m0 MCSE"]:
                display[col] = display[col].map(lambda x: f"{float(x):.3f}")
        write_tex_table(display, tex_path, caption=caption, label=label, notes=notes,
                        column_format=fmt, resize=True)
        mirror(csv_path, Path("supplementary") / csv_path.name)
        mirror(tex_path, Path("supplementary") / tex_path.name)

    paired = paired_contrasts(data)
    paired_csv = SUPPLEMENT_DIR / "paired_contrasts.csv"
    paired.to_csv(paired_csv, index=False)
    paired_display = paired.copy()
    for col in ["mean_paired_difference", "paired_sd", "paired_mcse",
                "mc_interval_lower", "mc_interval_upper"]:
        paired_display[col] = paired_display[col].map(lambda x: f"{x:.4f}")
    write_tex_table(
        paired_display,
        SUPPLEMENT_DIR / "paired_contrasts.tex",
        caption="Paired Monte Carlo contrasts at $\\alpha=0.10$.",
        label="tab:paired-contrasts",
        notes=["Intervals are mean paired differences $\\pm 1.96$ paired Monte Carlo SE and quantify simulation precision. They are not patient-level confidence intervals or significance tests."],
        column_format="lllrrrrrr",
        resize=True,
    )
    mirror(paired_csv, Path("supplementary") / paired_csv.name)
    mirror(SUPPLEMENT_DIR / "paired_contrasts.tex", Path("supplementary") / "paired_contrasts.tex")

    figure2(data)
    figure3(data)
    figure4(data)
    figure5(data)
    figure6(data)

    summary = numerical_summary(data, checks)
    summary_path = POSTPROCESS_DIR / "primary_numerical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    mirror(summary_path, Path("primary_numerical_summary.json"))
    paired_summary = paired.groupby(["scenario", "contrast"], as_index=False).size()
    paired_summary.to_csv(POSTPROCESS_DIR / "paired_contrast_inventory.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Run locked-result sanity checks without writing outputs.")
    args = parser.parse_args()
    data = load_results()
    checks = sanity_checks(data)
    print(json.dumps({"status": "sanity_checks_passed", "values": checks}, indent=2, sort_keys=True))
    if args.check_only:
        return
    build(data, checks)
    print(json.dumps({"status": "manuscript_outputs_built", "output": str(FINAL_DIR)}, indent=2))


if __name__ == "__main__":
    main()
