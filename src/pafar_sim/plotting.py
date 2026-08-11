"""Aggregated manuscript tables and scenario-explicit figures."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregation import aggregate_learner_metrics, aggregate_results
from .io_utils import atomic_write_csv, atomic_write_json


def _save(fig: plt.Figure, path: Path, source: Path, filters: dict[str, object], scenarios: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    csv_sidecar = path.with_suffix(".csv")
    if source.resolve() != csv_sidecar.resolve():
        shutil.copyfile(source, csv_sidecar)
    atomic_write_json(
        {"input": str(source.resolve()), "filters": filters, "scenarios": scenarios, "aggregation": "replicate mean unless stated"},
        path.with_suffix(".json"),
    )


def _selected_condition(data: pd.DataFrame) -> str:
    conditions = set(data.get("condition", pd.Series(["primary"])).dropna().astype(str))
    return "primary" if "primary" in conditions else sorted(conditions)[0]


def _format_estimate(mean: float, mcse: float) -> str:
    return "NA" if not np.isfinite(mean) else (f"{mean:.6f}" if not np.isfinite(mcse) else f"{mean:.6f} ({mcse:.6f})")


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    """Write a dependency-free tabular representation with escaped text."""
    def escape(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
            text = text.replace(old, new)
        return text
    columns = list(frame.columns)
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(escape(x) for x in columns) + r" \\", r"\hline"]
    lines.extend(" & ".join(escape(value) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _table_from_summary(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    identifiers = [c for c in ("scenario", "method", "target_m0", "calibration_m0", "operating_alpha") if c in summary]
    selected = summary[summary.metric.isin(metrics + ["threshold"])].copy()
    selected["display"] = [
        ((f"{row.threshold_sd:.6f}" if np.isfinite(getattr(row, "threshold_sd", np.nan)) else "NA")
         if row.metric == "threshold" else _format_estimate(float(row.mean), float(row.mcse)))
        for row in selected.itertuples()
    ]
    selected.loc[selected.metric == "threshold", "metric"] = "threshold_sd"
    if selected.empty:
        return pd.DataFrame(columns=identifiers + metrics + ["threshold_sd"])
    estimate = selected.pivot_table(index=identifiers, columns="metric", values="display", aggfunc="first", dropna=False).reset_index()
    counts = selected.pivot_table(index=identifiers, columns="metric", values="n_defined", aggfunc="first", dropna=False).reset_index()
    counts = counts.rename(columns={column: f"{column}_n_defined" for column in counts.columns if column not in identifiers})
    frequencies = selected.groupby(identifiers, dropna=False, as_index=False).agg(
        infinite_threshold_frequency=("infinite_threshold_frequency", "first"),
        learner_failure_frequency=("learner_failure_frequency", "first"),
        n_total=("n_total", "first"),
    )
    return estimate.merge(counts, on=identifiers, how="left").merge(frequencies, on=identifiers, how="left")


def make_tables_and_figures(results_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate Tables 3--5 from aggregate rows and Figures 2--6 from explicit scenarios."""
    source, out = Path(results_path), Path(output_dir)
    data = pd.read_csv(source, float_precision="round_trip")
    if "is_alias" in data:
        data = data.loc[~data.is_alias.fillna(False).astype(bool)].copy()
    condition = _selected_condition(data)
    operating = pd.to_numeric(data.get("operating_alpha", data.get("alpha")), errors="coerce")
    data["operating_alpha"] = operating
    summary = aggregate_results(data)
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"; figures = out / "figures"; inputs = out / "figure_inputs"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True); inputs.mkdir(exist_ok=True)
    atomic_write_csv(summary, out / "summary_long.csv")
    outputs: list[Path] = [out / "summary_long.csv"]

    primary = summary[(summary.condition.astype(str) == condition) & np.isclose(pd.to_numeric(summary.operating_alpha), .10)]
    exp1 = primary[primary.experiment == "Experiment I"]
    if "calibration_m0" in exp1 and not exp1.empty:
        available = pd.to_numeric(exp1.calibration_m0, errors="coerce")
        chosen = 500 if np.isclose(available, 500).any() else float(np.nanmax(available))
        exp1 = exp1[np.isclose(pd.to_numeric(exp1.calibration_m0, errors="coerce"), chosen)]
    table3 = _table_from_summary(exp1, ["pfa", "sens3", "sens0", "ppv3_standardized", "alert_burden_100d"])
    path = tables / "table3_experiment1.csv"; atomic_write_csv(table3, path); outputs.append(path)
    _write_latex(table3, path.with_suffix(".tex")); outputs.append(path.with_suffix(".tex"))

    table4_source = primary[(primary.experiment == "Experiment II") & primary.scenario.isin(["E1", "E2"])]
    table4 = _table_from_summary(table4_source, ["pfa", "sens3", "premature", "median_lead", "ppv3_standardized"])
    path = tables / "table4_experiment2_e1_e2.csv"; atomic_write_csv(table4, path); outputs.append(path)
    _write_latex(table4, path.with_suffix(".tex")); outputs.append(path.with_suffix(".tex"))

    target = primary[primary.scenario.isin(["S4", "E3"])].copy()
    m0 = pd.to_numeric(target.get("target_m0"), errors="coerce")
    keep = ((target.method == "Direct source transfer") & (m0 == 0))
    keep |= ((target.method == "Local PaFAR-F") & m0.isin([100, 250, 500]))
    keep |= ((target.method == "Local PaFAR-T") & (m0 == 500))
    keep |= ((target.method == "Local PaFAR-HC") & (m0 == 500))
    table5 = _table_from_summary(target[keep], ["pfa", "sens3", "sens0", "ppv3_standardized", "alert_burden_100d"])
    path = tables / "table5_target_shift.csv"; atomic_write_csv(table5, path); outputs.append(path)
    _write_latex(table5, path.with_suffix(".tex")); outputs.append(path.with_suffix(".tex"))

    learner_path = out / "learner_metrics.csv"
    if learner_path.exists():
        learner = pd.read_csv(learner_path, float_precision="round_trip")
        learner_summary = aggregate_learner_metrics(learner)
        path = tables / "supplementary_learner_metrics.csv"
        atomic_write_csv(learner_summary, path); outputs.append(path)
        atomic_write_csv(learner_summary, out / "learner_metrics_summary.csv"); outputs.append(out / "learner_metrics_summary.csv")

    scenarios = sorted(data.scenario.dropna().astype(str).unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, max(3.2, 2.8 * len(scenarios))), squeeze=False)
    curve_input = []
    for ax, scenario in zip(axes[:, 0], scenarios):
        block = data[(data.scenario == scenario) & np.isclose(data.operating_alpha, .10)]
        for method, group in block.groupby("method"):
            curves = []
            for value in group.get("cumulative_false_alert_curve", pd.Series(dtype=object)).dropna():
                try: curves.append(np.asarray(json.loads(value), dtype=float))
                except (json.JSONDecodeError, TypeError): pass
            if curves:
                mean_curve = np.nanmean(curves, axis=0)
                ax.plot(np.arange(1, len(mean_curve) + 1), mean_curve, label=method)
                curve_input.extend({"scenario": scenario, "method": method, "hour": j + 1, "mean": y} for j, y in enumerate(mean_curve))
        ax.set_title(scenario); ax.set(xlabel="Elapsed hour", ylabel="Cumulative false-alert probability"); ax.legend(fontsize=6, ncol=3)
    curve_input_path = inputs / "figure2.csv"; atomic_write_csv(pd.DataFrame(curve_input), curve_input_path)
    path = figures / "figure2_cumulative_false_alert.png"; _save(fig, path, curve_input_path, {"operating_alpha": .10}, scenarios); outputs += [curve_input_path, path]

    alpha_data = data[data.method.isin(["Pointwise-alpha", "Binwise Bonferroni", "Naive maximum", "PaFAR-F", "Direct source transfer", "PaFAR-T", "PaFAR-HC"])].copy()
    alpha_means = alpha_data.groupby(["scenario", "method", "operating_alpha"], as_index=False).pfa.mean()
    alpha_input = inputs / "figure3.csv"; atomic_write_csv(alpha_means, alpha_input)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(7, max(3.2, 2.8 * len(scenarios))), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        for method, group in alpha_means[alpha_means.scenario == scenario].groupby("method"):
            ax.plot(group.operating_alpha, group.pfa, "o-", label=method)
        ax.plot([.02, .20], [.02, .20], "k--"); ax.set_title(scenario); ax.set(xlabel="Nominal alpha", ylabel="Mean PFA"); ax.legend(fontsize=6, ncol=3)
    path = figures / "figure3_pfa_vs_alpha.png"; _save(fig, path, alpha_input, {"alpha_grid": [.02, .05, .10, .15, .20]}, scenarios); outputs += [alpha_input, path]

    points = data[data.operating_alpha.isin([.05, .10])].groupby(["scenario", "method", "operating_alpha"], as_index=False)[["pfa", "sens3"]].mean()
    point_input = inputs / "figure4.csv"; atomic_write_csv(points, point_input)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for alpha, ax in zip((.05, .10), axes):
        block = points[np.isclose(points.operating_alpha, alpha)]
        for scenario, group in block.groupby("scenario"):
            ax.scatter(group.pfa, group.sens3, label=scenario)
        ax.set_title(f"alpha={alpha:.2f}"); ax.set(xlabel="Mean PFA", ylabel="Mean Sens3")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    path = figures / "figure4_sens3_vs_pfa.png"; _save(fig, path, point_input, {"operating_alpha": [.05, .10]}, scenarios); outputs += [point_input, path]

    conditional = data[data.method.isin(["PaFAR-F", "Direct source transfer", "Local PaFAR-F", "PaFAR-HC", "Local PaFAR-HC"]) & np.isclose(data.operating_alpha, .10)]
    cond_input = inputs / "figure5.csv"; atomic_write_csv(conditional[[c for c in ["scenario", "method", "replicate", "conditional_pfa_oracle"] if c in conditional]], cond_input)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, family in zip(axes, ("F", "HC")):
        block = conditional[~conditional.method.str.contains("HC")] if family == "F" else conditional[conditional.method.str.contains("HC")]
        grouped = [(f"{s}:{m}", g.conditional_pfa_oracle.dropna().to_numpy()) for (s, m), g in block.groupby(["scenario", "method"]) if g.conditional_pfa_oracle.notna().any()]
        if grouped: ax.boxplot([x[1] for x in grouped], tick_labels=[x[0] for x in grouped]); ax.tick_params(axis="x", rotation=45)
        ax.axhline(.10, color="black", linestyle="--"); ax.set_title(f"PaFAR-{family}"); ax.set(ylabel="Conditional PFA")
    path = figures / "figure5_conditional_pfa.png"; _save(fig, path, cond_input, {"operating_alpha": .10, "panels": ["F", "HC"]}, sorted(conditional.scenario.unique())); outputs += [cond_input, path]

    target_points = data[data.scenario.isin(["S4", "E3"]) & np.isclose(data.operating_alpha, .10)].groupby(["scenario", "method", "target_m0"], as_index=False)[["pfa", "sens3"]].mean()
    target_input = inputs / "figure6.csv"; atomic_write_csv(target_points, target_input)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, scenario in zip(axes, ("S4", "E3")):
        for method, group in target_points[target_points.scenario == scenario].groupby("method"):
            ax.plot(group.target_m0, group.pfa, "o-", label=f"{method} PFA")
            ax.plot(group.target_m0, group.sens3, "x--", label=f"{method} Sens3")
        ax.set_title(scenario); ax.set(xlabel="Target calibration m0", ylabel="Mean operating metric"); ax.legend(fontsize=6)
    path = figures / "figure6_target_recalibration.png"; _save(fig, path, target_input, {"operating_alpha": .10, "panels": ["S4", "E3"]}, ["S4", "E3"]); outputs += [target_input, path]
    return outputs
