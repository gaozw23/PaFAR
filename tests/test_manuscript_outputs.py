"""Artifact assertions for the filled PRIMARY-results manuscript."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(LIT / "generated_tables" / name)


def test_primary_table_shapes_keys_and_defined_rows() -> None:
    t3 = _table("table3_experiment1.csv")
    t4 = _table("table4_experiment2.csv")
    t5 = _table("table5_target_shift.csv")
    assert (len(t3), len(t4), len(t5)) == (12, 8, 12)
    assert not t3[["Scenario", "Method"]].duplicated().any()
    assert not t4[["Scenario", "Method"]].duplicated().any()
    assert not t5[["Experiment", "Strategy", "Target m0"]].duplicated().any()
    for table, identifiers in [
        (t3, ["Scenario", "Method"]),
        (t4, ["Scenario", "Method"]),
        (t5, ["Experiment", "Strategy", "Target m0"]),
    ]:
        metrics = table.drop(columns=identifiers)
        assert not metrics.isna().all(axis=1).any()
        assert not metrics.astype(str).apply(
            lambda row: row.str.fullmatch(r"(?:--|nan|inf)", case=False).all(), axis=1
        ).any()
        assert "replicate" not in {column.lower() for column in table.columns}


def test_primary_table_filters() -> None:
    t3 = _table("table3_experiment1.csv")
    t4 = _table("table4_experiment2.csv")
    t5 = _table("table5_target_shift.csv")
    assert "S4" not in set(t3["Scenario"])
    assert not set(t4["Method"]) & {"Fixed 0.5", "Naive maximum", "Binwise Bonferroni"}
    assert not ((t4["Scenario"] == "E1") & (t4["Method"] == "PaFAR-HC")).any()
    assert t5.groupby("Experiment").size().to_dict() == {"Experiment I": 6, "Experiment II": 6}
    expected_m0 = [0, 100, 250, 500, 500, 500]
    assert t5.groupby("Experiment")["Target m0"].apply(list).apply(
        lambda values: values == expected_m0
    ).all()


def test_figure_filters_are_prespecified() -> None:
    f5 = pd.read_csv(LIT / "figures" / "figure5_conditional_pfa.csv")
    local = f5[f5["method"].str.startswith("Local")]
    assert set(local["target_m0"].dropna()) == {500.0}
    panel_b = f5[f5["panel"] == 2]
    observed = set(zip(panel_b["method"], panel_b["target_m0"]))
    assert observed == {
        ("Direct source transfer", 0.0),
        ("Local PaFAR-F", 500.0),
        ("PaFAR-HC", 0.0),
        ("Local PaFAR-HC", 500.0),
    }
    f6 = pd.read_csv(LIT / "figures" / "figure6_target_recalibration.csv")
    assert set(f6["strategy"]) == {
        "Direct source transfer", "Local PaFAR-F", "Local PaFAR-T", "Local PaFAR-HC"
    }


def test_tex_has_no_simulation_placeholders_or_primary_table_dashes() -> None:
    tex = (LIT / "PaFAR.tex").read_text(encoding="utf-8")
    stale = [
        "Planned simulation outputs",
        "Simulation/analysis output placeholder",
        "Insert the final reproducible figure here after the code has been run",
        "Numerical result fields are left blank",
        "execute the simulation",
        "populate the reserved tables",
        "remaining work is computational",
        "TODO",
        "TBD",
        "NaN",
    ]
    assert all(item not in tex for item in stale)
    for name in ["table3_experiment1.tex", "table4_experiment2.tex", "table5_target_shift.tex"]:
        table_tex = (LIT / "generated_tables" / name).read_text(encoding="utf-8")
        assert "\\resultcell" not in table_tex
        assert not re.search(r"(?<!-)--(?!-)", table_tex)


def test_production_hashes_counts_and_mtime_are_unchanged() -> None:
    manifest = json.loads((LIT / "manuscript_data_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["before_sha256"].items():
        if relative.startswith("outputs/production/"):
            assert _sha256(ROOT / relative) == expected
    raw = ROOT / "outputs" / "production" / "raw"
    checkpoints = sorted(raw.glob("**/rep_*.csv.gz"))
    sidecars = sorted(raw.glob("**/rep_*.csv.gz.json"))
    all_files = checkpoints + sidecars
    before = manifest["production_raw_before"]
    assert len(checkpoints) == before["checkpoint_count"] == 2250
    assert len(sidecars) == before["sidecar_count"] == 2250
    assert sum(path.stat().st_size for path in all_files) == before["total_bytes"]
    assert max(path.stat().st_mtime_ns for path in all_files) == before["latest_mtime_ns"]


def test_manuscript_was_modified_and_final_pdf_compiled() -> None:
    manifest = json.loads((LIT / "manuscript_data_manifest.json").read_text(encoding="utf-8"))
    assert _sha256(LIT / "PaFAR.tex") != manifest["before_sha256"]["literature/PaFAR.tex"]
    pdf = LIT / "PaFAR_results_filled.pdf"
    log = LIT / "PaFAR_results_filled.log"
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert log.is_file() and log.stat().st_size > 0
    log_text = log.read_text(encoding="latin-1")
    forbidden = [
        "! LaTeX Error",
        "Undefined control sequence",
        "There were undefined citations",
        "There were undefined references",
        "multiply defined",
        "File `figures/",
    ]
    assert all(item not in log_text for item in forbidden)

