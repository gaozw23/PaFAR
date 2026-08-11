"""Tables 6--7 and review-bundle assembly from locked real-data summaries."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import zipfile

import numpy as np
import pandas as pd

from pafar_sim.io_utils import atomic_write_csv
from .internal_analysis import METHODS
from .schema import RealDataConfig


def _fmt(value: float, digits: int=3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def build_table6(config: RealDataConfig) -> pd.DataFrame:
    source=pd.read_csv(config.outputs/"internal_primary"/"internal_results_summary.csv")
    table=source[np.isclose(source.alpha,.10)].copy(); table["_order"]=table.method.map({m:i for i,m in enumerate(METHODS)}); table=table.sort_values("_order")
    out=pd.DataFrame({"Method":table.method,"Patient PFA":[f"{_fmt(x)} [{_fmt(lo)}, {_fmt(hi)}]" for x,lo,hi in zip(table.pfa,table.pfa_lower_95,table.pfa_upper_95)],
                      "Sens3":table.sens3.map(_fmt),"Sens0":table.sens0.map(_fmt),"Premature":table.premature.map(_fmt),"Median lead":table.median_lead.map(lambda x:_fmt(x,1)),
                      "PPV3":table.ppv3.map(_fmt),"Alerts/100 d":table.alerts_per_100d.map(lambda x:_fmt(x,1)),"Utility":table.utility.map(_fmt)})
    path=config.root/"literature"/"generated_tables"; atomic_write_csv(out,path/"table6_physionet_internal.csv")
    lines=["\\begin{table}[!htbp]","\\centering","\\caption{Internal PhysioNet test performance at target patient-level false-alarm probability $\\alpha=0.10$.}","\\label{tab:real-internal}","\\resizebox{\\linewidth}{!}{%","\\begin{tabular}{lcccccccc}","\\toprule","Method & Patient PFA [95\\% bootstrap interval] & Sens$_3$ & Sens$_0$ & Premature & Median lead & PPV$_3$ & Alerts/100 d & Utility \\\\","\\midrule"]
    for row in out.itertuples(index=False): lines.append(" & ".join(str(x).replace("%","\\%") for x in row)+" \\\\")
    lines += ["\\bottomrule","\\end{tabular}%","}","\\begin{minipage}{0.98\\linewidth}\\footnotesize Overall PFA intervals use the hospital-stratified whole-patient bootstrap; hospital-specific exact intervals are reported in the supplement. Utility is the official normalized Challenge utility.\\end{minipage}","\\end{table}"]
    (path/"table6_physionet_internal.tex").write_text("\n".join(lines)+"\n",encoding="utf-8"); return out


def build_table7(config: RealDataConfig) -> pd.DataFrame:
    source=pd.read_csv(config.outputs/"transfer_primary"/"transfer_results_summary.csv")
    selected=[]
    for direction in ("A→B","B→A"):
        selected.append(source[(source.direction==direction)&(source.strategy=="Direct source transfer")&(source.target_m0==0)])
        selected.append(source[(source.direction==direction)&(source.strategy=="Local PaFAR-F")&source.target_m0.isin([100,250,500,1000])])
        selected.append(source[(source.direction==direction)&(source.strategy=="Local PaFAR-T")&(source.target_m0==500)])
        selected.append(source[(source.direction==direction)&(source.strategy=="Local PaFAR-HC")&(source.target_m0==500)])
    table=pd.concat(selected,ignore_index=True)
    out=pd.DataFrame({"Direction":table.direction,"Strategy":table.strategy,"Target m0":table.target_m0.astype(int),
                      "Target PFA":[f"{_fmt(x)} [{_fmt(lo)}, {_fmt(hi)}]" for x,lo,hi in zip(table.pfa,table.pfa_lower_95,table.pfa_upper_95)],
                      "Sens3":table.sens3.map(_fmt),"PPV3":table.ppv3.map(_fmt),"Utility":table.utility.map(_fmt)})
    path=config.root/"literature"/"generated_tables"; atomic_write_csv(out,path/"table7_cross_hospital.csv")
    lines=["\\begin{table}[!htbp]","\\centering","\\caption{Cross-hospital deployment and local scalar-threshold recalibration at $\\alpha=0.10$.}","\\label{tab:real-external}","\\resizebox{\\linewidth}{!}{%","\\begin{tabular}{llccccc}","\\toprule","Direction & Strategy & Target $m_0$ & Target PFA [95\\% exact interval] & Sens$_3$ & PPV$_3$ & Utility \\\\","\\midrule"]
    for row in out.itertuples(index=False): lines.append(" & ".join(str(x).replace("→","$\\rightarrow$") for x in row)+" \\\\")
    lines += ["\\bottomrule","\\end{tabular}%","}","\\begin{minipage}{0.98\\linewidth}\\footnotesize Target PFA intervals are two-sided Clopper--Pearson intervals. Direct transfer uses the frozen source PaFAR-F threshold; local methods change only the scalar threshold.\\end{minipage}","\\end{table}"]
    (path/"table7_cross_hospital.tex").write_text("\n".join(lines)+"\n",encoding="utf-8"); return out


def build_manuscript_outputs(config: RealDataConfig) -> dict[str,object]:
    mpl=config.outputs/".mplconfig"; mpl.mkdir(parents=True,exist_ok=True); os.environ["MPLCONFIGDIR"]=str(mpl)
    from .plotting import make_all_figures
    table6=build_table6(config); table7=build_table7(config); make_all_figures(config)
    table_out=config.outputs/"tables"; figure_out=config.outputs/"figures"; table_out.mkdir(parents=True,exist_ok=True); figure_out.mkdir(parents=True,exist_ok=True)
    for stem in ("table6_physionet_internal","table7_cross_hospital"):
        for suffix in (".csv",".tex"):
            shutil.copy2(config.root/"literature"/"generated_tables"/(stem+suffix),table_out/(stem+suffix))
    for stem in ("figure7_cohort_flow","figure8_realdata_false_alert","figure9_realdata_tradeoff","figure10_cross_hospital"):
        for suffix in (".pdf",".png",".csv",".json"):
            shutil.copy2(config.root/"literature"/"figures"/(stem+suffix),figure_out/(stem+suffix))
    return {"table6_rows":len(table6),"table7_rows":len(table7)}


def build_review_bundle(config: RealDataConfig, pdf: Path) -> Path:
    bundle=config.outputs/"REALDATA_REVIEW_BUNDLE.zip"
    include=[config.outputs/"qc",config.outputs/"feature_smoke",config.outputs/"internal_primary",config.outputs/"transfer_primary",config.outputs/"bootstrap",config.outputs/"robustness",config.outputs/"tables",config.outputs/"figures",config.outputs/"manifests",
             config.root/"configs"/"realdata_primary.yaml",config.root/"configs"/"realdata_robustness.yaml",config.outputs/"REALDATA_LOCK.json",config.outputs/"REALDATA_LOCK_V2.json",config.outputs/"REALDATA_LOCK_V2_1.json",config.outputs/"REALDATA_FINAL_REPORT.md",config.root/"literature"/"PaFAR.tex",pdf,
             config.root/"src"/"pafar_sim"/"realdata",config.root/"tests"/"realdata"]
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for item in include:
            if not item.exists(): continue
            paths=[item] if item.is_file() else sorted(p for p in item.rglob("*") if p.is_file())
            for path in paths:
                rel=path.relative_to(config.root).as_posix()
                if rel.startswith("data/physionet2019/raw/"): raise RuntimeError("raw PhysioNet data attempted to enter review bundle")
                archive.write(path,rel)
        for stem in ("table6_physionet_internal","table7_cross_hospital"):
            for suffix in (".csv",".tex"):
                path=config.root/"literature"/"generated_tables"/(stem+suffix); archive.write(path,path.relative_to(config.root).as_posix())
        for stem in ("figure7_cohort_flow","figure8_realdata_false_alert","figure9_realdata_tradeoff","figure10_cross_hospital"):
            for suffix in (".pdf",".png",".csv",".json"):
                path=config.root/"literature"/"figures"/(stem+suffix); archive.write(path,path.relative_to(config.root).as_posix())
    return bundle
