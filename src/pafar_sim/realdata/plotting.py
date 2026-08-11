"""Manuscript Figures 7--10 with CSV inputs and JSON provenance sidecars."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pafar_sim.io_utils import atomic_write_csv, atomic_write_json, file_checksum
from .schema import RealDataConfig


COLORS={"Pointwise-alpha":"#d62728","Bonferroni":"#9467bd","PaFAR-F":"#2ca02c","PaFAR-T":"#1f77b4","PaFAR-HC":"#000000","Fixed 0.5":"#ff7f0e","Youden":"#8c564b","Direct source transfer":"#333333","Local PaFAR-F":"#2ca02c","Local PaFAR-T":"#1f77b4","Local PaFAR-HC":"#d62728"}


def _save(fig, base: Path, data: pd.DataFrame, metadata: dict[str,object]) -> None:
    base.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"),dpi=220,bbox_inches="tight")
    atomic_write_csv(data,base.with_suffix(".csv"))
    atomic_write_json({**metadata,"input_rows":len(data),"input_csv_sha256":file_checksum(base.with_suffix(".csv"))},base.with_suffix(".json"))
    plt.close(fig)


def figure7(config: RealDataConfig) -> None:
    manifest=pd.read_csv(config.data_root/"processed"/"patient_manifest.csv.gz",float_precision="round_trip")
    split=pd.read_csv(config.data_root/"processed"/"internal_primary_split.csv")
    rows=[]
    for hospital in ("A","B"):
        block=manifest[manifest.hospital_set==hospital]
        rows += [{"panel":"flow","hospital":hospital,"category":"raw","value":len(block)},
                 {"panel":"flow","hospital":hospital,"category":"primary non-event","value":int((block.primary_cohort & ~block.any_sepsis_label).sum())},
                 {"panel":"flow","hospital":hospital,"category":"primary event","value":int((block.primary_cohort & block.any_sepsis_label).sum())},
                 {"panel":"flow","hospital":hospital,"category":"below landmark","value":int(block.exclusion_reason.fillna("").str.contains("below_landmark").sum())},
                 {"panel":"flow","hospital":hospital,"category":"left truncated","value":int(block.exclusion_reason.fillna("").str.contains("left_truncated").sum())},
                 {"panel":"flow","hospital":hospital,"category":"onset unreconstructable/outside","value":int(block.exclusion_reason.fillna("").str.contains("onset_unreconstructable|onset_outside_record").sum())}]
    for (part,hospital,event),n in split.groupby(["split","hospital_set","any_sepsis_label"]).size().items(): rows.append({"panel":"split","hospital":hospital,"category":f"{part} {'event' if event else 'non-event'}","value":n})
    bins=np.asarray([0,6,12,24,48,72,96,120,168,np.inf]); labels=["<6","6-12","13-24","25-48","49-72","73-96","97-120","121-168",">168"]
    for hospital in ("A","B"):
        counts=pd.cut(manifest.loc[manifest.hospital_set==hospital,"last_ICULOS"],bins=bins,labels=labels,right=True,include_lowest=True).value_counts(sort=False)
        rows += [{"panel":"length","hospital":hospital,"category":str(k),"value":int(v)} for k,v in counts.items()]
    data=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.1))
    flow=data[(data.panel=="flow")&data.category.isin(["raw","primary non-event","primary event","below landmark","left truncated","onset unreconstructable/outside"])]
    pivot=flow.pivot(index="category",columns="hospital",values="value").loc[["raw","primary non-event","primary event","below landmark","left truncated","onset unreconstructable/outside"]]
    pivot.plot.bar(ax=axes[0],color=["#4C78A8","#F58518"]); axes[0].set_title("A. Cohort construction"); axes[0].set_ylabel("Patients"); axes[0].tick_params(axis="x",rotation=55)
    split_data=data[data.panel=="split"]; split_data.assign(split=split_data.category.str.split().str[0]).groupby(["split","hospital"]).value.sum().unstack().loc[["train","validation","calibration","test"]].plot.bar(ax=axes[1],color=["#4C78A8","#F58518"])
    axes[1].set_title("B. Patient-level partition"); axes[1].set_ylabel("Patients"); axes[1].tick_params(axis="x",rotation=25)
    length=data[data.panel=="length"].pivot(index="category",columns="hospital",values="value").loc[labels]; length.plot.bar(ax=axes[2],color=["#4C78A8","#F58518"])
    axes[2].set_title("C. Observed monitoring length"); axes[2].set_ylabel("Patients"); axes[2].tick_params(axis="x",rotation=55)
    for ax in axes: ax.grid(axis="y",alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); _save(fig,config.root/"literature"/"figures"/"figure7_cohort_flow",data,{"figure":7,"description":"Cohort flow, split counts, and observed monitoring length"})


def figure8(config: RealDataConfig) -> None:
    data=pd.read_csv(config.outputs/"internal_primary"/"cumulative_false_alert.csv")
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),sharey=True)
    for ax,panel in zip(axes,("overall","A","B")):
        block=data[data.panel==panel]
        for method in ("Pointwise-alpha","Bonferroni","PaFAR-F","PaFAR-T","PaFAR-HC"):
            m=block[block.method==method]; ax.plot(m.hour,m.estimate,label=method,color=COLORS[method],lw=1.7); ax.fill_between(m.hour,m.lower_95,m.upper_95,color=COLORS[method],alpha=.10)
        ax.set_title(panel if panel=="overall" else f"Hospital {panel}"); ax.set_xlabel("ICU hour"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Cumulative first-false-alert probability"); axes[-1].legend(frameon=False,fontsize=7,loc="upper left")
    fig.tight_layout(); _save(fig,config.root/"literature"/"figures"/"figure8_realdata_false_alert",data,{"figure":8,"alpha":.10,"bootstrap":"hospital-stratified whole-patient, B=1000"})


def figure9(config: RealDataConfig) -> None:
    data=pd.read_csv(config.outputs/"internal_primary"/"internal_results_summary.csv")
    keep=data[data.method.isin(("Pointwise-alpha","Bonferroni","PaFAR-F","PaFAR-T","PaFAR-HC","Fixed 0.5","Youden"))]
    fig,ax=plt.subplots(figsize=(7.4,5.1))
    markers={.05:"o",.10:"s"}
    for row in keep.itertuples():
        ax.scatter(row.pfa,row.sens3,color=COLORS[row.method],marker=markers[float(row.alpha)],s=55)
        ax.annotate(f"{row.method} ({row.alpha:.2f})",(row.pfa,row.sens3),xytext=(4,3),textcoords="offset points",fontsize=7)
    ax.axvline(.05,color="#999999",ls=":",lw=1); ax.axvline(.10,color="#666666",ls="--",lw=1)
    ax.set_xlabel("Patient-level PFA"); ax.set_ylabel("Sens3"); ax.grid(alpha=.25); fig.tight_layout()
    _save(fig,config.root/"literature"/"figures"/"figure9_realdata_tradeoff",keep,{"figure":9,"alphas":[.05,.10]})


def figure10(config: RealDataConfig) -> None:
    data=pd.read_csv(config.outputs/"transfer_primary"/"transfer_results_summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.2),sharex="col")
    for col,direction in enumerate(("A→B","B→A")):
        block=data[data.direction==direction]
        for strategy in ("Direct source transfer","Local PaFAR-F","Local PaFAR-T","Local PaFAR-HC"):
            m=block[block.strategy==strategy].sort_values("target_m0")
            axes[0,col].plot(m.target_m0,m.pfa,marker="o",label=strategy,color=COLORS[strategy]); axes[1,col].plot(m.target_m0,m.sens3,marker="o",label=strategy,color=COLORS[strategy])
        axes[0,col].axhline(.10,color="black",ls="--",lw=1); axes[0,col].set_title(direction); axes[1,col].set_xlabel("Target non-event calibration size")
        for ax in axes[:,col]: ax.grid(alpha=.25)
    axes[0,0].set_ylabel("Target PFA"); axes[1,0].set_ylabel("Sens3"); axes[0,1].legend(frameon=False,fontsize=8)
    fig.tight_layout(); _save(fig,config.root/"literature"/"figures"/"figure10_cross_hospital",data,{"figure":10,"alpha":.10,"direct_transfer_m0":0})


def make_all_figures(config: RealDataConfig) -> None:
    figure7(config); figure8(config); figure9(config); figure10(config)

