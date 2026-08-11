"""Prespecified descriptive subgroup summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def age_group(age: float) -> str:
    if not np.isfinite(age): return "missing"
    if age < 50: return "<50"
    if age < 65: return "50-64"
    if age < 80: return "65-79"
    return ">=80"


def icu_type(unit1: float, unit2: float) -> str:
    one, two = unit1 == 1, unit2 == 1
    if one and two: return "both"
    if one: return "Unit1 only"
    if two: return "Unit2 only"
    return "neither/missing"


def summarize_subgroups(patient_results: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    frame=patient_results.merge(manifest[["patient_id","age","gender","unit1","unit2"]],on="patient_id",how="left",validate="one_to_one")
    frame["age_group"]=[age_group(x) for x in frame.age]
    frame["icu_type"]=[icu_type(a,b) for a,b in zip(frame.unit1,frame.unit2)]
    non=frame.loc[~frame.event.astype(bool)].copy()
    non["exposure_quartile"]=pd.qcut(non.exposure_days,4,labels=["Q1 shortest","Q2","Q3","Q4 longest"],duplicates="drop")
    frame=frame.merge(non[["patient_id","exposure_quartile"]],on="patient_id",how="left")
    rows=[]
    for variable in ("hospital","gender","icu_type","age_group","exposure_quartile"):
        for value,block in frame.groupby(variable,dropna=False,observed=False):
            nonblock=block.loc[~block.event.astype(bool)]
            rows.append({"subgroup_variable":variable,"subgroup":str(value),"n_patients":len(block),"n_non_events":len(nonblock),"n_events":int(block.event.sum()),
                         "pfa":float(nonblock.alerted.mean()) if len(nonblock) else np.nan,
                         "sens3":float(block.loc[block.eval3,"valid3"].mean()) if block.eval3.any() else np.nan,
                         "ppv3":float(block.valid3.sum()/block.alerted.sum()) if block.alerted.sum() else np.nan})
    return pd.DataFrame(rows)

