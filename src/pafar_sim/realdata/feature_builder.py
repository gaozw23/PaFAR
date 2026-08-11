"""Causal rolling features on the actual ICU-hour coordinate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import BASELINE_COLUMNS, LONGITUDINAL_COLUMNS, WINDOWS, WINDOW_STATS, raw_feature_names


@dataclass(frozen=True)
class PatientFeatures:
    values: np.ndarray
    labels: np.ndarray
    hours: np.ndarray
    names: tuple[str, ...]


def _summaries(values: np.ndarray, times: np.ndarray) -> tuple[float, ...]:
    keep = np.isfinite(values)
    x, t = values[keep], times[keep]
    count = x.size
    if count == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0)
    mean, minimum, maximum = float(x.mean()), float(x.min()), float(x.max())
    sd = float(x.std(ddof=1)) if count >= 2 else np.nan
    change = float(x[-1] - x[0]) if count >= 2 and np.unique(t).size >= 2 else np.nan
    if count >= 2 and np.unique(t).size >= 2:
        centered = t - t.mean()
        slope = float(np.dot(centered, x - x.mean()) / np.dot(centered, centered))
    else:
        slope = np.nan
    return mean, minimum, maximum, sd, change, slope, float(count)


def _baseline_value(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(float)
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else np.nan


def build_patient_features(
    frame: pd.DataFrame, *, hospital: str, onset: float, tmin: int = 6,
    hmax: int = 168, smooth_length: int = 3, include_hospital: bool = True,
) -> PatientFeatures:
    """Build features from history only; score-history rows begin at tmin-L+1."""
    first_score_hour = max(1, int(tmin) - int(smooth_length) + 1)
    last_hour = int(min(float(frame.ICULOS.iloc[-1]), hmax))
    output_hours = np.arange(first_score_hour, last_hour + 1, dtype=np.int16)
    names = raw_feature_names(include_hospital)
    out = np.empty((len(output_hours), len(names)), dtype=np.float64)
    actual_hours = frame.ICULOS.to_numpy(int)
    observed_values = frame.loc[:, LONGITUDINAL_COLUMNS].to_numpy(float)
    p = len(LONGITUDINAL_COLUMNS)
    grid = np.full((last_hour, p), np.nan, dtype=np.float64)
    within = (actual_hours >= 1) & (actual_hours <= last_hour)
    grid[actual_hours[within]-1] = observed_values[within]
    valid = np.isfinite(grid); times = np.arange(1,last_hour+1,dtype=float)[:,None]
    last = np.full_like(grid,np.nan); last_time = np.full_like(grid,np.nan)
    current_last=np.full(p,np.nan); current_time=np.full(p,np.nan)
    for t in range(last_hour):
        obs=valid[t]; current_last[obs]=grid[t,obs]; current_time[obs]=t+1
        last[t]=current_last; last_time[t]=current_time
    never=~np.isfinite(last_time); since=times-last_time; current=valid.astype(float)
    window_arrays={}
    for window in WINDOWS:
        arrays={name:np.full((last_hour,p),np.nan) for name in WINDOW_STATS}
        for t in range(last_hour):
            lo=max(0,t-window+1); hi=t+1
            block=grid[lo:hi]; block_valid=valid[lo:hi]
            n=block_valid.sum(axis=0).astype(float); has=n>0; sx=np.sum(np.where(block_valid,block,0.0),axis=0)
            mean=np.divide(sx,n,out=np.full(p,np.nan),where=has); arrays["count"][t]=n; arrays["mean"][t]=mean
            centered=np.where(block_valid,block-mean,0.0); arrays["sd"][t]=np.sqrt(np.divide(np.sum(centered*centered,axis=0),n-1,out=np.full(p,np.nan),where=n>1))
            block_times=np.broadcast_to(np.arange(lo+1,hi+1,dtype=float)[:,None],block.shape)
            time_mean=np.divide(np.sum(np.where(block_valid,block_times,0.0),axis=0),n,out=np.zeros(p),where=has)
            time_centered=np.where(block_valid,block_times-time_mean,0.0); denominator=np.sum(time_centered*time_centered,axis=0)
            arrays["slope"][t]=np.divide(np.sum(time_centered*centered,axis=0),denominator,out=np.full(p,np.nan),where=(n>1)&(denominator>0))
            arrays["min"][t]=np.min(np.where(block_valid,block,np.inf),axis=0); arrays["min"][t,~has]=np.nan
            arrays["max"][t]=np.max(np.where(block_valid,block,-np.inf),axis=0); arrays["max"][t,~has]=np.nan
            first=np.argmax(block_valid,axis=0); last_idx=len(block)-1-np.argmax(block_valid[::-1],axis=0); cols=np.arange(p)
            change=block[last_idx,cols]-block[first,cols]; change[n<2]=np.nan; arrays["change"][t]=change
        window_arrays[window]=arrays
    selected=np.arange(first_score_hour-1,last_hour)
    baseline = [_baseline_value(frame, column) for column in BASELINE_COLUMNS]
    column=0; out[:,column:column+len(baseline)]=baseline; column+=len(baseline); out[:,column]=output_hours; column+=1
    if include_hospital: out[:,column]=float(hospital=="B"); column+=1
    for j in range(p):
        out[:,column:column+4]=np.column_stack((last[selected,j],never[selected,j].astype(float),since[selected,j],current[selected,j])); column+=4
        for window in WINDOWS:
            out[:,column:column+len(WINDOW_STATS)]=np.column_stack([window_arrays[window][stat][selected,j] for stat in WINDOW_STATS]); column+=len(WINDOW_STATS)
    if column != len(names): raise AssertionError("feature schema width mismatch")
    labels = ((np.isfinite(onset)) & ((onset - output_hours) > 0) & ((onset - output_hours) <= 6)).astype(np.uint8)
    return PatientFeatures(out, labels, output_hours, names)


def reference_feature_row(
    frame: pd.DataFrame, hour: int, *, hospital: str, onset: float,
    include_hospital: bool = True,
) -> np.ndarray:
    """Slow independent single-row reference used only for smoke auditing."""
    history = frame.loc[frame.ICULOS <= hour].copy()
    result: list[float] = []
    for column in BASELINE_COLUMNS:
        finite = history[column][np.isfinite(history[column])].to_numpy(float)
        result.append(float(finite[0]) if finite.size else np.nan)
    result.append(float(hour))
    if include_hospital:
        result.append(float(hospital == "B"))
    for variable in LONGITUDINAL_COLUMNS:
        series = history[["ICULOS", variable]].dropna()
        if series.empty:
            result.extend((np.nan, 1.0, np.nan, 0.0))
        else:
            result.extend((float(series[variable].iloc[-1]), 0.0, float(hour - series.ICULOS.iloc[-1]), float(series.ICULOS.iloc[-1] == hour)))
        for window in WINDOWS:
            block = history.loc[(history.ICULOS > hour - window) & (history.ICULOS <= hour), ["ICULOS", variable]].dropna()
            if block.empty:
                result.extend((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0))
            else:
                x, t = block[variable].to_numpy(float), block.ICULOS.to_numpy(float)
                mean = float(np.mean(x))
                sd = float(np.std(x,ddof=1)) if len(x) >= 2 else np.nan
                change = float(x[-1] - x[0]) if len(np.unique(t)) >= 2 else np.nan
                if len(np.unique(t)) >= 2:
                    tm = float(np.mean(t))
                    slope = float(np.dot(t-tm,x-mean)/np.dot(t-tm,t-tm))
                else:
                    slope = np.nan
                result.extend((mean, float(min(x)), float(max(x)), sd, change, slope, float(len(x))))
    return np.asarray(result, dtype=np.float64)


def fitting_mask(hours: np.ndarray, onset: np.ndarray, tmin: int, hmax: int) -> np.ndarray:
    return (hours >= tmin) & (hours <= hmax) & (hours < onset)


def utility_mask(hours: np.ndarray, horizon: np.ndarray, tmin: int, hmax: int) -> np.ndarray:
    return (hours >= tmin) & (hours <= np.minimum(horizon, hmax))
