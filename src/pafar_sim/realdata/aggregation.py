"""Descriptive aggregation for overlapping robustness partitions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def descriptive_summary(frame: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    rows=[]
    for keys,block in frame.groupby(group,dropna=False):
        keys=(keys,) if not isinstance(keys,tuple) else keys
        base=dict(zip(group,keys))
        for metric in metrics:
            x=block[metric].to_numpy(float); x=x[np.isfinite(x)]
            rows.append({**base,"metric":metric,"n_defined":len(x),"median":np.median(x) if len(x) else np.nan,
                         "q1":np.quantile(x,.25) if len(x) else np.nan,"q3":np.quantile(x,.75) if len(x) else np.nan,
                         "minimum":x.min() if len(x) else np.nan,"maximum":x.max() if len(x) else np.nan})
    return pd.DataFrame(rows)

