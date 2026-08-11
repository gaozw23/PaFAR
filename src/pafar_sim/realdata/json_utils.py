"""Strict-JSON codec for numeric analysis artifacts.

JSON has no standard representation for infinities.  PaFAR thresholds may
legitimately contain them, so encode them explicitly instead of relying on
the non-standard ``Infinity``/``-Infinity`` tokens accepted by CPython.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


_FLOAT_TAG = "__float__"
_POS_INF = "pos_inf"
_NEG_INF = "neg_inf"


def encode_json_numeric(obj: Any) -> Any:
    """Return a standard-JSON-safe copy of a nested numeric object.

    Infinities use an explicit tagged representation.  NaN is rejected: it
    denotes an unexpected undefined value unless a caller first maps a
    specifically documented structural-NA field to its own representation.
    """
    if isinstance(obj, np.ndarray):
        return encode_json_numeric(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if np.isnan(value):
            raise ValueError("NaN is not permitted by the strict JSON numeric codec")
        if np.isposinf(value):
            return {_FLOAT_TAG: _POS_INF}
        if np.isneginf(value):
            return {_FLOAT_TAG: _NEG_INF}
        return value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {key: encode_json_numeric(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_json_numeric(value) for value in obj]
    return obj


def decode_json_numeric(obj: Any) -> Any:
    """Losslessly restore tagged infinities in a decoded JSON object."""
    if isinstance(obj, dict):
        if set(obj) == {_FLOAT_TAG}:
            tag = obj[_FLOAT_TAG]
            if tag == _POS_INF:
                return np.inf
            if tag == _NEG_INF:
                return -np.inf
            raise ValueError(f"unsupported tagged float value: {tag!r}")
        return {key: decode_json_numeric(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode_json_numeric(value) for value in obj]
    return obj


def dumps_json_numeric(obj: Any, **kwargs: Any) -> str:
    """Encode numeric sentinels and serialize as strict standard JSON."""
    if "allow_nan" in kwargs and kwargs["allow_nan"] is not False:
        raise ValueError("dumps_json_numeric always requires allow_nan=False")
    kwargs["allow_nan"] = False
    return json.dumps(encode_json_numeric(obj), **kwargs)


def loads_json_numeric(text: str, **kwargs: Any) -> Any:
    """Parse JSON and restore values written by :func:`dumps_json_numeric`."""
    return decode_json_numeric(json.loads(text, **kwargs))


def numeric_state(obj: Any) -> dict[str, int | bool | str]:
    """Summarize finite/infinite state for CSV threshold audit columns."""
    values = np.asarray(obj, dtype=float)
    nan_count = int(np.isnan(values).sum())
    if nan_count:
        raise ValueError("NaN threshold encountered")
    neg_count = int(np.isneginf(values).sum())
    pos_count = int(np.isposinf(values).sum())
    nonfinite_count = neg_count + pos_count
    state = "finite"
    if neg_count and not pos_count:
        state = "neg_inf"
    elif pos_count and not neg_count:
        state = "pos_inf"
    elif neg_count and pos_count:
        state = "mixed_inf"
    return {
        "threshold_state": state,
        "threshold_nonfinite_count": nonfinite_count,
        "threshold_has_neg_inf": bool(neg_count),
        "threshold_has_pos_inf": bool(pos_count),
    }
