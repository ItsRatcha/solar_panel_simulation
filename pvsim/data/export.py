"""Resampling and CSV export of recorded runs.

Resampling groups rows by panel and a floored time bucket.  Numeric columns
are averaged (power is the mean of the per-sample power, never mean V times
mean I), categoricals take the most frequent value, and ratio columns are
recomputed from the bucket sums of their numerator and denominator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import columns

RESAMPLE_CHOICES = ("raw", "1s", "1min")
_FREQ = {"1s": "1s", "1min": "1min"}


def _cols_in(df: pd.DataFrame):
    extended = "p_mpp_true" in df.columns
    return [c for c in columns(extended) if c.key in df.columns]


def resample(df: pd.DataFrame, rule: str = "raw") -> pd.DataFrame:
    if rule == "raw" or df.empty:
        return df
    if rule not in _FREQ:
        raise ValueError(f"resample must be one of {RESAMPLE_CHOICES}, got {rule!r}")
    cols = _cols_in(df)
    work = df.copy()
    work["timestamp"] = work["timestamp"].dt.floor(_FREQ[rule])
    g = work.groupby(["panel_id", "timestamp"], sort=False)

    out: dict[str, pd.Series] = {}
    means = [c.key for c in cols if c.agg == "mean"]
    if means:
        m = g[means].mean()
        for k in means:
            out[k] = m[k]
    for c in cols:
        if c.agg == "mode":
            out[c.key] = g[c.key].agg(lambda s: s.value_counts(sort=True).index[0])
        elif isinstance(c.agg, tuple) and c.agg[0] == "ratio":
            _, num, den = c.agg
            sums = g[[num, den]].sum()
            r = sums[num] / sums[den]
            out[c.key] = r.where(sums[den] > 1e-9, 0.0)
    res = pd.DataFrame(out).reset_index()
    res = res[[c.key for c in cols]]
    for c in cols:
        if c.dtype == "int":
            res[c.key] = res[c.key].astype(int)
        elif c.dtype == "bool":
            res[c.key] = res[c.key].astype(bool)
    return res.sort_values(["timestamp", "panel_id"], kind="stable", ignore_index=True)


def to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal keys to CSV names and format the timestamp as ISO 8601."""
    cols = _cols_in(df)
    out = df[[c.key for c in cols]].copy()
    ts = out["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + ts.dt.strftime("%z").str.replace(
            r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    return out.rename(columns={c.key: c.name for c in cols})


def write_csv(df: pd.DataFrame, path: str | Path, rule: str = "raw", float_format: str = "%.6g") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_csv_frame(resample(df, rule)).to_csv(path, index=False, float_format=float_format)
    return path
