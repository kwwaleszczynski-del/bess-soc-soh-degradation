"""Rainflow cycle counting (ASTM E1049-85, three-point method; residual counted as half cycles).

A state-of-charge profile is reduced to its turning points (reversals); every closed hysteresis loop in it is a full
cycle, and the unclosed remainder is counted as half cycles. Each cycle has a depth (range of SoC), a mean SoC and
the positions where it starts and ends.

Usage:
    from bess_degradation.rainflow import count_cycles
    cycles = count_cycles(soc_series)               # DataFrame: depth, mean, count, start, end
    equivalent_full_cycles = (cycles["depth"] * cycles["count"]).sum()
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reversals(x, min_range: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the turning points of ``x`` (first and last point included).

    Plateaus are collapsed, NaNs dropped. ``min_range`` > 0 applies a hysteresis filter: a reversal only counts once
    the signal has moved back by at least that much (removes measurement noise).
    """
    x = np.asarray(x, float)
    idx = np.flatnonzero(~np.isnan(x))
    if len(idx) == 0:
        return idx, x[idx]
    v = x[idx]
    keep = np.r_[True, np.diff(v) != 0]              # collapse plateaus
    idx, v = idx[keep], v[keep]
    if len(v) < 3:
        return idx, v
    d = np.sign(np.diff(v))
    turn = np.r_[True, d[1:] != d[:-1], True]
    idx, v = idx[turn], v[turn]
    if min_range <= 0 or len(v) < 3:
        return idx, v
    out_i, out_v = [idx[0]], [v[0]]
    direction = 0
    for i, val in zip(idx[1:], v[1:]):
        if direction == 0:                           # wait until the signal has left the start by min_range
            if abs(val - out_v[0]) >= min_range:
                direction = 1 if val > out_v[0] else -1
                out_i.append(i)
                out_v.append(val)
        elif (val - out_v[-1]) * direction > 0:      # extends the current excursion
            out_i[-1], out_v[-1] = i, val
        elif abs(val - out_v[-1]) >= min_range:      # a real reversal
            direction = -direction
            out_i.append(i)
            out_v.append(val)
    if out_i[-1] != idx[-1] and v[-1] != out_v[-1]:  # the end of the series closes the last half cycle
        if (v[-1] - out_v[-1]) * direction > 0:
            out_i[-1], out_v[-1] = idx[-1], v[-1]
        else:
            out_i.append(idx[-1])
            out_v.append(v[-1])
    return np.asarray(out_i), np.asarray(out_v)


def count_cycles(x, min_range: float = 0.0) -> pd.DataFrame:
    """Rainflow cycles of ``x``: columns depth, mean, count (1 or 0.5), start, end (positions in ``x``)."""
    idx, v = reversals(x, min_range)
    stack_i: list[int] = []
    stack_v: list[float] = []
    rows: list[tuple[float, float, float, int, int]] = []
    for i, val in zip(idx, v):
        stack_i.append(int(i))
        stack_v.append(float(val))
        while len(stack_v) >= 3:
            x_rng = abs(stack_v[-1] - stack_v[-2])
            y_rng = abs(stack_v[-2] - stack_v[-3])
            if x_rng < y_rng:
                break
            if len(stack_v) == 3:                    # Y contains the starting point: half cycle
                rows.append((y_rng, (stack_v[0] + stack_v[1]) / 2, 0.5, stack_i[0], stack_i[1]))
                del stack_v[0], stack_i[0]
            else:                                    # full cycle: remove the two points of Y
                rows.append((y_rng, (stack_v[-2] + stack_v[-3]) / 2, 1.0, stack_i[-3], stack_i[-2]))
                del stack_v[-3:-1], stack_i[-3:-1]
    for k in range(len(stack_v) - 1):                # residual: half cycles
        rows.append((abs(stack_v[k + 1] - stack_v[k]), (stack_v[k + 1] + stack_v[k]) / 2, 0.5,
                     stack_i[k], stack_i[k + 1]))
    df = pd.DataFrame(rows, columns=["depth", "mean", "count", "start", "end"])
    return df[df["depth"] > 0].reset_index(drop=True)


def depth_histogram(cycles: pd.DataFrame, bins=None) -> pd.DataFrame:
    """Number of cycles and equivalent full cycles (depth x count) per depth bin."""
    bins = np.linspace(0, 1, 11) if bins is None else np.asarray(bins)
    cut = pd.cut(cycles["depth"], bins, include_lowest=True)
    g = cycles.assign(efc=cycles["depth"] * cycles["count"]).groupby(cut, observed=False)
    return pd.DataFrame({"cycles": g["count"].sum(), "efc": g["efc"].sum()})
