"""State of health from cell test data: capacity by coulomb counting, incremental capacity analysis (dQ/dV), a small
regression model from voltage-curve features to capacity, and the fit of the ageing model of
:mod:`bess_degradation.aging`.

Checkups (reference performance tests) charge and discharge every cell between 2.5 V and 3.65 V at 50 A (C/3.6)
several times; the later checkups add the same at 18 A (C/10). The discharge capacity is the integral of the current
over each constant-current discharge, the checkup capacity the mean of its last two discharges.

Usage:
    from bess_degradation.cellfit import checkup_table, fit_calendar, fit_cycle
    caps = checkup_table(Path("data/cells"))
    cal = fit_calendar(caps)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from bess_degradation.aging import R_GAS, T_REF, AgingModel
from bess_degradation.cells import CELLS, read_checkup

log = logging.getLogger(__name__)


def cc_segments(df: pd.DataFrame, min_hours: float = 1.0, min_current: float = 5.0) -> pd.DataFrame:
    """Constant-current charge (+) and discharge (-) segments of a test: start/end time, current, coulomb-counted Ah,
    voltage at start and end."""
    i = df["current_a"].to_numpy(float)
    sign = np.where(i > min_current, 1, np.where(i < -min_current, -1, 0))
    seg = np.r_[0, np.cumsum((sign[1:] != sign[:-1]) | (df["step"].to_numpy()[1:] != df["step"].to_numpy()[:-1]))]
    rows = []
    t = df["time_h"].to_numpy(float)
    v = df["voltage"].to_numpy(float)
    for k in np.unique(seg):
        m = np.flatnonzero(seg == k)
        if sign[m[0]] == 0 or len(m) < 10:
            continue
        dur = t[m[-1]] - t[m[0]]
        cur = float(np.median(i[m]))
        if dur < min_hours or np.std(i[m]) > 0.05 * abs(cur) + 0.5:     # not constant current (CV phase etc.)
            continue
        rows.append({"start_h": t[m[0]], "end_h": t[m[-1]], "current_a": cur,
                     "ah": float(np.trapezoid(i[m], t[m])), "v_start": v[m[0]], "v_end": v[m[-1]], "rows": m})
    return pd.DataFrame(rows)


def checkup_table(data_dir: Path) -> pd.DataFrame:
    """One row per (cell, checkup, current): discharge and charge capacity, time of the test and cell conditions."""
    rows = []
    for f in sorted((data_dir / "checkups").glob("*/*.parquet")):
        cell, checkup = f.parent.name, int(f.name[:4])
        df, meta = read_checkup(f)
        seg = cc_segments(df)
        if seg.empty:
            continue
        start = pd.Timestamp(meta["start"])
        nominal = np.where(seg["current_a"].abs() < 30, 18.0, 50.0)       # C/10 or C/3.6 test
        for cur, grp in seg.groupby(nominal):
            dis = grp[grp["current_a"] < 0]
            chg = grp[grp["current_a"] > 0]
            full_dis = dis[dis["v_end"] < 2.55]
            if full_dis.empty:
                continue
            paper, mode, temp, soc = CELLS[cell]
            rows.append({"cell": cell, "paper_no": paper, "mode": mode, "temp_c": temp, "storage_soc": soc,
                         "checkup": checkup, "current_a": float(cur), "file": f.name,
                         "time": start + pd.Timedelta(hours=float(full_dis["end_h"].iloc[-1])),
                         "q_dis_ah": float(-full_dis["ah"].tail(2).mean()),
                         "q_chg_ah": float(chg["ah"].tail(2).mean()) if not chg.empty else np.nan,
                         "n_dis": len(full_dis), "test_temp_c": float(df["temp_c"].mean())})
    out = pd.DataFrame(rows)
    first = out[out["checkup"] == 0].set_index("cell")["time"]
    out["days"] = (out["time"] - out["cell"].map(first)) / pd.Timedelta(days=1)
    return out.sort_values(["cell", "checkup", "current_a"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# Incremental capacity analysis
# ---------------------------------------------------------------------------------------------------------------------

def ica_curve(df: pd.DataFrame, current_a: float, direction: str = "charge", dv: float = 0.002,
              v_range: tuple[float, float] = (3.2, 3.5), window: int = 11) -> pd.Series:
    """dQ/dV (Ah/V) of the last constant-current charge or discharge at about ``current_a``, on a voltage grid of
    ``dv`` volts: charge accumulated in each voltage bin divided by the bin width, then Savitzky-Golay smoothed."""
    seg = cc_segments(df)
    sign = 1 if direction == "charge" else -1
    seg = seg[(np.sign(seg["current_a"]) == sign) & ((seg["current_a"].abs() - current_a).abs() < 0.2 * current_a)]
    if seg.empty:
        raise ValueError(f"no {direction} at {current_a} A")
    rows = seg["rows"].iloc[-1]
    v = df["voltage"].to_numpy(float)[rows]
    q = np.abs(np.concatenate([[0.0], np.cumsum(np.diff(df["time_h"].to_numpy(float)[rows])
                                               * df["current_a"].to_numpy(float)[rows][1:])]))
    v = np.maximum.accumulate(v) if sign > 0 else np.minimum.accumulate(v)
    edges = np.arange(v_range[0], v_range[1] + dv / 2, dv)
    q_at = np.interp(edges, v if sign > 0 else v[::-1], q if sign > 0 else q[::-1])
    dqdv = np.abs(np.diff(q_at)) / dv
    if window and len(dqdv) > window:
        dqdv = savgol_filter(dqdv, window, 2)
    return pd.Series(dqdv, index=np.round((edges[:-1] + edges[1:]) / 2, 4), name="dqdv")


def ica_features(curve: pd.Series, windows=((3.20, 3.30), (3.30, 3.36), (3.36, 3.42), (3.42, 3.50))) -> dict:
    """Height and position of the largest peak, and the charge (Ah) in fixed voltage windows."""
    dv = float(np.diff(curve.index[:2])[0])
    out = {"peak_v": float(curve.idxmax()), "peak_ah_per_v": float(curve.max())}
    for lo, hi in windows:
        out[f"ah_{lo:.2f}_{hi:.2f}"] = float(curve[(curve.index >= lo) & (curve.index < hi)].sum() * dv)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Fitting the ageing model to the checkups
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class FitResult:
    params: dict
    stderr: dict
    c0: pd.Series            # fitted initial capacity per cell (Ah)
    rmse_pct: float          # residual RMSE, % of initial capacity
    residuals: pd.DataFrame


def _arr(temp_c, ea):
    return np.exp(-ea / R_GAS * (1.0 / (np.asarray(temp_c, float) + 273.15) - 1.0 / T_REF))


def _fit(data: pd.DataFrame, loss_fn, names: list[str], x0: list[float], lower: list[float], upper: list[float],
         scale: list[float]) -> FitResult:
    cells = list(data["cell"].unique())
    ci = data["cell"].map({c: i for i, c in enumerate(cells)}).to_numpy()
    c0_init = data.groupby("cell")["q_dis_ah"].max().reindex(cells).to_numpy()
    q = data["q_dis_ah"].to_numpy(float)

    def resid(z):
        p = dict(zip(names, z[:len(names)] * scale))
        c0 = z[len(names):]
        return (c0[ci] * (1 - loss_fn(p, data)) - q) / c0[ci]

    z0 = np.r_[np.asarray(x0) / scale, c0_init]
    lb = np.r_[np.asarray(lower) / scale, c0_init * 0.9]
    ub = np.r_[np.asarray(upper) / scale, c0_init * 1.1]
    res = least_squares(resid, z0, bounds=(lb, ub))
    r = res.fun
    dof = max(len(r) - len(res.x), 1)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * (r @ r) / dof
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(res.x), np.nan)
    p = dict(zip(names, res.x[:len(names)] * scale))
    pse = dict(zip(names, se[:len(names)] * np.asarray(scale)))
    out = data[["cell", "checkup", "days", "q_dis_ah"]].copy()
    out["resid_pct"] = r * 100
    return FitResult(params=p, stderr=pse, c0=pd.Series(res.x[len(names):], index=cells),
                     rmse_pct=float(np.sqrt(np.mean(r ** 2)) * 100), residuals=out)


def fit_calendar(caps: pd.DataFrame, current_a: float = 50.0, z_fixed: float | None = None) -> FitResult:
    """Fit k_cal, Ea_cal, gamma (and z_cal unless fixed) on the storage cells, with a free initial capacity per cell.

    Checkup 0 (formation, on a different cycler) is excluded: several cells gained 1-3 % capacity between it and the
    first regular checkup, which is a measurement offset rather than ageing.
    """
    d = caps[(caps["mode"] == "calendar") & (caps["current_a"] == current_a) & (caps["checkup"] > 0)].copy()
    d["soc"] = d["storage_soc"] / 100.0
    names = ["k_cal", "ea_cal", "gamma"] + ([] if z_fixed else ["z_cal"])

    def loss(p, x):
        z = z_fixed or p["z_cal"]
        return p["k_cal"] * _arr(x["temp_c"], p["ea_cal"]) * np.exp(p["gamma"] * (x["soc"] - 0.5)) * x["days"] ** z

    x0 = [2e-3, 40_000, 1.0] + ([] if z_fixed else [0.5])
    lo = [1e-5, 0, -10] + ([] if z_fixed else [0.2])
    hi = [1e-1, 150_000, 10] + ([] if z_fixed else [1.0])
    sc = [1e-3, 1e4, 1.0] + ([] if z_fixed else [1.0])
    res = _fit(d, loss, names, x0, lo, hi, sc)
    if z_fixed:
        res.params["z_cal"] = z_fixed
    return res


def fit_cycle(caps: pd.DataFrame, calendar: AgingModel, cycles_per_checkup: dict | None = None,
              cell_temp: dict | None = None, current_a: float = 50.0, z_fixed: float | None = None) -> FitResult:
    """Fit k_cyc, Ea_cyc (and z_cyc) on the cycling cells after subtracting the calendar ageing that the fitted
    calendar model predicts for their test time (full cycles spend equal time at every SoC, so the calendar rate is
    averaged over SoC 0..1). ``cell_temp`` overrides the chamber temperature with the measured cell temperature."""
    d = caps[(caps["mode"] == "cycling") & (caps["current_a"] == current_a) & (caps["checkup"] > 0)].copy()
    if cell_temp:
        d["temp_c"] = d["cell"].map(cell_temp).fillna(d["temp_c"])
    d["n_eq"] = d["checkup"].map(cycles_per_checkup) if cycles_per_checkup else d["checkup"].astype(float)
    socs = np.linspace(0, 1, 21)
    names = ["k_cyc", "ea_cyc"] + ([] if z_fixed else ["z_cyc"])

    def loss(p, x):
        k = np.array([np.mean(calendar.k_cal_at(socs, t) ** (1 / calendar.z_cal)) ** calendar.z_cal
                      for t in x["temp_c"]])
        cal = k * x["days"].to_numpy(float) ** calendar.z_cal
        z = z_fixed or p["z_cyc"]
        return cal + p["k_cyc"] * _arr(x["temp_c"], p["ea_cyc"]) * x["n_eq"].to_numpy(float) ** z

    x0 = [3e-5, 30_000] + ([] if z_fixed else [1.0])
    lo = [1e-8, 0] + ([] if z_fixed else [0.3])
    hi = [1e-2, 150_000] + ([] if z_fixed else [1.5])
    sc = [1e-5, 1e4] + ([] if z_fixed else [1.0])
    res = _fit(d, loss, names, x0, lo, hi, sc)
    if z_fixed:
        res.params["z_cyc"] = z_fixed
    return res


def model_from_fits(cal: FitResult, cyc: FitResult, beta: float = 1.0) -> AgingModel:
    return AgingModel(k_cal=cal.params["k_cal"], ea_cal=cal.params["ea_cal"], gamma=cal.params["gamma"],
                      z_cal=cal.params["z_cal"], k_cyc=cyc.params["k_cyc"], ea_cyc=cyc.params["ea_cyc"],
                      z_cyc=cyc.params["z_cyc"], beta=beta)
