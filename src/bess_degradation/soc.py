"""State of charge from metered power: reconstruction, calibration against a reported SoC, and validation metrics.

A battery's state of charge (SoC, fraction of usable energy) follows from the energy it exchanges with the grid:

    SoC_{k+1} = SoC_k + (eta_c * C_k - D_k / eta_d - P_sb * dt) / E

with C_k and D_k the MWh charged and discharged at the connection point in interval k, eta_c and eta_d the one-way
efficiencies, P_sb a constant standby drain (auxiliaries, self-discharge; MW of stored energy) and E the usable
energy (MWh). Over any window w the change of SoC is linear in the window's charged energy, discharged energy and
duration:

    dSoC_w = a * C_w - b * D_w - c * H_w,     a = eta_c / E,   b = 1 / (eta_d * E),   c = P_sb / E

so a, b and c can be estimated by least squares from a reported SoC. Two quantities follow without further
assumptions: the round-trip efficiency ``a / b = eta_c * eta_d`` and the standby drain. Splitting the efficiency
symmetrically (eta_c = eta_d) also identifies the usable energy, ``E = 1 / sqrt(a * b)``, which is what makes an
operational estimate of capacity, and hence state of health, possible from metered data.

Usage:
    from bess_degradation.soc import SocModel, calibrate, reconstruct, error_metrics
    cal = calibrate(soc_obs, charge_mwh, discharge_mwh, window="1h")
    soc = reconstruct(charge_mwh, discharge_mwh, cal.model, soc0=soc_obs.iloc[0])
    error_metrics(soc_obs, soc)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DT_5MIN = 5 / 60


@dataclass(frozen=True)
class SocModel:
    energy_mwh: float            # usable energy between SoC 0 and 1, stored-energy basis
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    standby_mw: float = 0.0      # constant drain of stored energy

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge

    def delta(self, charge_mwh, discharge_mwh, dt_hours=DT_5MIN) -> np.ndarray:
        """Change of SoC (fraction) in each interval."""
        c, d = np.asarray(charge_mwh, float), np.asarray(discharge_mwh, float)
        return (self.eta_charge * c - d / self.eta_discharge - self.standby_mw * dt_hours) / self.energy_mwh


def reconstruct(charge_mwh, discharge_mwh, model: SocModel, soc0: float, dt_hours: float = DT_5MIN,
                clip: bool = False) -> pd.Series | np.ndarray:
    """SoC at the start of every interval, starting from ``soc0`` (open-loop integration).

    With ``clip`` the SoC is held within [0, 1], as the battery's own limits would do; without it the raw integral is
    returned, which is what the validation uses (clipping would hide drift).
    """
    delta = model.delta(charge_mwh, discharge_mwh, dt_hours)
    if not clip:
        soc = soc0 + np.concatenate([[0.0], np.cumsum(delta[:-1])])
    else:
        soc = np.empty(len(delta))
        s = soc0
        for i, dlt in enumerate(delta):
            soc[i] = s
            s = min(max(s + dlt, 0.0), 1.0)
    if isinstance(charge_mwh, pd.Series):
        return pd.Series(soc, index=charge_mwh.index, name="soc")
    return soc


@dataclass
class Calibration:
    a: float                  # SoC per MWh charged
    b: float                  # SoC per MWh discharged
    c: float                  # SoC per hour of standby drain
    n_windows: int
    n_outliers: int
    r2: float
    rmse_window: float        # residual RMSE of dSoC per window (fraction)

    @property
    def round_trip_efficiency(self) -> float:
        return self.a / self.b

    @property
    def energy_mwh(self) -> float:
        """Usable energy assuming eta_c = eta_d."""
        return 1.0 / np.sqrt(self.a * self.b)

    @property
    def model(self) -> SocModel:
        eta = float(np.sqrt(self.a / self.b))
        e = self.energy_mwh
        return SocModel(energy_mwh=e, eta_charge=eta, eta_discharge=eta, standby_mw=self.c * e)


def window_sums(soc: pd.Series, charge_mwh: pd.Series, discharge_mwh: pd.Series, window: str = "1h",
                dt_hours: float = DT_5MIN) -> pd.DataFrame:
    """Per window: SoC change between the first interval of the window and the first of the next, and the energy
    charged / discharged and hours in between. Windows without a SoC reading at both ends, or with missing power
    data inside, are dropped.
    """
    frame = pd.DataFrame({"soc": soc, "c": charge_mwh, "d": discharge_mwh})
    key = frame.index.floor(window)
    g = frame.groupby(key)
    out = pd.DataFrame({"C": g["c"].sum(min_count=1), "D": g["d"].sum(min_count=1), "n": g["c"].count()})
    at_start = frame.loc[frame.index == key, "soc"]           # reading exactly at the window start (may be NaN)
    out["soc_start"] = at_start.reindex(out.index)
    out["soc_end"] = out["soc_start"].shift(-1)
    step = out.index.to_series().diff().shift(-1)
    full = pd.Timedelta(window)
    expected = int(round(full / pd.Timedelta(hours=dt_hours)))
    keep = (step == full) & (out["n"] == expected)
    out = out[keep].dropna(subset=["soc_start", "soc_end", "C", "D"])
    out["dsoc"] = out["soc_end"] - out["soc_start"]
    out["H"] = out["n"] * dt_hours
    return out


def calibrate(soc: pd.Series, charge_mwh: pd.Series, discharge_mwh: pd.Series, window: str = "1h",
              dt_hours: float = DT_5MIN, outlier_k: float = 6.0, max_iter: int = 5) -> Calibration:
    """Least-squares fit of dSoC = a*C - b*D - c*H over windows, iteratively dropping outliers (BMS recalibration
    jumps, data glitches) beyond ``outlier_k`` robust standard deviations."""
    w = window_sums(soc, charge_mwh, discharge_mwh, window, dt_hours)
    if len(w) < 10:
        raise ValueError(f"only {len(w)} usable windows")
    x = np.column_stack([w["C"], -w["D"], -w["H"]])
    y = w["dsoc"].to_numpy()
    keep = np.ones(len(y), bool)
    for _ in range(max_iter):
        coef, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
        res = y - x @ coef
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        new = np.abs(res) <= outlier_k * max(mad, 1e-6)
        if (new == keep).all():
            break
        keep = new
    res_k = res[keep]
    ss_tot = np.sum((y[keep] - y[keep].mean()) ** 2)
    return Calibration(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), n_windows=int(keep.sum()),
                       n_outliers=int((~keep).sum()), r2=float(1 - np.sum(res_k ** 2) / ss_tot) if ss_tot else np.nan,
                       rmse_window=float(np.sqrt(np.mean(res_k ** 2))))


def error_metrics(observed: pd.Series, reconstructed: pd.Series) -> dict:
    """RMSE, MAE, bias, largest error (all in SoC percentage points) and the linear drift of the error (pp/day)."""
    err = (reconstructed - observed).dropna() * 100
    if err.empty:
        return {"n": 0, "rmse_pp": np.nan, "mae_pp": np.nan, "bias_pp": np.nan, "max_abs_pp": np.nan,
                "drift_pp_per_day": np.nan}
    days = (err.index - err.index[0]) / pd.Timedelta(days=1)
    drift = float(np.polyfit(np.asarray(days, float), err.to_numpy(), 1)[0]) if len(err) > 2 else np.nan
    return {"n": int(len(err)), "rmse_pp": float(np.sqrt(np.mean(err ** 2))), "mae_pp": float(err.abs().mean()),
            "bias_pp": float(err.mean()), "max_abs_pp": float(err.abs().max()), "drift_pp_per_day": drift}


def reanchored_errors(observed: pd.Series, charge_mwh: pd.Series, discharge_mwh: pd.Series, model: SocModel,
                      anchor: str = "1D", dt_hours: float = DT_5MIN) -> pd.DataFrame:
    """Integrate from the observed SoC at the start of every ``anchor`` period (e.g. daily at midnight, as a market
    model fed with the operator's daily SoC would) and return the error (pp) with the hours since the anchor."""
    rows = []
    key = observed.index.floor(anchor)
    for start, obs in observed.groupby(key):
        if obs.isna().iloc[0]:
            continue
        sl = slice(obs.index[0], obs.index[-1])
        rec = reconstruct(charge_mwh.loc[sl], discharge_mwh.loc[sl], model, float(obs.iloc[0]), dt_hours)
        err = (rec - obs) * 100
        hours = (obs.index - start) / pd.Timedelta(hours=1)
        rows.append(pd.DataFrame({"anchor": start, "hours": np.asarray(hours, float), "error_pp": err.to_numpy()},
                                 index=obs.index))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["anchor", "hours", "error_pp"])
