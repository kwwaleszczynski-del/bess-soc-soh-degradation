import numpy as np
import pandas as pd
import pytest

from bess_degradation import soc


def _profile(days=20, seed=0):
    """Random daily charge/discharge blocks on a 5-minute grid (MWh per interval)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01", periods=288 * days, freq="5min")
    hour = idx.hour + idx.minute / 60
    charge = np.where((hour >= 10) & (hour < 14), rng.uniform(4, 8, len(idx)), 0.0)
    discharge = np.where((hour >= 17) & (hour < 20), rng.uniform(5, 9, len(idx)), 0.0)
    return pd.Series(charge, idx), pd.Series(discharge, idx)


def test_reconstruct_matches_energy_balance():
    c, d = _profile(days=1)
    m = soc.SocModel(energy_mwh=250, eta_charge=0.95, eta_discharge=0.94, standby_mw=0.5)
    s = soc.reconstruct(c, d, m, soc0=0.2)
    assert s.iloc[0] == 0.2
    expected_end = 0.2 + (0.95 * c.iloc[:-1].sum() - d.iloc[:-1].sum() / 0.94 - 0.5 * (len(c) - 1) / 12) / 250
    assert s.iloc[-1] == pytest.approx(expected_end)
    clipped = soc.reconstruct(c * 5, d * 0, m, soc0=0.2, clip=True)
    assert clipped.max() <= 1.0


def test_calibration_recovers_parameters():
    c, d = _profile()
    true = soc.SocModel(energy_mwh=262, eta_charge=0.94, eta_discharge=0.94, standby_mw=0.4)
    rng = np.random.default_rng(1)
    obs = soc.reconstruct(c, d, true, soc0=0.3) + rng.normal(0, 0.001, len(c))
    obs.iloc[3000] += 0.3                                       # a BMS reset: removed as an outlier
    cal = soc.calibrate(obs, c, d, window="1h")
    assert cal.energy_mwh == pytest.approx(262, rel=0.01)
    assert cal.round_trip_efficiency == pytest.approx(0.94 ** 2, abs=0.005)
    assert cal.model.standby_mw == pytest.approx(0.4, abs=0.05)
    assert cal.n_outliers >= 1


def test_window_sums_drop_incomplete_windows():
    c, d = _profile(days=1)
    s = soc.reconstruct(c, d, soc.SocModel(200), 0.5)
    c2 = c.copy()
    c2.iloc[30] = np.nan                                         # 02:30 -> window 02:00 incomplete
    w = soc.window_sums(s, c2, d, "1h")
    assert pd.Timestamp("2026-06-01 02:00") not in w.index
    assert pd.Timestamp("2026-06-01 03:00") in w.index
    assert (w["H"] == 1.0).all()


def test_error_metrics_and_drift():
    idx = pd.date_range("2026-07-01", periods=288 * 10, freq="5min")
    obs = pd.Series(0.5, idx)
    rec = obs + np.arange(len(idx)) / 288 * 0.01                  # +1 pp per day
    m = soc.error_metrics(obs, rec)
    assert m["drift_pp_per_day"] == pytest.approx(1.0, rel=1e-6)
    assert m["bias_pp"] > 0 and m["max_abs_pp"] == pytest.approx((len(idx) - 1) / 288, rel=1e-6)


def test_reanchored_errors_restart_every_day():
    c, d = _profile(days=3)
    true = soc.SocModel(energy_mwh=200, eta_charge=0.95, eta_discharge=0.95)
    obs = soc.reconstruct(c, d, true, soc0=0.3)
    wrong = soc.SocModel(energy_mwh=180, eta_charge=0.95, eta_discharge=0.95)
    e = soc.reanchored_errors(obs, c, d, wrong)
    assert e.groupby("anchor")["error_pp"].first().abs().max() == pytest.approx(0.0)
    assert e["hours"].max() == pytest.approx(23 + 55 / 60)
    assert e["error_pp"].abs().max() > 1
