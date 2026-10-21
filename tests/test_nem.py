import numpy as np
import pandas as pd
import pytest

from bess_degradation import nem


def test_split_energy_exact_for_linear_ramps():
    dis, chg = nem.split_energy(np.array([10.0, -10.0, 10.0, 0.0]), np.array([20.0, -30.0, -30.0, 0.0]), 1.0)
    np.testing.assert_allclose(dis, [15.0, 0.0, 10 ** 2 / (2 * 40), 0.0])
    np.testing.assert_allclose(chg, [0.0, 20.0, 30 ** 2 / (2 * 40), 0.0])
    np.testing.assert_allclose(dis - chg, [15.0, -20.0, -10.0, 0.0])


def _synthetic(days=2):
    """One battery in SA1 charging 10-14 h and discharging 18-20 h, reporting its stored energy."""
    end = pd.date_range("2025-10-01 00:05", periods=288 * days, freq="5min")     # AEMO interval-end labels
    start = end - pd.Timedelta("5min")
    hour = start.hour
    price = np.where((hour >= 10) & (hour < 14), -20.0, np.where(hour >= 18, 300.0, 60.0))
    mw = np.where((hour >= 10) & (hour < 14), -50.0, np.where((hour >= 18) & (hour < 20), 100.0, 0.0))
    scada = pd.DataFrame({"SETTLEMENTDATE": end, "DUID": "BAT1", "SCADAVALUE": mw})
    prices = pd.DataFrame({"SETTLEMENTDATE": end, "REGIONID": "SA1", "RRP": price,
                           **{f"{s}RRP": 0.0 for s in nem.FCAS_SERVICES}})
    prices["RAISE6SECRRP"] = 2.0
    disp = pd.DataFrame({"SETTLEMENTDATE": end, "DUID": "BAT1", "INTERVENTION": 0,
                         "INITIAL_ENERGY_STORAGE": np.arange(len(end), dtype=float),
                         **{s: 0.0 for s in nem.FCAS_SERVICES}})
    disp["RAISE6SEC"] = 10.0
    return scada, disp, prices


def test_unit_frame_alignment_energy_and_fcas():
    scada, disp, prices = _synthetic()
    iv = nem.unit_frame(scada, disp, prices, "BAT1", "SA1")
    assert iv.index[0] == pd.Timestamp("2025-10-01 00:00")
    assert (iv.index[1] - iv.index[0]) / pd.Timedelta(minutes=1) == 5
    assert iv["valid"].all()
    day = iv.loc["2025-10-01"]
    assert day["charge_mwh"].sum() == pytest.approx(200.0, rel=1e-6)
    assert day["discharge_mwh"].sum() == pytest.approx(200.0, rel=1e-6)
    assert day["fcas_aud"].sum() == pytest.approx(10 * 2 * 24, rel=1e-9)
    # the value labelled 00:05 by AEMO is the stored energy at the start of the interval beginning 00:00
    assert iv["reported_mwh"].iloc[0] == 0.0 and iv["reported_mwh"].iloc[1] == 1.0


def test_battery_table_and_params_by_day():
    reg = pd.DataFrame({
        "DUID": ["B1", "B1"], "EFFECTIVEDATE": pd.to_datetime(["2025-01-01", "2026-03-01"]), "VERSIONNO": [1, 1],
        "REGISTEREDCAPACITY": [100, 100], "MAXCAPACITY": [100, 100], "MINCAPACITY": [0, 80],
        "MAXSTORAGECAPACITY": [200.0, 300.0], "STORAGEIMPORTEFFICIENCYFACTOR": [0.9, 0.92],
        "STORAGEEXPORTEFFICIENCYFACTOR": [0.9, 0.93], "REGIONID": "NSW1", "STATIONID": "S1", "PARTICIPANTID": "P"})
    t = nem.battery_table(reg, pd.DataFrame({"STATIONID": ["S1"], "STATIONNAME": [" Battery One "]}))
    assert t.loc["B1", "name"] == "Battery One"
    assert t.loc["B1", "energy_mwh"] == 300.0 and t.loc["B1", "charge_mw"] == 80
    p = nem.params_by_day(reg, "B1", pd.date_range("2026-02-27", periods=4, freq="D"))
    assert p["energy_mwh"].tolist() == [200.0, 200.0, 300.0, 300.0]
