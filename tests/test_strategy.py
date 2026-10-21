import numpy as np
import pandas as pd
import pytest

from bess_degradation import pipeline
from bess_degradation.strategy import DT_5MIN, AgingCost, Battery, optimize_day

LOSSLESS = Battery(discharge_mw=1.0, charge_mw=1.0, energy_mwh=1.0, round_trip_efficiency=1.0)


def test_buys_low_sells_high():
    prices = np.r_[np.full(144, 20.0), np.full(144, 300.0)]
    res = optimize_day(prices, LOSSLESS, AgingCost(throughput=1.0), e_start=0.0)
    assert res.revenue == pytest.approx(280.0, rel=1e-6)
    assert res.objective == pytest.approx(279.0, rel=1e-6)
    assert res.schedule["charge_mw"].iloc[144:].sum() == pytest.approx(0.0, abs=1e-9)


def test_constraints_hold_with_segments():
    rng = np.random.default_rng(0)
    prices = rng.normal(80, 150, 288)
    b = Battery(discharge_mw=100, charge_mw=80, energy_mwh=150, round_trip_efficiency=0.85)
    cost = AgingCost(throughput=1.0, segment=np.array([1.0, 2.0, 5.0, 10.0]), holding=0.5)
    res = optimize_day(prices, b, cost, e_start=50, max_discharge_mwh=150)
    s = res.schedule
    assert s["energy_mwh"].min() >= -1e-6 and s["energy_mwh"].max() <= 150 + 1e-6
    assert s["energy_mwh"].iloc[-1] >= 50 - 1e-6
    assert ((s["charge_mw"] > 1e-6) & (s["discharge_mw"] > 1e-6)).sum() == 0
    assert (s["charge_mw"] / 80 + s["discharge_mw"] / 100).max() <= 1 + 1e-6
    assert res.discharged_mwh <= 150 + 1e-6
    eta = b.eta
    e = 50 + np.cumsum(eta * DT_5MIN * s["charge_mw"] - DT_5MIN * s["discharge_mw"] / eta)
    np.testing.assert_allclose(e, s["energy_mwh"], atol=1e-6)


def test_ageing_cost_reduces_cycling():
    prices = np.tile(np.r_[np.full(36, 10.0), np.full(36, 60.0)], 4)          # four 50 AUD spreads a day
    b = Battery(1, 1, 1, 0.9)
    cheap = optimize_day(prices, b, AgingCost(throughput=1.0), e_start=0.0)
    dear = optimize_day(prices, b, AgingCost(throughput=1.0, segment=np.array([60.0])), e_start=0.0)
    assert cheap.discharged_mwh > 2.0
    assert dear.discharged_mwh == pytest.approx(0.0, abs=1e-6)


def test_depth_dependent_cost_prefers_shallow_cycles():
    # a small spread pays only for the cheap (shallow) segment; a large spread also for the deep one
    prices = np.r_[np.full(72, 50.0), np.full(72, 80.0), np.full(72, 50.0), np.full(72, 400.0)]
    b = Battery(10, 10, 10, 1.0)
    cost = AgingCost(throughput=0.0, segment=np.array([5.0, 100.0]))
    res = optimize_day(prices, b, cost, e_start=0.0)
    s = res.schedule
    first_peak = s["discharge_mw"].iloc[72:144].sum() * DT_5MIN
    last_peak = s["discharge_mw"].iloc[216:].sum() * DT_5MIN
    assert first_peak == pytest.approx(5.0, rel=1e-6)                # only the shallow half
    assert last_peak == pytest.approx(10.0, rel=1e-6)                # the whole battery


def test_run_year_chains_days():
    idx = pd.date_range("2026-01-01", periods=288 * 3, freq="5min")
    hour = idx.hour
    prices = pd.Series(np.where(hour < 12, 20.0, 200.0), idx)
    daily, prof = pipeline.run_year(prices, Battery(1, 1, 2, 0.9), AgingCost(throughput=1.0), soc_start=0.5)
    assert len(daily) == 3 and len(prof) == 288 * 3
    assert prof.iloc[0] == pytest.approx(0.5)
    assert prof.between(0, 1).all()
    capped, _ = pipeline.run_year(prices, Battery(1, 1, 2, 0.9), AgingCost(throughput=1.0), max_cycles=0.25)
    assert (capped["discharged_mwh"] <= 0.5 + 1e-6).all()
