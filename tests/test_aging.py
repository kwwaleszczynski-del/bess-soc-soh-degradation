import numpy as np
import pandas as pd
import pytest

from bess_degradation import aging


def test_arrhenius_reference_and_direction():
    assert aging.arrhenius(25.0, 30_000) == pytest.approx(1.0)
    assert aging.arrhenius(35.0, 30_000) > 1.0 > aging.arrhenius(15.0, 30_000)


def test_virtual_time_equals_closed_form_at_constant_conditions():
    m = aging.AgingModel(k_cal=2e-3, gamma=1.0, z_cal=0.5, k_cyc=1e-3, z_cyc=0.5)
    soc = np.full(288 * 365, 0.7)                                 # one year at a constant SoC, no cycles
    st = aging.profile_stress(soc)
    path = aging.soh_path(m, st, years=4, temp_c=30)
    t = path["years"].to_numpy() * 365.25
    np.testing.assert_allclose(path["calendar_loss"], m.calendar_loss(t, 0.7, 30), rtol=1e-2)
    assert path["cycle_loss"].abs().max() == 0.0


def test_mixed_soc_rate_lies_between_the_extremes():
    m = aging.AgingModel(gamma=2.0)
    st = aging.profile_stress(np.r_[np.full(1000, 0.1), np.full(1000, 0.9)])
    k = aging.calendar_rate(m, st, 25)
    assert m.k_cal_at(0.1, 25) < k < m.k_cal_at(0.9, 25)


def test_cycle_stress_and_soh_path():
    day = np.r_[np.linspace(0.1, 0.9, 144), np.linspace(0.9, 0.1, 144)]
    st = aging.profile_stress(np.tile(day, 365))
    assert st.fec_per_year == pytest.approx(0.8 * 365, rel=0.01)
    m = aging.AgingModel(beta=1.5)
    assert st.n_eq_per_year(1.5) == pytest.approx(0.8 ** 1.5 * 365, rel=0.01)
    p = aging.soh_path(m, st, years=20)
    assert (np.diff(p["soh"]) < 0).all()
    y = aging.years_to(p, 0.8)
    i = int(np.searchsorted(-p["soh"].to_numpy(), -0.8))
    assert p["years"].iloc[i - 1] <= y <= p["years"].iloc[i]


def test_years_to_is_nan_when_never_reached():
    p = pd.DataFrame({"years": [0, 1, 2], "soh": [1.0, 0.95, 0.9]})
    assert np.isnan(aging.years_to(p, 0.8))


def test_segment_costs_add_up_to_full_cycle_cost():
    m = aging.AgingModel(k_cyc=2e-5, z_cyc=1.0, beta=1.5)
    g = aging.segment_costs(m, energy_mwh=200, fade_cost_aud=1e8, segments=8)
    assert np.all(np.diff(g) > 0)                                 # deeper segments cost more
    # emptying the whole battery once costs fade_cost * k_cyc
    assert (g * 200 / 8).sum() == pytest.approx(1e8 * 2e-5)
    flat = aging.segment_costs(m.with_(beta=1.0), energy_mwh=200, fade_cost_aud=1e8, segments=8)
    np.testing.assert_allclose(flat, flat[0])


def test_holding_cost_follows_soc_sensitivity():
    assert aging.holding_cost(aging.AgingModel(gamma=1.0), 200, 1e8) > 0
    assert aging.holding_cost(aging.AgingModel(gamma=0.0), 200, 1e8) == pytest.approx(0.0)
