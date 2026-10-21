import numpy as np
import pytest

from bess_degradation.rainflow import count_cycles, depth_histogram, reversals


def test_astm_e1049_example():
    # ASTM E1049-85 section 5.4.4 example: ranges 3 (0.5), 4 (1.5), 6 (0.5), 8 (1.0), 9 (0.5)
    cycles = count_cycles([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    got = cycles.groupby("depth")["count"].sum().to_dict()
    assert got == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


def test_full_cycles_of_a_repeated_profile():
    # ten daily cycles 0.1 -> 0.9 -> 0.1: ten cycles of depth 0.8 (the open ends are two halves of one)
    x = np.r_[np.tile(np.r_[np.linspace(0.1, 0.9, 50), np.linspace(0.9, 0.1, 50)[1:-1]], 10), 0.1]
    c = count_cycles(x)
    assert (c["depth"] * c["count"]).sum() == pytest.approx(10 * 0.8, rel=1e-9)
    assert c["depth"].max() == pytest.approx(0.8)
    assert c["mean"].round(9).unique().tolist() == [0.5]


def test_nested_small_cycle_is_counted_inside_the_large_one():
    # 0 -> 1 with a 0.6 -> 0.4 -> 0.6 wiggle on the way up, back to 0
    x = [0.0, 0.6, 0.4, 1.0, 0.0]
    c = count_cycles(x)
    full = c[c["count"] == 1.0]
    assert len(full) == 1 and full["depth"].iloc[0] == pytest.approx(0.2)
    assert c.loc[c["count"] == 0.5, "depth"].tolist() == pytest.approx([1.0, 1.0])


def test_reversals_drop_plateaus_and_nans():
    idx, v = reversals([0, 1, 1, 1, 2, np.nan, 1, 0, 0])
    assert v.tolist() == [0, 2, 0]
    assert idx.tolist() == [0, 4, 7]


def test_hysteresis_filter_removes_noise():
    rng = np.random.default_rng(0)
    base = np.r_[np.linspace(0, 1, 200), np.linspace(1, 0, 200)]
    noisy = base + rng.normal(0, 0.01, base.size)
    assert len(count_cycles(noisy)) > 50
    c = count_cycles(noisy, min_range=0.08)
    assert (c["depth"] * c["count"]).sum() == pytest.approx(1.0, abs=0.05)


def test_depth_histogram_counts_efc():
    c = count_cycles([0, 1, 0, 0.5, 0, 0.5, 0])
    h = depth_histogram(c, bins=[0, 0.5, 1.0])
    assert h["efc"].sum() == pytest.approx((c["depth"] * c["count"]).sum())
