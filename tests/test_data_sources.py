import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from bess_degradation import acen, cellfit, cells
from bess_degradation.aging import AgingModel
from bess_degradation.remotezip import RemoteZip

# --------------------------------------------------------------------------------------------------------- ACEN CSV

ACEN_HEADER = "t_stamp," + ",".join(f"f{i}" for i in range(20)) + \
    ",Duid1_Status_acPowerStatus_realPowerW1,Duid1_Status_socStatus_socPct1," \
    "Duid2_Status_acPowerStatus_realPowerW1,Duid2_Status_socStatus_socPct1\n"


def test_acen_parse_both_time_formats(tmp_path):
    old = ACEN_HEADER + "1/05/2026 0:00," + ",".join(["0"] * 20) + ",-50000,40.5,0,30\n" + \
        "1/05/2026 0:05," + ",".join(["0"] * 20) + ",-49000,41.0,0,30\n"
    new = ACEN_HEADER + '"2026-09-01 00:00:00.000",' + "," * 19 + ',"90000.5","60.0","0.0","25.0"\n'
    (tmp_path / "2026-05.csv").write_text(old)
    (tmp_path / "2026-09.csv").write_text(new)
    df = acen.load(tmp_path)
    assert df.index.tolist() == pd.to_datetime(["2026-05-01 00:00", "2026-05-01 00:05", "2026-09-01 00:00"]).tolist()
    assert df["mw_1"].tolist() == pytest.approx([-50.0, -49.0, 90.0005])
    assert df["soc_1"].tolist() == pytest.approx([0.405, 0.41, 0.60])
    assert df["soc_2"].iloc[-1] == pytest.approx(0.25)
    assert "fcas_RAISEREG_1" in df and "fcas_LOWER5MIN_2" in df


# ------------------------------------------------------------------------------------------------------ remote zip

def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Data/a.txt", "hello " * 1000, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("Data/sub/b.bin", bytes(range(256)), compress_type=zipfile.ZIP_STORED)
    return buf.getvalue()


def test_remote_zip_reads_members_with_range_requests(tmp_path):
    blob = _zip_bytes()
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return blob[start:end + 1]

    rz = RemoteZip("mem://x", fetch=fetch, size=len(blob))
    assert [m.name for m in rz.members] == ["Data/a.txt", "Data/sub/b.bin"]
    assert rz.read("Data/a.txt") == b"hello " * 1000
    out = rz.extract("Data/sub/b.bin", tmp_path)
    assert out.read_bytes() == bytes(range(256))
    assert all(e - s < len(blob) for s, e in calls)               # never the whole file in one request
    with pytest.raises(KeyError):
        rz.read("nope")


def test_remote_zip_rejects_unsafe_names(tmp_path):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as zf:
        zf.writestr("../evil.txt", "x")
    data = blob.getvalue()
    rz = RemoteZip("mem://x", fetch=lambda s, e: data[s:e + 1], size=len(data))
    with pytest.raises(ValueError):
        rz.extract("../evil.txt", tmp_path)


def test_wanted_members():
    assert cells.wanted("Data/Capacities in checkups - all cells.xlsx")
    assert cells.wanted("Data/Checkups/Checkup_500/Capacity/CA-181103-19_3368_Checkup_500_cycles.txt")
    assert not cells.wanted("Data/Checkups/Checkup_500/IR/CA-181103-19_3379_Checkup_500_cycles.mat")
    assert not cells.wanted("Data/Cycling/CA-181103-19_complete_cycling_data.mat")


# ------------------------------------------------------------------------------------------------- checkup files

def _cycle(q_ah=180.0, current=50.0, n=2, dt_h=0.01):
    """Pause, then n x (CC charge, pause, CC discharge, pause) on an LFP-like voltage curve."""
    rows, t, ah = [], 0.0, 0.0
    hours = q_ah / current

    def volt(frac):                                               # flat plateau with two steps (ICA peaks)
        return 2.5 + 1.1 * frac + 0.03 * np.tanh((frac - 0.3) * 40) + 0.03 * np.tanh((frac - 0.7) * 40)

    def add(cmd, i, steps, frac0, dfrac, line):
        nonlocal t, ah
        for k in range(steps):
            frac = frac0 + dfrac * k / max(steps - 1, 1)
            rows.append((t, k, line, cmd, volt(frac) if cmd != "Pause" else 3.0, i, ah, 0, 0, 25.0, 1, 1))
            t += dt_h
            ah += i * dt_h
    add("Pause", 0.0, 10, 0, 0, 2)
    for _ in range(n):
        add("Charge", current, int(round(hours / dt_h)) + 1, 0.0, 1.0, 6)
        add("Pause", 0.0, 10, 0, 0, 8)
        add("Discharge", -current, int(round(hours / dt_h)) + 1, 1.0, -1.0, 9)
        add("Pause", 0.0, 10, 0, 0, 11)
    return rows


def _basytec_text(rows):
    head = ("~Resultfile from Basytec Battery Test System\n~Name of Test: Checkup_500_cycles\n"
            "~Battery: CA-181103-19\n~Start of Test: 13.11.2019 08:00:00\n~\n"
            "~Time[h]\tDataSet\tLine\tCommand\tU[V]\tI[A]\tAh[Ah]\tAh-Step\tWh[Wh]\tT1[°C]\tCyc-Count\tState\n")
    return head + "".join("\t".join(str(v) for v in r) + "\n" for r in rows)


def test_basytec_conversion_and_coulomb_counting(tmp_path):
    txt = tmp_path / "x.txt"
    txt.write_text(_basytec_text(_cycle(q_ah=180.0)), encoding="latin-1")
    out = cells.convert_checkup(txt, tmp_path / "checkups" / "CA-181103-19" / "0500_x.parquet")
    df, meta = cells.read_checkup(out)
    assert meta["test"] == "Checkup_500_cycles" and meta["start"].startswith("2019-11-13 08:00")
    seg = cellfit.cc_segments(df)
    dis = seg[seg["current_a"] < 0]
    assert len(dis) == 2
    assert (-dis["ah"]).tolist() == pytest.approx([180.0, 180.0], rel=2e-3)


def test_checkup_table_and_ica(tmp_path):
    for checkup, q in ((0, 200.0), (100, 196.0)):
        txt = tmp_path / f"{checkup}.txt"
        txt.write_text(_basytec_text(_cycle(q_ah=q)), encoding="latin-1")
        cells.convert_checkup(txt, cells.checkup_parquet_path(
            tmp_path, f"Data/Checkups/Checkup_{checkup}/Capacity/CA-181103-19_1_test.txt"))
    caps = cellfit.checkup_table(tmp_path)
    assert caps["q_dis_ah"].tolist() == pytest.approx([200.0, 196.0], rel=2e-3)
    assert caps["mode"].unique().tolist() == ["cycling"] and caps["temp_c"].iloc[0] == 35
    df, _ = cells.read_checkup(next((tmp_path / "checkups").glob("*/0100_*.parquet")))
    curve = cellfit.ica_curve(df, 50.0, "charge", v_range=(2.4, 3.7), window=0)
    # the whole charge is accounted for, and the steep steps at 30 % and 70 % of the charge make dQ/dV dips
    assert curve.sum() * 0.002 == pytest.approx(196.0, rel=0.02)
    step = curve[(curve.index > 2.80) & (curve.index < 2.86)]          # steep step at 30 % of the charge
    plateau = curve[(curve.index > 3.00) & (curve.index < 3.10)]       # flat part between the steps
    assert step.min() < 0.6 * plateau.mean()
    assert plateau.mean() == pytest.approx(196.0 / 1.1, rel=0.05)


def test_fit_calendar_recovers_synthetic_parameters():
    true = AgingModel(k_cal=1.5e-3, ea_cal=35_000, gamma=0.8, z_cal=0.5)
    rows = []
    for i, (temp, soc) in enumerate([(35, 75), (35, 100), (50, 75), (50, 100)] * 2):
        c0 = 195.0 + i
        for k, days in enumerate((80, 230, 530, 800), start=1):
            q = c0 * (1 - float(true.calendar_loss(days, soc / 100, temp)))
            rows.append({"cell": f"c{i}", "mode": "calendar", "temp_c": temp, "storage_soc": soc,
                         "checkup": k * 100, "current_a": 50.0, "days": days, "q_dis_ah": q})
    res = cellfit.fit_calendar(pd.DataFrame(rows), z_fixed=0.5)
    assert res.params["ea_cal"] == pytest.approx(35_000, rel=0.02)
    assert res.params["gamma"] == pytest.approx(0.8, abs=0.05)
    assert res.params["k_cal"] == pytest.approx(1.5e-3, rel=0.05)
    assert res.rmse_pct < 0.01
