import io
import zipfile

import pandas as pd
import pytest

from bess_degradation import nem as data
from bess_degradation.mms import mmsdm_url, read_mms_lines, read_mms_zip, to_timestamp

SCADA = """C,NEMP.WORLD,DISPATCH_UNIT_SCADA,AEMO,PUBLIC,2025/11/01,00:00:00,0000000001,MMSDM,0000000001
I,DISPATCH,UNIT_SCADA,1,SETTLEMENTDATE,DUID,SCADAVALUE,LASTCHANGED
D,DISPATCH,UNIT_SCADA,1,"2025/10/01 00:05:00",HPR1,9,"2025/10/01 00:00:05"
D,DISPATCH,UNIT_SCADA,1,"2025/10/01 00:05:00",OTHER1,55.5,"2025/10/01 00:00:05"
D,DISPATCH,UNIT_SCADA,1,"2025/10/01 00:10:00",HPR1,-14.9,"2025/10/01 00:05:05"
C,"END OF REPORT",5
"""


def test_read_filters_rows_and_columns():
    df = read_mms_lines(io.StringIO(SCADA), columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE", "NOT_IN_FILE"],
                        where={"DUID": {"HPR1"}})
    assert list(df.columns) == ["SETTLEMENTDATE", "DUID", "SCADAVALUE"]
    assert df["DUID"].tolist() == ["HPR1", "HPR1"]
    assert df["SCADAVALUE"].tolist() == ["9", "-14.9"]
    assert to_timestamp(df["SETTLEMENTDATE"]).iloc[1] == pd.Timestamp("2025-10-01 00:10")


def test_read_all_columns_and_empty_result():
    df = read_mms_lines(io.StringIO(SCADA))
    assert len(df) == 3 and "LASTCHANGED" in df
    empty = read_mms_lines(io.StringIO(SCADA), columns=["DUID"], where={"DUID": {"NOPE"}})
    assert empty.empty and list(empty.columns) == ["DUID"]


def test_header_of_a_second_table_is_honoured():
    text = SCADA + 'I,DISPATCH,UNIT_SCADA,2,DUID,SETTLEMENTDATE,SCADAVALUE\nD,DISPATCH,UNIT_SCADA,2,HPR1,"2025/10/01 00:15:00",3\n'
    df = read_mms_lines(io.StringIO(text), columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"], where={"DUID": {"HPR1"}})
    assert df.iloc[-1].tolist() == ["2025/10/01 00:15:00", "HPR1", "3"]


def test_mmsdm_url():
    url = mmsdm_url("DISPATCH_UNIT_SCADA", 2025, 10, 2)
    assert url.endswith("/2025/MMSDM_2025_10/MMSDM_Historical_Data_SQLLoader/DATA/"
                        "PUBLIC_ARCHIVE%23DISPATCH_UNIT_SCADA%23FILE02%23202510010000.zip")


def _zip(path, name, text):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, text)
    return path


def test_read_zip(tmp_path):
    z = _zip(tmp_path / "x.zip", "PUBLIC_ARCHIVE#DISPATCH_UNIT_SCADA#FILE01#202510010000.CSV", SCADA)
    assert len(read_mms_zip(z, where={"DUID": {"OTHER1"}})) == 1


def test_fetch_month_caches_filtered_parquet(tmp_path, monkeypatch):
    calls = []

    def fake_download(table, year, month, raw_dir, session=None):
        calls.append(table)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return [_zip(raw_dir / f"{table}.zip", f"{table}.CSV", SCADA)]

    monkeypatch.setattr(data, "download_table", fake_download)
    out = data.fetch_month("scada", 2025, 10, tmp_path, {"HPR1"}, session=None)
    df = pd.read_parquet(out)
    assert df["DUID"].astype(str).tolist() == ["HPR1", "HPR1"]
    assert df["SCADAVALUE"].tolist() == pytest.approx([9.0, -14.9])
    assert str(df["SETTLEMENTDATE"].dtype).startswith("datetime64")
    assert not (tmp_path / "raw" / "DISPATCH_UNIT_SCADA.zip").exists()       # raw zip removed after caching
    data.fetch_month("scada", 2025, 10, tmp_path, {"HPR1"}, session=None)    # cached: no second download
    assert calls == ["DISPATCH_UNIT_SCADA"]
    assert len(data.load("scada", tmp_path, duids=["HPR1"])) == 2


def test_months():
    from datetime import date
    assert data.months(date(2025, 11, 15), date(2026, 1, 2)) == [(2025, 11), (2025, 12), (2026, 1)]
