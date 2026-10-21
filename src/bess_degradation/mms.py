"""Reader for AEMO "MMS CSV" files and a downloader for the monthly MMSDM archive on NEMWeb.

AEMO publishes NEM market data as zipped CSV files in which every line starts with a record type:

    C,...                                  comment, file header or footer
    I,<package>,<table>,<version>,<cols>   column header of the table that follows
    D,<package>,<table>,<version>,<vals>   data row

The monthly archive (``MMSDM_Historical_Data_SQLLoader``) has one zip per table and month, e.g.
``PUBLIC_ARCHIVE#DISPATCH_UNIT_SCADA#FILE01#202510010000.zip``; large tables can be split into FILE01, FILE02, ...
Files are streamed row by row and filtered on the fly, so a 1.5 GB monthly CSV never has to fit in memory.

Usage:
    from bess_degradation.mms import read_mms_lines, download_table
    df = read_mms_lines(open("PUBLIC_ARCHIVE#DUDETAIL#FILE01#202509010000.CSV"), where={"DUID": {"HPR1"}})
    zips = download_table("DISPATCHPRICE", 2025, 10, Path("data/raw"))
"""
from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
USER_AGENT = "bess-soc-soh-degradation (open-source research; github)"
log = logging.getLogger(__name__)


def mmsdm_url(table: str, year: int, month: int, part: int = 1) -> str:
    """URL of one monthly MMSDM archive file (``#`` is URL-encoded as ``%23``)."""
    folder = f"{BASE_URL}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA"
    return f"{folder}/PUBLIC_ARCHIVE%23{table}%23FILE{part:02d}%23{year}{month:02d}010000.zip"


def read_mms_lines(lines: Iterable[str], columns: list[str] | None = None,
                   where: Mapping[str, Collection[str]] | None = None) -> pd.DataFrame:
    """Parse the data rows of an MMS CSV stream into a DataFrame of strings.

    ``columns`` limits the returned columns (names missing from the file are ignored, so one call works across data
    model versions); ``where`` keeps only rows whose column value is in the given set, e.g. ``{"DUID": {"HPR1"}}``.
    If the stream holds several tables, the header of each one is honoured and the output columns are those of the
    first table.
    """
    out: list[list[str]] = []
    out_cols: list[str] | None = None
    take: list[int] = []
    tests: list[tuple[int, Collection[str]]] = []
    for row in csv.reader(lines):
        if not row:
            continue
        kind = row[0]
        if kind == "I":
            header = row[4:]
            if out_cols is None:
                out_cols = [c for c in (columns or header) if c in header]
            pos = {c: i + 4 for i, c in enumerate(header)}
            take = [pos.get(c, -1) for c in out_cols]
            tests = [(pos.get(c, -1), set(v)) for c, v in (where or {}).items()]
            continue
        if kind != "D" or out_cols is None:
            continue
        if any(i < 0 or row[i] not in vals for i, vals in tests):
            continue
        out.append([row[i] if 0 <= i < len(row) else "" for i in take])
    return pd.DataFrame(out, columns=out_cols if out_cols is not None else list(columns or []))


def read_mms_zip(path: Path, columns: list[str] | None = None,
                 where: Mapping[str, Collection[str]] | None = None) -> pd.DataFrame:
    """Read every CSV member of a zip file with :func:`read_mms_lines`."""
    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.upper().endswith(".CSV"):
                continue
            with zf.open(name) as raw:
                frames.append(read_mms_lines(io.TextIOWrapper(raw, encoding="utf-8", newline=""), columns, where))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or [])


def to_timestamp(s: pd.Series) -> pd.Series:
    """AEMO datetimes ("2025/10/01 00:05:00") to naive pandas timestamps in NEM market time (AEST, UTC+10)."""
    return pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S")


def _download(session: requests.Session, url: str, dest: Path, retries: int = 4) -> bool:
    """Stream ``url`` to ``dest``. Returns False on HTTP 404 (file does not exist)."""
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return True
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            log.warning("download failed (%s), retrying", e)
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def download_table(table: str, year: int, month: int, raw_dir: Path,
                   session: requests.Session | None = None) -> list[Path]:
    """Download all parts (FILE01, FILE02, ...) of one table for one month; returns the local zip paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    paths = []
    for part in range(1, 20):
        dest = raw_dir / f"{table}_{year}{month:02d}_{part:02d}.zip"
        if not dest.exists():
            url = mmsdm_url(table, year, month, part)
            log.info("downloading %s", url.rsplit("/", 1)[-1].replace("%23", "#"))
            if not _download(session, url, dest):
                break
        paths.append(dest)
    if not paths:
        raise FileNotFoundError(f"{table} {year}-{month:02d} not found on NEMWeb")
    return paths
