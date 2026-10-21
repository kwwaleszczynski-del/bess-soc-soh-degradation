"""ACEN Australia's published operational data of the New England BESS (NSW): dispatch, state of charge, FCAS.

The battery (2 x 100 MW / 200 MWh, AEMO units NESBESS1 and NESBESS2) is co-located with New England Solar near
Uralla. ACEN publishes one CSV per month on https://acenrenewables.com.au/new-england-bess-operational-data/ .
No licence is stated on the page, so the files are only downloaded to ``data/acen`` for local use and are not
redistributed; this project publishes derived statistics only.

Usage:
    from bess_degradation.acen import fetch, load
    fetch(Path("data/acen"))
    df = load(Path("data/acen"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from bess_degradation.remotezip import USER_AGENT

log = logging.getLogger(__name__)

PAGE = "https://acenrenewables.com.au/new-england-bess-operational-data/"
# month -> CSV link as listed on the page in October 2026 (file names do not follow a pattern)
MONTHLY_CSV = {
    "2026-05": "https://assets.acenrenewables.com.au/uploads/2026/06/nebess_eep_monthly_may.csv",
    "2026-06": "https://assets.acenrenewables.com.au/uploads/2026/07/export_20260706_163256.csv",
    "2026-07": "https://assets.acenrenewables.com.au/uploads/2026/09/export_20260811_094003_july.csv",
    "2026-08": "https://assets.acenrenewables.com.au/uploads/2026/09/export_20260902_131617_august.csv",
    "2026-09": "https://assets.acenrenewables.com.au/uploads/2026/10/export_20261008_112335.csv",
}
DUIDS = ["NESBESS1", "NESBESS2"]


FCAS_ORDER = ["RAISEREG", "LOWERREG", "RAISE1SEC", "RAISE6SEC", "RAISE60SEC", "RAISE5MIN",
              "LOWER1SEC", "LOWER6SEC", "LOWER60SEC", "LOWER5MIN"]


def parse(path: Path) -> pd.DataFrame:
    """One monthly file -> frame indexed by interval start (NEM time) with, per section k = 1, 2: ``mw_k`` (average
    power over the 5-minute interval, + discharge / - charge), ``soc_k`` (fraction) and ``fcas_<service>_k`` (MW).

    Format notes (as published): the time stamp format changed from ``1/05/2026 0:00`` to ``2026-09-01 00:00:00.000``;
    power is labelled "realPowerW" but is in kW; several FCAS column labels are duplicated, so the 20 FCAS columns
    are mapped by position (outputs ao0010-ao0019 for section 1, ao0020-ao0029 for section 2, in AEMO's order). The
    time stamp is the START of the interval: the power matches the mean of AEMO's SCADA readings at the start and end
    of the interval (correlation 0.997), and the SoC is the reading at the start.
    """
    raw = pd.read_csv(path, header=0, dtype=str)
    stamp = raw.iloc[:, 0].str.strip()
    fmt = "%d/%m/%Y %H:%M" if stamp.str.contains("/").any() else "%Y-%m-%d %H:%M:%S.%f"
    idx = pd.DatetimeIndex(pd.to_datetime(stamp, format=fmt), name="interval_start")
    num = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").set_axis(idx)
    out = pd.DataFrame(index=idx)
    for k, (p_col, s_col) in ((1, (-4, -3)), (2, (-2, -1))):
        out[f"mw_{k}"] = num.iloc[:, p_col] / 1000.0
        out[f"soc_{k}"] = num.iloc[:, s_col] / 100.0
        for i, service in enumerate(FCAS_ORDER):
            out[f"fcas_{service}_{k}"] = num.iloc[:, (k - 1) * 10 + i]
    return out


def load(data_dir: Path) -> pd.DataFrame:
    """All downloaded months, without the duplicated stamp at each month boundary."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no ACEN files in {data_dir}; run the 'fetch' command first")
    df = pd.concat([parse(f) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="first")]


def fetch(data_dir: Path, refresh: bool = False) -> list[Path]:
    """Download the monthly CSV files to ``data_dir/<YYYY-MM>.csv``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    paths = []
    for month, url in MONTHLY_CSV.items():
        out = data_dir / f"{month}.csv"
        if refresh or not out.exists():
            log.info("downloading ACEN %s", month)
            r = session.get(url, timeout=120)
            r.raise_for_status()
            out.write_bytes(r.content)
        paths.append(out)
    return paths
