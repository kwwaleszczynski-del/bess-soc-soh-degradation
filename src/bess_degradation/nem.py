"""Local cache of AEMO NEM data (5-minute battery dispatch and prices), built from the monthly MMSDM archive on NEMWeb.

The cache format is the one of the companion project ``nem-battery-behaviour``; an existing cache can be reused with
``--nem-data-dir``.

For every month the raw zips are downloaded, filtered to what the analysis needs and stored as small Parquet files;
the zips are deleted afterwards unless ``keep_raw`` is set. Only bidirectional units (grid-scale batteries registered
as a single "BDU" since the 2024 integrated-storage rule change) are kept.

    registry.parquet               battery registration history: region, MW, MWh, efficiencies  (DUDETAIL, DUDETAILSUMMARY)
    stations.parquet               station names                                                (STATION)
    scada/2025-10.parquet          SCADA MW of every battery, 5-min                             (DISPATCH_UNIT_SCADA)
    price/2025-10.parquet          regional energy and FCAS prices, 5-min                       (DISPATCHPRICE)
    dispatch/2025-10.parquet       dispatch targets and FCAS enablement of every battery, 5-min  (DISPATCHLOAD)

All timestamps are AEMO ``SETTLEMENTDATE`` values: interval END in NEM market time (AEST, UTC+10, no daylight saving).

Usage:
    from bess_degradation.nem import fetch, load
    fetch(date(2025, 10, 1), date(2026, 9, 30), Path("data/nem"))
    scada = load("scada", Path("data/nem"))
    iv = unit_frame(scada, dispatch, prices, "HPR1", "SA1")
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bess_degradation.mms import USER_AGENT, download_table, read_mms_zip, to_timestamp

log = logging.getLogger(__name__)

FCAS_SERVICES = ["RAISE1SEC", "RAISE6SEC", "RAISE60SEC", "RAISE5MIN", "RAISEREG",
                 "LOWER1SEC", "LOWER6SEC", "LOWER60SEC", "LOWER5MIN", "LOWERREG"]

# table -> (MMSDM table name, columns to keep, numeric columns)
MONTHLY = {
    "scada": ("DISPATCH_UNIT_SCADA", ["SETTLEMENTDATE", "DUID", "SCADAVALUE"], ["SCADAVALUE"]),
    "price": ("DISPATCHPRICE",
              ["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "RRP"] + [f"{s}RRP" for s in FCAS_SERVICES],
              ["RRP"] + [f"{s}RRP" for s in FCAS_SERVICES]),
    "dispatch": ("DISPATCHLOAD",
                 ["SETTLEMENTDATE", "DUID", "INTERVENTION", "DISPATCHMODE", "INITIALMW", "TOTALCLEARED",
                  "AVAILABILITY", "INITIAL_ENERGY_STORAGE", "ENERGY_STORAGE"] + FCAS_SERVICES,
                 ["INITIALMW", "TOTALCLEARED", "AVAILABILITY", "INITIAL_ENERGY_STORAGE", "ENERGY_STORAGE"]
                 + FCAS_SERVICES),
}
REGISTRY_COLS = ["DUID", "EFFECTIVEDATE", "VERSIONNO", "REGISTEREDCAPACITY", "MAXCAPACITY", "MINCAPACITY",
                 "MAXSTORAGECAPACITY", "STORAGEIMPORTEFFICIENCYFACTOR", "STORAGEEXPORTEFFICIENCYFACTOR"]
SUMMARY_COLS = ["DUID", "START_DATE", "END_DATE", "DISPATCHTYPE", "REGIONID", "STATIONID", "PARTICIPANTID"]


def months(start: date, end: date) -> list[tuple[int, int]]:
    """(year, month) pairs covering start..end inclusive."""
    return [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def _read_parts(paths: list[Path], columns: list[str], where=None) -> pd.DataFrame:
    return pd.concat([read_mms_zip(p, columns, where) for p in paths], ignore_index=True)


def _cleanup(paths: list[Path], keep_raw: bool):
    if not keep_raw:
        for p in paths:
            p.unlink(missing_ok=True)


def build_registry(year: int, month: int, data_dir: Path, session: requests.Session, keep_raw: bool = False) -> pd.DataFrame:
    """Registration history of all bidirectional units, as published in the MMSDM archive of ``year-month``."""
    raw = data_dir / "raw"
    sp = download_table("DUDETAILSUMMARY", year, month, raw, session)
    summary = _read_parts(sp, SUMMARY_COLS)
    dp = download_table("DUDETAIL", year, month, raw, session)
    detail = _read_parts(dp, REGISTRY_COLS)
    bdu = set(summary.loc[summary["DISPATCHTYPE"] == "BIDIRECTIONAL", "DUID"])
    latest = (summary[summary["DUID"].isin(bdu)].sort_values("START_DATE").groupby("DUID").tail(1)
              [["DUID", "REGIONID", "STATIONID", "PARTICIPANTID"]])
    reg = detail[detail["DUID"].isin(bdu)].merge(latest, on="DUID", how="left")
    reg["EFFECTIVEDATE"] = to_timestamp(reg["EFFECTIVEDATE"])
    for c in REGISTRY_COLS[2:]:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg = reg.sort_values(["DUID", "EFFECTIVEDATE", "VERSIONNO"]).reset_index(drop=True)
    reg.to_parquet(data_dir / "registry.parquet", index=False)
    _cleanup(sp + dp, keep_raw)

    try:
        st = download_table("STATION", year, month, raw, session)
        stations = _read_parts(st, ["STATIONID", "STATIONNAME", "STATE", "LASTCHANGED"])
        stations = stations.sort_values("LASTCHANGED").groupby("STATIONID").tail(1).drop(columns="LASTCHANGED")
        stations.to_parquet(data_dir / "stations.parquet", index=False)
        _cleanup(st, keep_raw)
    except FileNotFoundError:
        log.warning("STATION table not found; batteries will be labelled by DUID")
    log.info("registry: %d bidirectional units", reg["DUID"].nunique())
    return reg


def fetch_month(kind: str, year: int, month: int, data_dir: Path, duids: set[str],
                session: requests.Session, refresh: bool = False, keep_raw: bool = False) -> Path:
    """Download, filter and cache one table for one month; returns the Parquet path."""
    table, columns, numeric = MONTHLY[kind]
    out = data_dir / kind / f"{year}-{month:02d}.parquet"
    if out.exists() and not refresh:
        return out
    paths = download_table(table, year, month, data_dir / "raw", session)
    if kind == "price":
        where = {"INTERVENTION": {"0"}}
    else:
        where = {"DUID": duids}
    df = _read_parts(paths, columns, where)
    df["SETTLEMENTDATE"] = to_timestamp(df["SETTLEMENTDATE"])
    for c in numeric:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ("DUID", "REGIONID"):
        if c in df:
            df[c] = df[c].astype("category")
    if "INTERVENTION" in df:
        df["INTERVENTION"] = pd.to_numeric(df["INTERVENTION"]).astype("int8")
    if "DISPATCHMODE" in df:
        df["DISPATCHMODE"] = pd.to_numeric(df["DISPATCHMODE"], errors="coerce").astype("float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _cleanup(paths, keep_raw)
    log.info("%s %d-%02d: %d rows -> %s", kind, year, month, len(df), out)
    return out


def fetch(start: date, end: date, data_dir: Path, kinds=("scada", "price", "dispatch"),
          refresh: bool = False, keep_raw: bool = False) -> None:
    """Build the registry (from the last month) and cache every month of the requested tables."""
    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    last = months(start, end)[-1]
    if refresh or not (data_dir / "registry.parquet").exists():
        build_registry(*last, data_dir, session, keep_raw)
    duids = set(pd.read_parquet(data_dir / "registry.parquet")["DUID"])
    for y, m in months(start, end):
        for kind in kinds:
            fetch_month(kind, y, m, data_dir, duids, session, refresh, keep_raw)


def load(kind: str, data_dir: Path, start: date | None = None, end: date | None = None,
         columns: list[str] | None = None, duids: list[str] | None = None) -> pd.DataFrame:
    """Concatenate cached monthly Parquet files of one table.

    Optionally only months overlapping start..end, only some columns, and only some units.
    """
    files = sorted((data_dir / kind).glob("*.parquet"))
    if start and end:
        wanted = {f"{y}-{m:02d}" for y, m in months(start, end)}
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(f"no cached {kind} data in {data_dir}; run the 'fetch' command first")
    filters = [("DUID", "in", list(duids))] if duids is not None else None
    df = pd.concat([pd.read_parquet(f, columns=columns, filters=filters) for f in files], ignore_index=True)
    for c in ("DUID", "REGIONID"):
        if c in df:
            df[c] = df[c].astype("category")
    return df


# ---------------------------------------------------------------------------------------------------------------------
# 5-minute frames of one battery
#
# * Time is NEM market time. Intervals are indexed by their START; AEMO labels them by their end (SETTLEMENTDATE), so
#   every AEMO timestamp is shifted back by 5 minutes.
# * DISPATCH_UNIT_SCADA.SCADAVALUE is an instantaneous MW reading at the start of the interval (= INITIALMW); energy
#   in an interval is the integral of the straight line between the readings at its start and end, split exactly into
#   its discharging (MW > 0) and charging (MW < 0) parts.
# * DISPATCHLOAD.INITIAL_ENERGY_STORAGE is the stored energy (MWh) the participant reports at the start of the interval.
# ---------------------------------------------------------------------------------------------------------------------

STEP = pd.Timedelta("5min")
DT_5MIN = 5 / 60


def battery_table(registry: pd.DataFrame, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest registration of every battery: name, region, MW, MWh and registered efficiencies (indexed by DUID)."""
    last = registry.sort_values(["DUID", "EFFECTIVEDATE", "VERSIONNO"]).groupby("DUID").tail(1).set_index("DUID")
    out = pd.DataFrame({
        "region": last["REGIONID"],
        "discharge_mw": last["MAXCAPACITY"],
        "charge_mw": last["MINCAPACITY"].where(last["MINCAPACITY"] > 0, last["MAXCAPACITY"]),
        "energy_mwh": last["MAXSTORAGECAPACITY"],
        "eta_charge": last["STORAGEIMPORTEFFICIENCYFACTOR"],
        "eta_discharge": last["STORAGEEXPORTEFFICIENCYFACTOR"],
    })
    out["name"] = out.index
    if stations is not None:
        names = stations.set_index("STATIONID")["STATIONNAME"].str.strip()
        out["name"] = last["STATIONID"].map(names).fillna(pd.Series(out.index, index=out.index))
    out["duration_h"] = out["energy_mwh"] / out["discharge_mw"]
    return out[["name", "region", "discharge_mw", "charge_mw", "energy_mwh", "duration_h", "eta_charge",
                "eta_discharge"]]


def params_by_day(registry: pd.DataFrame, duid: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Registered MW/MWh valid on each day (registrations change, e.g. when a battery is extended)."""
    reg = registry[registry["DUID"] == duid].sort_values(["EFFECTIVEDATE", "VERSIONNO"])
    reg = reg.groupby("EFFECTIVEDATE").tail(1)
    frame = pd.DataFrame({"day": days})
    m = pd.merge_asof(frame, reg, left_on="day", right_on="EFFECTIVEDATE", direction="backward")
    m = m.bfill()                  # days before the first registration record: use the earliest one
    return pd.DataFrame({
        "discharge_mw": m["MAXCAPACITY"].to_numpy(float),
        "charge_mw": m["MINCAPACITY"].where(m["MINCAPACITY"] > 0, m["MAXCAPACITY"]).to_numpy(float),
        "energy_mwh": m["MAXSTORAGECAPACITY"].to_numpy(float),
    }, index=days)


def split_energy(a: np.ndarray, b: np.ndarray, dt: float = DT_5MIN) -> tuple[np.ndarray, np.ndarray]:
    """Discharged and charged MWh of a straight-line MW trajectory from ``a`` to ``b`` over ``dt`` hours."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    width = np.where(hi > lo, hi - lo, 1.0)
    pos = np.where(lo >= 0, (a + b) / 2, np.where(hi <= 0, 0.0, hi ** 2 / (2 * width)))
    neg = np.where(hi <= 0, -(a + b) / 2, np.where(lo >= 0, 0.0, lo ** 2 / (2 * width)))
    return pos * dt, neg * dt


def unit_frame(scada: pd.DataFrame, dispatch: pd.DataFrame | None, prices: pd.DataFrame,
               duid: str, region: str) -> pd.DataFrame:
    """5-minute frame for one battery, indexed by interval start (market time).

    Columns: mw (reading at start), discharge_mwh, charge_mwh, net_mwh, rrp, energy_aud, valid (both SCADA readings
    and the price exist), reported_mwh (stored energy at start as reported to AEMO), availability, fcas_aud and
    enable_<service> (MW).
    """
    pr = prices[prices["REGIONID"] == region].drop_duplicates("SETTLEMENTDATE")
    pr = pr.set_index(pr["SETTLEMENTDATE"] - STEP).sort_index()
    idx = pd.DatetimeIndex(pr.index, name="interval_start")
    sc = scada[scada["DUID"] == duid]
    inst = pd.Series(sc["SCADAVALUE"].to_numpy(float), index=sc["SETTLEMENTDATE"] - STEP).sort_index()
    inst = inst[~inst.index.duplicated()]
    a = inst.reindex(idx).to_numpy()
    b = inst.reindex(idx + STEP).to_numpy()
    b = np.where(np.isnan(b), a, b)            # last reading of the series: hold the value
    dis, chg = split_energy(np.nan_to_num(a), np.nan_to_num(b))
    out = pd.DataFrame({"mw": a, "discharge_mwh": dis, "charge_mwh": chg}, index=idx)
    out["net_mwh"] = out["discharge_mwh"] - out["charge_mwh"]
    out["rrp"] = pr["RRP"].to_numpy(float)
    out["valid"] = ~np.isnan(a) & ~np.isnan(out["rrp"].to_numpy())
    out.loc[~out["valid"], ["discharge_mwh", "charge_mwh", "net_mwh"]] = 0.0
    out["energy_aud"] = out["rrp"].fillna(0) * out["net_mwh"]
    out["reported_mwh"] = np.nan
    out["availability"] = np.nan
    out["fcas_aud"] = 0.0
    if dispatch is not None:
        d = dispatch[dispatch["DUID"] == duid].sort_values("INTERVENTION").drop_duplicates("SETTLEMENTDATE", keep="last")
        d = d.set_index(d["SETTLEMENTDATE"] - STEP).reindex(idx)
        if "INITIAL_ENERGY_STORAGE" in d:
            out["reported_mwh"] = d["INITIAL_ENERGY_STORAGE"].to_numpy(float)
        if "AVAILABILITY" in d:
            out["availability"] = d["AVAILABILITY"].to_numpy(float)
        for s in FCAS_SERVICES:
            if s not in d or f"{s}RRP" not in pr:
                continue
            enable = d[s].fillna(0).to_numpy(float)
            out[f"enable_{s}"] = enable
            out["fcas_aud"] += enable * pr[f"{s}RRP"].fillna(0).to_numpy(float) * DT_5MIN
    return out
