"""Cell-level ageing data: large-format stationary-storage LFP cells (Yagci, Richter, Behmann, Bessler 2025).

Twelve commercial 180 Ah LFP/graphite cells for stationary storage were aged for 807-875 days: four by full
cycling (35 and 50 degC) and eight by storage (calendar ageing) at 75 % and 100 % state of charge (35 and 50 degC).
Reference performance tests ("checkups") were run before ageing and after the equivalent of 100, 500, 1000 and 1500
cycles. Data: Zenodo, https://doi.org/10.5281/zenodo.22939281, CC BY 4.0.

The archive is 2.5 GB; :func:`fetch` extracts only what this project uses with HTTP range requests (about
0.46 GB): the summary spreadsheets (including the authors' per-cycle analysis of the cycled cells) and the raw
capacity tests of every checkup. The raw cycling records are MATLAB table objects and are not used.

Usage:
    from bess_degradation.cells import fetch
    fetch(Path("data/cells"))
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bess_degradation.remotezip import RemoteZip

log = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/22939281/files/Data.zip?download=1"
CYCLE_SHEET = "Data/Analysis individual cycles - cyclic aging cells.xlsx"

# lab label -> (paper number, ageing mode, temperature degC, storage SOC %); from "Cell labels paper vs. raw data.xlsx"
CELLS = {
    "CA-181103-19": (1, "cycling", 35, None), "CA-181103-42": (2, "cycling", 35, None),
    "CA-181103-26": (3, "calendar", 35, 100), "CA-181103-30": (4, "calendar", 35, 100),
    "CA-181103-20": (5, "calendar", 35, 75), "CA-181103-38": (6, "calendar", 35, 75),
    "CA-181103-22": (7, "cycling", 50, None), "CA-181103-28": (8, "cycling", 50, None),
    "CA-181103-31": (9, "calendar", 50, 100), "CA-181103-35": (10, "calendar", 50, 100),
    "CA-181103-24": (11, "calendar", 50, 75), "CA-181103-34": (12, "calendar", 50, 75),
}


def wanted(name: str) -> bool:
    """Members of the archive this project needs."""
    if name.endswith("/"):
        return False
    if name.startswith("Data/") and name.count("/") == 1:
        return name.endswith(".xlsx")
    return "/Checkups/" in name and "/Capacity/" in name


def fetch(data_dir: Path) -> None:
    """Extract the needed members of the Zenodo archive into ``data_dir``.

    Text files of the checkups are converted to Parquet (``data_dir/checkups/<cell>/<checkup>_<test>.parquet``) right
    after extraction and the text deleted; members already converted are skipped.
    """
    rz = RemoteZip(ZENODO_URL)
    names = [m.name for m in rz.members if wanted(m.name)]
    total = sum(m.compressed_size for m in rz.members if m.name in names)
    log.info("extracting %d of %d members (%.0f MB compressed)", len(names), len(rz.members), total / 1e6)
    for name in names:
        if name.endswith(".txt"):
            out = checkup_parquet_path(data_dir, name)
            if out.exists():
                continue
            txt = rz.extract(name, data_dir / "raw")
            convert_checkup(txt, out)
            txt.unlink()
        else:
            rz.extract(name, data_dir / "raw")


# ---------------------------------------------------------------------------------------------------------------------
# Checkup files. Two cyclers were used: the initial checkup (formation) was exported from a BioLogic potentiostat
# (EC-Lab text export, mA / mA.h / seconds, no header block), all later checkups from a Basytec cycler (header block
# starting with "~", h / A / Ah). Both are converted to one layout:
#   time_h, voltage, current_a (+ charge, - discharge), ah (charge counter from the cycler), temp_c, step, command
# ---------------------------------------------------------------------------------------------------------------------

CHECKUP_RE = re.compile(r"Checkups/Checkup_(\d+)/Capacity/(CA-181103-\d+)_(.+)\.txt$")


def checkup_parquet_path(data_dir: Path, member: str) -> Path:
    m = CHECKUP_RE.search(member)
    if not m:
        raise ValueError(f"not a checkup capacity file: {member}")
    checkup, cell, test = m.groups()
    return data_dir / "checkups" / cell / f"{int(checkup):04d}_{re.sub(r'[^A-Za-z0-9_#.-]+', '_', test)}.parquet"


def read_basytec(path: Path) -> tuple[pd.DataFrame, dict]:
    """Basytec result file: header block of ``~key: value`` lines, the last of which holds the column names."""
    meta, header = {}, None
    with open(path, encoding="latin-1") as f:
        for n, line in enumerate(f):
            if not line.startswith("~"):
                break
            body = line[1:].rstrip("\n")
            if "\t" in body:
                header = body.split("\t")
            elif ":" in body:
                k, v = body.split(":", 1)
                meta[k.strip()] = v.strip()
    raw = pd.read_csv(path, sep="\t", skiprows=n, header=None, names=header, encoding="latin-1", low_memory=False)
    col = {c.split("[")[0]: c for c in header}
    df = pd.DataFrame({
        "time_h": pd.to_numeric(raw[col["Time"]], errors="coerce"),
        "voltage": pd.to_numeric(raw[col["U"]], errors="coerce").astype("float32"),
        "current_a": pd.to_numeric(raw[col["I"]], errors="coerce").astype("float32"),
        "ah": pd.to_numeric(raw[col["Ah"]], errors="coerce"),
        "temp_c": pd.to_numeric(raw[col["T1"]], errors="coerce").astype("float32") if "T1" in col else np.float32(np.nan),
        "step": pd.to_numeric(raw[col["Line"]], errors="coerce").astype("Int32"),
        "command": raw[col["Command"]].astype("category"),
    })
    start = meta.get("Start of Test")
    meta = {"start": str(pd.to_datetime(start, format="%d.%m.%Y %H:%M:%S")) if start else "",
            "test": meta.get("Name of Test", ""), "cycler": "basytec"}
    return df, meta


def read_biologic(path: Path) -> tuple[pd.DataFrame, dict]:
    """EC-Lab text export (one header line). Start date taken from the file name (``..._YYYYMMDD_...``)."""
    raw = pd.read_csv(path, sep="\t", encoding="latin-1", low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    df = pd.DataFrame({
        "time_h": raw["time/s"].astype(float) / 3600,
        "voltage": raw["Ewe/V"].astype("float32"),
        "current_a": (raw["I/mA"] / 1000).astype("float32"),
        "ah": raw["(Q-Qo)/mA.h"].astype(float) / 1000,
        "temp_c": np.float32(np.nan),
        "step": raw["Ns"].astype("Int32"),
        "command": raw["mode"].map({1: "CC", 2: "CV", 3: "Pause"}).fillna("other").astype("category"),
    })
    m = re.search(r"_(20\d{6})_", path.name)
    meta = {"start": str(pd.to_datetime(m.group(1), format="%Y%m%d")) if m else "", "test": path.stem,
            "cycler": "biologic"}
    return df, meta


def convert_checkup(txt: Path, out: Path) -> Path:
    """Text checkup file -> Parquet with the metadata (start time, test name, cycler) in the schema metadata."""
    with open(txt, encoding="latin-1") as f:
        first = f.readline()
    df, meta = read_basytec(txt) if first.startswith("~") else read_biologic(txt)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"checkup": json.dumps(meta).encode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    return out


def read_checkup(path: Path) -> tuple[pd.DataFrame, dict]:
    """A converted checkup file and its metadata."""
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata.get(b"checkup", b"{}"))
    return table.to_pandas(), meta


def read_cycle_analysis(data_dir: Path) -> pd.DataFrame:
    """The authors' per-cycle analysis of the four cycled cells: capacity of every constant-current discharge (Ah),
    mean cell temperature, coulombic and energy efficiency, internal resistance."""
    sheets = pd.read_excel(data_dir / "raw" / CYCLE_SHEET, sheet_name=None, header=None)
    frames = []
    for cell, raw in sheets.items():
        raw = raw.dropna(axis=1, how="all")
        header = raw.iloc[0].astype(str).str.strip().tolist()
        df = raw.iloc[1:].set_axis(header, axis=1).apply(pd.to_numeric, errors="coerce").dropna(subset=["cycle_no"])
        frames.append(pd.DataFrame({
            "cell": cell.strip(), "cycle": df["cycle_no"].astype(int), "q_dis_ah": df["max_CC_DIS"],
            "q_chg_ah": df["max_CC_CHA"], "temp_c": df["T_mean"], "coulomb_eff": df["Coulomb_Eff"],
            "energy_eff": df["Energy_Eff"], "r_ohm": df["Ri"]}))
    return pd.concat(frames, ignore_index=True)
