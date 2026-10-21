"""The analysis steps behind the command line: SoC validation, fleet survey of reported SoC, cell data and ageing
model, and the strategy comparison. Each step writes CSV/JSON to ``results/`` and returns what the charts need.

Usage:
    from bess_degradation import pipeline
    val = pipeline.soc_validation(Path("data/nem"), Path("data/acen"), Path("results"))
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bess_degradation import acen, aging, cellfit, nem, soc, strategy
from bess_degradation.cells import read_checkup, read_cycle_analysis

log = logging.getLogger(__name__)


def _load_units(nem_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_parquet(nem_dir / "registry.parquet")
    stations = pd.read_parquet(nem_dir / "stations.parquet") if (nem_dir / "stations.parquet").exists() else None
    return registry, nem.battery_table(registry, stations)


# ---------------------------------------------------------------------------------------------------------------------
# 1. SoC reconstruction validated on the New England BESS
# ---------------------------------------------------------------------------------------------------------------------

ACEN_UNITS = {1: "NESBESS1", 2: "NESBESS2"}


def soc_validation(nem_dir: Path, acen_dir: Path, out: Path, train: tuple[str, str] = ("2026-06-01", "2026-06-30"),
                   test_end: dict | None = None, reset_jump: float = 0.05) -> dict:
    """Calibrate the SoC model on one month of ACEN's published SoC and validate it on the following months.

    Inputs compared: AEMO SCADA (public market data, instantaneous MW at each interval start) and ACEN's own
    interval-average power. Models compared: registered parameters (AEMO MWh and efficiencies) and calibrated ones.
    Metrics: error with a daily anchor (reconstruction restarted from the published SoC at midnight), its growth with
    hours since the anchor, and the drift of a never-anchored integral.
    """
    test_end = test_end or {1: "2026-09-30", 2: "2026-08-31"}       # section 2 idles from September
    start, end = date(2026, 5, 1), date(2026, 9, 30)
    duids = list(ACEN_UNITS.values())
    scada = nem.load("scada", nem_dir, start, end, duids=duids)
    prices = nem.load("price", nem_dir, start, end)
    dispatch = nem.load("dispatch", nem_dir, start, end, duids=duids)
    _, units = _load_units(nem_dir)
    ac = acen.load(acen_dir)

    rows, monthly, by_hour, examples, jumps = [], [], [], {}, []
    for k, duid in ACEN_UNITS.items():
        u = units.loc[duid]
        iv = nem.unit_frame(scada, dispatch, prices, duid, "NSW1")
        obs = ac[f"soc_{k}"].reindex(iv.index)
        p = ac[f"mw_{k}"].reindex(iv.index)
        inputs = {
            "AEMO SCADA": (iv["charge_mwh"].where(iv["valid"]), iv["discharge_mwh"].where(iv["valid"])),
            "ACEN power": ((-p).clip(lower=0) * soc.DT_5MIN, p.clip(lower=0) * soc.DT_5MIN),
        }
        test = slice(str((pd.Timestamp(train[1]) + pd.Timedelta(days=1)).date()), test_end[k])
        o = obs.loc[test]
        for src, (c, d) in inputs.items():
            cal = soc.calibrate(obs.loc[train[0]:train[1]], c.loc[train[0]:train[1]], d.loc[train[0]:train[1]])
            models = {
                "registered": soc.SocModel(u["energy_mwh"], u["eta_charge"], u["eta_discharge"]),
                "calibrated": cal.model,
            }
            for name, m in models.items():
                cc, dd = c.loc[test].fillna(0), d.loc[test].fillna(0)
                ra = soc.reanchored_errors(o, cc, dd, m)
                open_loop = soc.reconstruct(cc, dd, m, float(o.iloc[0]))
                ol = soc.error_metrics(o, open_loop)
                eod = ra[ra["hours"] >= 23.9]["error_pp"].abs()
                rows.append({
                    "duid": duid, "input": src, "model": name, "energy_mwh": m.energy_mwh,
                    "rte": m.round_trip_efficiency, "standby_mw": m.standby_mw,
                    "test_from": test.start, "test_to": test.stop,
                    "daily_anchor_rmse_pp": float(np.sqrt(np.nanmean(ra["error_pp"] ** 2))),
                    "daily_anchor_mae_pp": float(np.nanmean(ra["error_pp"].abs())),
                    "end_of_day_p95_pp": float(eod.quantile(0.95)),
                    "open_loop_rmse_pp": ol["rmse_pp"], "open_loop_drift_pp_per_day": ol["drift_pp_per_day"],
                    "calib_r2": cal.r2 if name == "calibrated" else np.nan,
                    "calib_windows": cal.n_windows if name == "calibrated" else np.nan,
                })
                if src == "AEMO SCADA":
                    h = ra.groupby(ra["hours"].astype(int))["error_pp"].apply(lambda e: np.sqrt(np.nanmean(e ** 2)))
                    by_hour.append(pd.DataFrame({"duid": duid, "model": name, "hour": h.index, "rmse_pp": h.values}))
                    if name == "calibrated":
                        examples[duid] = pd.DataFrame({"observed": o, "reconstructed": ra["error_pp"] / 100 + o})
            for mo in pd.period_range(train[0], test_end[k], freq="M"):
                s = str(mo)
                try:
                    cm = soc.calibrate(obs.loc[s], c.loc[s], d.loc[s])
                except ValueError:
                    continue
                if cm.a <= 0 or cm.b <= 0:
                    continue
                monthly.append({"duid": duid, "input": src, "month": s, "rte": cm.round_trip_efficiency,
                                "energy_mwh": cm.energy_mwh, "standby_mw": cm.model.standby_mw, "r2": cm.r2,
                                "windows": cm.n_windows})
        # BMS resets: 5-minute SoC steps far from what the metered energy explains
        cal = soc.calibrate(obs.loc[train[0]:train[1]], *(x.loc[train[0]:train[1]] for x in inputs["AEMO SCADA"]))
        pred = cal.model.delta(inputs["AEMO SCADA"][0].fillna(0), inputs["AEMO SCADA"][1].fillna(0))
        resid = (obs.shift(-1) - obs - pred).loc[train[0]:test_end[k]]
        big = resid[resid.abs() > reset_jump]
        jumps.append({"duid": duid, "resets": int(len(big)), "largest_pp": float(big.abs().max() * 100) if len(big) else 0.0})

    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "soc_validation.csv", index=False, float_format="%.4g")
    month_table = pd.DataFrame(monthly)
    month_table.to_csv(out / "soc_monthly_calibration.csv", index=False, float_format="%.4g")
    hours = pd.concat(by_hour, ignore_index=True)
    hours.to_csv(out / "soc_error_by_hour.csv", index=False, float_format="%.4g")
    pd.DataFrame(jumps).to_csv(out / "soc_bms_resets.csv", index=False)
    reported = _reported_vs_acen(dispatch, ac)
    reported.to_csv(out / "acen_vs_aemo_reported.csv", index=False, float_format="%.4g")
    return {"table": table, "monthly": month_table, "by_hour": hours, "examples": examples, "resets": jumps,
            "reported": reported}


def _reported_vs_acen(dispatch: pd.DataFrame, ac: pd.DataFrame) -> pd.DataFrame:
    """Per month: correlation between the stored energy reported to AEMO and ACEN's published SoC."""
    rows = []
    for k, duid in ACEN_UNITS.items():
        d = dispatch[(dispatch["DUID"] == duid) & (dispatch["INTERVENTION"] == 0)]
        e = pd.Series(d["INITIAL_ENERGY_STORAGE"].to_numpy(float), index=d["SETTLEMENTDATE"] - nem.STEP).sort_index()
        j = pd.DataFrame({"soc": ac[f"soc_{k}"], "e": e}).dropna()
        j = j[j["e"] > 0]
        for mo, g in j.groupby(j.index.to_period("M")):
            if len(g) < 500:
                continue
            slope = np.polyfit(g["soc"], g["e"], 1)
            rows.append({"duid": duid, "month": str(mo), "n": len(g), "corr": float(g["soc"].corr(g["e"])),
                         "mwh_per_soc": float(slope[0]), "intercept_mwh": float(slope[1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Stored energy reported to AEMO across the fleet: efficiency, standby drain and apparent capacity per month
# ---------------------------------------------------------------------------------------------------------------------

def fleet_reported_soc(nem_dir: Path, out: Path, start: date, end: date, min_mw: float = 100.0) -> pd.DataFrame:
    """Calibrate the SoC model monthly against INITIAL_ENERGY_STORAGE for every battery of ``min_mw`` or more.

    ``scale`` = fitted usable energy / registered MWh. If the operator reports SoC x a fixed nameplate, capacity fade
    shows up as a falling scale; a constant scale means either no fade or a BMS that reports energy relative to its
    current capacity (both are common), which is why this is only an indirect SoH indicator.
    """
    registry, units = _load_units(nem_dir)
    big = units[units["discharge_mw"] >= min_mw].index.tolist()
    scada = nem.load("scada", nem_dir, start, end, duids=big)
    prices = nem.load("price", nem_dir, start, end)
    dispatch = nem.load("dispatch", nem_dir, start, end, duids=big,
                        columns=["SETTLEMENTDATE", "DUID", "INTERVENTION", "INITIAL_ENERGY_STORAGE"])
    rows = []
    for duid in big:
        u = units.loc[duid]
        iv = nem.unit_frame(scada, dispatch, prices, duid, u["region"])
        e = iv["reported_mwh"].where(iv["reported_mwh"] > 0)
        c, d = iv["charge_mwh"].where(iv["valid"]), iv["discharge_mwh"].where(iv["valid"])
        for mo in iv.index.to_period("M").unique():
            s = str(mo)
            days = d.loc[s].index.normalize().nunique()
            if d.loc[s].sum() < 0.5 * u["energy_mwh"]:
                continue
            try:
                cal = soc.calibrate(e.loc[s] / u["energy_mwh"], c.loc[s], d.loc[s])
            except ValueError:
                continue
            ok = cal.a > 0 and cal.b > 0
            rows.append({"duid": duid, "name": u["name"], "region": u["region"], "mw": u["discharge_mw"],
                         "mwh": u["energy_mwh"], "month": s, "r2": cal.r2, "windows": cal.n_windows,
                         "rte": cal.round_trip_efficiency if ok else np.nan,
                         "scale": cal.energy_mwh / u["energy_mwh"] if ok else np.nan,
                         "standby_mw": cal.model.standby_mw if ok else np.nan,
                         "cycles_per_day": d.loc[s].sum() / u["energy_mwh"] / days})
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "fleet_reported_soc_monthly.csv", index=False, float_format="%.5g")
    return table


def fleet_summary(monthly: pd.DataFrame, min_r2: float = 0.99, min_months: int = 8) -> pd.DataFrame:
    """Per battery: months with a consistent reported SoC, median RTE and standby drain, apparent-capacity trend."""
    good = monthly[(monthly["r2"] >= min_r2) & (monthly["windows"] > 200)].copy()
    good["t"] = pd.PeriodIndex(good["month"], freq="M").astype("int64") / 12
    rows = []
    for (duid, name), g in good.groupby(["duid", "name"]):
        n_all = int((monthly["duid"] == duid).sum())
        row = {"duid": duid, "name": name, "region": g["region"].iloc[0], "mw": g["mw"].iloc[0],
               "mwh": g["mwh"].iloc[0], "months": n_all, "consistent_months": len(g),
               "rte": g["rte"].median(), "standby_mw": g["standby_mw"].median(),
               "standby_pct_of_mw": 100 * g["standby_mw"].median() / g["mw"].iloc[0],
               "scale": g["scale"].median(), "cycles_per_day": g["cycles_per_day"].mean(),
               "scale_trend_pct_per_yr": np.nan}
        if len(g) >= min_months:
            row["scale_trend_pct_per_yr"] = 100 * np.polyfit(g["t"], g["scale"], 1)[0] / g["scale"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mw", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Cell data: capacity, ICA, a small capacity regression, and the ageing-model fit
# ---------------------------------------------------------------------------------------------------------------------

def cell_analysis(cells_dir: Path, out: Path, beta: float = 1.0) -> dict:
    caps = cellfit.checkup_table(cells_dir)
    out.mkdir(parents=True, exist_ok=True)
    caps.to_csv(out / "cell_checkup_capacity.csv", index=False, float_format="%.5g")
    # calendar: square-root law (z_cal = 0.5); a free exponent fits no better (see aging_fit.json)
    cal = cellfit.fit_calendar(caps, z_fixed=0.5)
    cal_free = cellfit.fit_calendar(caps)
    cal_model = aging.AgingModel(k_cal=cal.params["k_cal"], ea_cal=cal.params["ea_cal"], gamma=cal.params["gamma"],
                                 z_cal=cal.params["z_cal"])
    per_cycle = read_cycle_analysis(cells_dir)          # the authors' values: used, not re-published
    cell_temp = per_cycle.groupby("cell")["temp_c"].mean().to_dict()     # 36 / 51.5 degC in 35 / 50 degC chambers
    cyc_pow = cellfit.fit_cycle(caps, cal_model, cell_temp=cell_temp)
    cyc = cellfit.fit_cycle(caps, cal_model, cell_temp=cell_temp, z_fixed=1.0)
    # headline: cycle ageing linear in equivalent cycles. The power law (z ~ 0.5) fits the 1500 tested cycles a
    # little better but cannot be extrapolated to the 5,000-20,000 cycles of a grid battery's life: it would make
    # every extra cycle nearly free. It is kept as the optimistic bound of the SoH fan.
    model = cellfit.model_from_fits(cal, cyc, beta=beta)
    model_pow = cellfit.model_from_fits(cal, cyc_pow, beta=beta)

    # incremental capacity of the C/10 charge at every checkup that has one, and of the 50 A charge at all checkups
    curves, feats = {}, []
    for f in sorted((cells_dir / "checkups").glob("*/*.parquet")):
        cell, checkup = f.parent.name, int(f.name[:4])
        df, _ = read_checkup(f)
        for current in (18.0, 50.0):
            try:
                cv = cellfit.ica_curve(df, current)
            except ValueError:
                continue
            curves[(cell, checkup, current)] = cv
            feats.append({"cell": cell, "checkup": checkup, "current_a": current, **cellfit.ica_features(cv)})
    feats = pd.DataFrame(feats).drop_duplicates(["cell", "checkup", "current_a"], keep="last")
    feats.to_csv(out / "cell_ica_features.csv", index=False, float_format="%.5g")
    ml = capacity_regression(caps, feats)

    fit = {"calendar": {**cal.params, "stderr": cal.stderr, "rmse_pct": cal.rmse_pct},
           "calendar_free_z": {**cal_free.params, "stderr": cal_free.stderr, "rmse_pct": cal_free.rmse_pct},
           "cycle": {**cyc.params, "stderr": cyc.stderr, "rmse_pct": cyc.rmse_pct},
           "cycle_power_law": {**cyc_pow.params, "stderr": cyc_pow.stderr, "rmse_pct": cyc_pow.rmse_pct},
           "cell_temperature_c": cell_temp,
           "model": asdict(model), "model_power_law": asdict(model_pow),
           "ica_regression": {k: v for k, v in ml.items() if k != "predictions"}}
    (out / "aging_fit.json").write_text(json.dumps(fit, indent=2, default=float))
    pd.concat([cal.residuals.assign(fit="calendar"), cyc.residuals.assign(fit="cycle"),
               cyc_pow.residuals.assign(fit="cycle_power_law")]).to_csv(
        out / "aging_fit_residuals.csv", index=False, float_format="%.5g")
    ml["predictions"].to_csv(out / "cell_ica_regression.csv", index=False, float_format="%.5g")
    return {"caps": caps, "calendar": cal, "cycle": cyc, "cycle_power_law": cyc_pow, "model": model,
            "model_power_law": model_pow, "curves": curves, "features": feats, "ml": ml, "per_cycle": per_cycle,
            "cell_temp": cell_temp}


def capacity_regression(caps: pd.DataFrame, feats: pd.DataFrame, current_a: float = 50.0) -> dict:
    """Ridge regression from dQ/dV features of the 50 A charge (charge in 3.30-3.36 and 3.36-3.42 V, the flat middle
    of an LFP curve that a grid battery passes through daily, plus the main peak) to the cell's capacity, validated
    leave-one-cell-out. The formation checkup (other cycler, other temperature) is excluded."""
    cap = caps[caps["current_a"] == current_a][["cell", "checkup", "q_dis_ah"]]
    d = feats[(feats["current_a"] == current_a) & (feats["checkup"] > 0)].merge(cap, on=["cell", "checkup"])
    cols = ["ah_3.30_3.36", "ah_3.36_3.42", "peak_ah_per_v", "peak_v"]
    x = d[cols].to_numpy(float)
    y = d["q_dis_ah"].to_numpy(float)
    preds = np.full(len(d), np.nan)
    lam = 1e-3
    for cell in d["cell"].unique():
        tr, te = (d["cell"] != cell).to_numpy(), (d["cell"] == cell).to_numpy()
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-12
        xt = (x[tr] - mu) / sd
        a = np.c_[np.ones(tr.sum()), xt]
        coef = np.linalg.solve(a.T @ a + lam * np.diag([0] + [1] * len(cols)), a.T @ y[tr])
        preds[te] = np.c_[np.ones(te.sum()), (x[te] - mu) / sd] @ coef
    base = np.full(len(d), np.nan)
    for cell in d["cell"].unique():                     # baseline: mean capacity of the other cells at that checkup
        te = (d["cell"] == cell).to_numpy()
        for i in np.flatnonzero(te):
            other = (d["checkup"] == d["checkup"].iloc[i]) & (d["cell"] != cell)
            base[i] = y[other.to_numpy()].mean()
    err = (preds - y) / y * 100
    berr = (base - y) / y * 100
    return {"features": cols, "n": int(len(d)), "mae_pct": float(np.nanmean(np.abs(err))),
            "rmse_pct": float(np.sqrt(np.nanmean(err ** 2))), "baseline_mae_pct": float(np.nanmean(np.abs(berr))),
            "predictions": d[["cell", "checkup", "q_dis_ah"]].assign(predicted_ah=preds, baseline_ah=base)}


# ---------------------------------------------------------------------------------------------------------------------
# 4. Strategies: actual operation vs perfect-foresight arbitrage with and without a cost of ageing
# ---------------------------------------------------------------------------------------------------------------------

STRATEGY_UNITS = ["TIB1", "BLYTHB1", "TARBESS1", "MREHA1"]    # stored energy reported consistently all year


def actual_soc(iv: pd.DataFrame, energy_mwh: float, max_gap: int = 12) -> pd.Series:
    """SoC (fraction of registered MWh) from the stored energy reported to AEMO; gaps up to ``max_gap`` intervals
    are interpolated, longer ones are left empty."""
    e = iv["reported_mwh"].where(iv["reported_mwh"] >= 0)
    e = e.interpolate(limit=max_gap, limit_area="inside")
    return (e / energy_mwh).clip(0.0, 1.0).rename("soc")


def run_year(prices: pd.Series, battery: strategy.Battery, cost: strategy.AgingCost, max_cycles: float | None = None,
             soc_start: float = 0.5) -> tuple[pd.DataFrame, pd.Series]:
    """Optimise day by day, each day starting where the previous one ended (at most ``max_cycles`` full cycles of
    discharge per day). Returns daily revenue/throughput and the SoC at the start of every interval (fraction)."""
    days, socs = [], []
    e = soc_start * battery.energy_mwh
    cap = max_cycles * battery.energy_mwh if max_cycles else None
    for day, p in prices.groupby(prices.index.normalize()):
        if len(p) != 288 or p.isna().any():
            continue
        res = strategy.optimize_day(p, battery, cost, e_start=e, max_discharge_mwh=cap)
        s = res.schedule
        start_e = np.r_[e, s["energy_mwh"].to_numpy()[:-1]]
        socs.append(pd.Series(start_e / battery.energy_mwh, index=p.index))
        e = float(s["energy_mwh"].iloc[-1])
        days.append({"day": day, "revenue_aud": res.revenue, "objective_aud": res.objective,
                     "discharged_mwh": res.discharged_mwh, "charged_mwh": res.charged_mwh, "binaries": res.binaries})
    return pd.DataFrame(days).set_index("day"), pd.concat(socs).clip(0.0, 1.0)


def _strategy_job(duid: str, name: str, prices: pd.Series, battery: strategy.Battery, cost: strategy.AgingCost,
                  max_cycles: float | None):
    logging.basicConfig(level=logging.WARNING)
    daily, soc_profile = run_year(prices, battery, cost, max_cycles)
    return duid, name, daily, soc_profile


def aging_cost(model: aging.AgingModel, battery: strategy.Battery, replacement_aud_per_kwh: float,
               eol_fade: float = 0.2, temp_c: float = 25.0, segments: int = 8) -> strategy.AgingCost:
    """Objective terms from the ageing model: the battery's replacement value is spread over the capacity it can
    lose before end of life (``eol_fade``). With beta = 1 all segments cost the same, so one segment is used."""
    fc = replacement_aud_per_kwh * 1000 * battery.energy_mwh / eol_fade
    seg = aging.segment_costs(model, battery.energy_mwh, fc, segments, temp_c)
    if np.allclose(seg, seg[0]):
        seg = seg[:1]
    return strategy.AgingCost(throughput=1.0, segment=seg,
                              holding=aging.holding_cost(model, battery.energy_mwh, fc, temp_c))


def evaluate(soc_profile: pd.Series, model: aging.AgingModel, temp_c: float = 25.0) -> dict:
    """Ageing of a SoC profile: rainflow, losses after 1 and 10 years, years to 80 % SoH."""
    st = aging.profile_stress(soc_profile.to_numpy(), dt_hours=soc.DT_5MIN)
    y1 = aging.loss_first_years(model, st, temp_c, 1.0)
    path = aging.soh_path(model, st, years=20, temp_c=temp_c)
    loss10 = 1 - float(path.loc[np.isclose(path["years"], 10), "soh"].iloc[0])
    return {"stress": st, "soc": soc_profile, "fec_per_year": st.fec_per_year, "mean_soc": st.mean_soc,
            "loss_y1_pct": 100 * y1["total"], "loss_y1_cycle_pct": 100 * y1["cycle"],
            "loss_y1_calendar_pct": 100 * y1["calendar"], "loss_10y_pct": 100 * loss10,
            "years_to_80": aging.years_to(path, 0.8), "path": path}


def _row(u: pd.Series, name: str, energy_rev: float, fcas_rev: float, dis: float, years: float, ev: dict,
         rte: float, fade_cost_aud: float) -> dict:
    rev_year = (energy_rev + fcas_rev) / years
    return {
        "duid": u.name, "name": u["name"], "region": u["region"], "mw": u["discharge_mw"], "mwh": u["energy_mwh"],
        "rte": rte, "strategy": name, "days": round(years * 365),
        "energy_revenue_k_aud_yr": energy_rev / years / 1e3, "fcas_revenue_k_aud_yr": fcas_rev / years / 1e3,
        "revenue_k_aud_per_mw_yr": rev_year / u["discharge_mw"] / 1e3,
        "discharged_mwh_yr": dis / years, "cycles_per_day": dis / years / 365 / u["energy_mwh"],
        **{k: v for k, v in ev.items() if k not in ("stress", "path", "soc")},
        "revenue_k_aud_per_pct_loss_y1": rev_year / 1e3 / ev["loss_y1_pct"],
        "revenue_k_aud_per_pct_loss_10y": 10 * rev_year / 1e3 / ev["loss_10y_pct"],
        # ageing cost of 10 years of this operation spread over its throughput
        "degradation_cost_aud_per_mwh": fade_cost_aud * ev["loss_10y_pct"] / 100 / max(10 * dis / years, 1e-9),
    }


def strategies(nem_dir: Path, out: Path, model: aging.AgingModel, start: date, end: date,
               units_sel: list[str] | None = None, replacement_aud_per_kwh: tuple[float, ...] = (125, 250, 500),
               headline_aud_per_kwh: float = 250.0, eol_fade: float = 0.2, temp_c: float = 25.0,
               segments: int = 8, jobs: int = 4) -> dict:
    """For each battery: actual operation (SoC reported to AEMO) and perfect-foresight strategies on the same prices:

    * ``pf_max``        - no ageing cost (1 AUD/MWh), at most 2 cycles a day (a typical warranty limit),
    * ``pf_rule``       - 10 AUD/MWh and at most 1.5 cycles a day (a common rule of thumb; the companion project's
                          benchmark),
    * ``pf_aware_<R>``  - the fitted ageing model in the objective (cycle-depth segments plus a SoC holding cost)
                          for replacement costs R in AUD/kWh, no cycle limit,
    * ``pf_aware_<R>_deep`` - the headline R with beta = 1.5 (deep cycles costlier per MWh than shallow ones),
                          optimised and evaluated with that model.

    Every strategy's SoC profile goes through the same rainflow count and ageing model.
    """
    registry, units = _load_units(nem_dir)
    units_sel = list(units_sel or STRATEGY_UNITS)
    scada = nem.load("scada", nem_dir, start, end, duids=units_sel)
    prices = nem.load("price", nem_dir, start, end)
    dispatch = nem.load("dispatch", nem_dir, start, end, duids=units_sel)

    frames, batteries = {}, {}
    for duid in units_sel:
        u = units.loc[duid]
        iv = nem.unit_frame(scada, dispatch, prices, duid, u["region"])
        iv = iv.loc[pd.Timestamp(start):pd.Timestamp(end) + pd.Timedelta("23h55min")]
        frames[duid] = iv
        obs = iv["reported_mwh"].where(iv["reported_mwh"] > 0) / u["energy_mwh"]
        try:
            cal = soc.calibrate(obs, iv["charge_mwh"].where(iv["valid"]), iv["discharge_mwh"].where(iv["valid"]))
            rte = float(np.clip(cal.round_trip_efficiency, 0.75, 0.97))
        except ValueError:
            rte = 0.88
        batteries[duid] = strategy.Battery(u["discharge_mw"], u["charge_mw"], u["energy_mwh"], rte)

    deep = model.with_(beta=1.5)
    deep_name = f"pf_aware_{int(headline_aud_per_kwh)}_deep"
    eval_model = {deep_name: deep}
    jobs_list = []
    for duid in units_sel:
        b = batteries[duid]
        p = frames[duid]["rrp"]
        jobs_list.append((duid, "pf_max", p, b, strategy.AgingCost(throughput=1.0), 2.0))
        jobs_list.append((duid, "pf_rule", p, b, strategy.AgingCost(throughput=10.0), 1.5))
        for r in replacement_aud_per_kwh:
            jobs_list.append((duid, f"pf_aware_{int(r)}", p, b, aging_cost(model, b, r, eol_fade, temp_c, segments),
                              None))
        jobs_list.append((duid, deep_name, p, b, aging_cost(deep, b, headline_aud_per_kwh, eol_fade, temp_c, segments),
                          None))

    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_strategy_job, *j) for j in jobs_list]
        for f in futures:
            duid, name, daily, prof = f.result()
            results[(duid, name)] = (daily, prof)
            log.info("%-9s %-14s revenue %.1f M AUD, %.0f MWh discharged", duid, name,
                     daily["revenue_aud"].sum() / 1e6, daily["discharged_mwh"].sum())

    rows, evals = [], {}
    for duid in units_sel:
        u, iv, b = units.loc[duid], frames[duid], batteries[duid]
        years = iv.index.normalize().nunique() / 365
        fade_cost = headline_aud_per_kwh * 1000 * u["energy_mwh"] / eol_fade
        cand = {"actual": (actual_soc(iv, u["energy_mwh"]).dropna(), iv["energy_aud"].sum(), iv["fcas_aud"].sum(),
                           iv["discharge_mwh"].sum())}
        for (d2, name), (daily, prof) in results.items():
            if d2 == duid:
                cand[name] = (prof, daily["revenue_aud"].sum(), 0.0, daily["discharged_mwh"].sum())
        for name, (prof, energy_rev, fcas_rev, dis) in cand.items():
            ev = evaluate(prof, eval_model.get(name, model), temp_c)
            evals[(duid, name)] = ev
            rows.append(_row(u, name, energy_rev, fcas_rev, dis, years, ev, b.round_trip_efficiency, fade_cost))
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "strategies.csv", index=False, float_format="%.5g")
    daily = pd.concat({k: v[0] for k, v in results.items()}, names=["duid", "strategy"])
    daily.to_csv(out / "strategies_daily.csv", float_format="%.6g")
    return {"table": table, "evals": evals, "daily": daily, "batteries": batteries, "frames": frames}


def soh_fan(evals: dict, model: aging.AgingModel, power_law: aging.AgingModel, years: float = 15.0,
            temps=(20.0, 25.0, 30.0), betas=(1.0, 1.5)) -> pd.DataFrame:
    """SoH over time for every (battery, strategy): the central model (25 degC, linear cycle ageing) and the
    envelope over cell temperature, depth exponent beta and the cycle-ageing law (linear or power law)."""
    rows = []
    for (duid, name), ev in evals.items():
        central = aging.soh_path(model, ev["stress"], years, 25.0, steps_per_year=4)
        variants = [aging.soh_path(m.with_(beta=b), ev["stress"], years, t, steps_per_year=4)["soh"].to_numpy()
                    for m in (model, power_law) for b in betas for t in temps]
        v = np.vstack(variants)
        rows.append(pd.DataFrame({"duid": duid, "strategy": name, "years": central["years"],
                                  "soh": central["soh"], "soh_low": v.min(axis=0), "soh_high": v.max(axis=0)}))
    return pd.concat(rows, ignore_index=True)
