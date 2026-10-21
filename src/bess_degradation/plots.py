"""Charts: Plotly figures written as PNG (for the README) and collected into one interactive HTML dashboard.

Usage:
    from bess_degradation import plots
    plots.make_all(soc_res, fleet_monthly, fleet, cell_res, strat, fan, Path("docs/img"), Path("docs/dashboard.html"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bess_degradation.cells import CELLS
from bess_degradation.rainflow import depth_histogram

log = logging.getLogger(__name__)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = "Segoe UI, DejaVu Sans, Arial, sans-serif"

# colour follows the strategy everywhere
STRATEGY_STYLE = {
    "actual": ("Actual operation", SLOTS[0], "circle"),
    "pf_max": ("Perfect foresight, no ageing cost (max 2 cycles/day)", SLOTS[1], "square"),
    "pf_rule": ("Perfect foresight, 10 AUD/MWh rule (max 1.5 cycles/day)", SLOTS[2], "diamond"),
    "pf_aware": ("Perfect foresight, ageing cost in the objective", SLOTS[6], "triangle-up"),
    "pf_aware_deep": ("Ageing cost with deep cycles costlier (beta 1.5)", SLOTS[4], "x"),
}


def _style(name: str) -> tuple[str, str, str]:
    if name.endswith("_deep"):
        return STRATEGY_STYLE["pf_aware_deep"]
    if name.startswith("pf_aware"):
        return STRATEGY_STYLE["pf_aware"]
    return STRATEGY_STYLE[name]


def _layout(fig: go.Figure, title: str, height: int = 420, width: int = 900, legend: bool = True,
            legend_bottom: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK2),
        height=height, width=width, margin=dict(l=70, r=30, t=70, b=60), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
    )
    if legend_bottom:                                   # keeps the legend clear of subplot titles
        fig.update_layout(legend=dict(y=-0.2, yanchor="top"), margin=dict(b=110), height=height + 50)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK2),
                     title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK2),
                     title_font=dict(color=INK2))
    return fig


def _short(name: str) -> str:
    return (name.replace("Battery Energy Storage System", "BESS").replace("Energy Storage System", "ESS")
            .replace("Melbourne Renewable Energy Hub Connection", "MREH").replace("Super Battery", "SB").strip())


# ---------------------------------------------------------------------------------------------------- SoC validation

def soc_week(soc_res: dict, duid: str = "NESBESS1", start: str = "2026-09-08", days: int = 7) -> go.Figure:
    ex = soc_res["examples"][duid]
    sl = ex.loc[start:str((pd.Timestamp(start) + pd.Timedelta(days=days - 1)).date())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sl.index, y=100 * sl["observed"], name="Published by the operator (ACEN)",
                             line=dict(color=SLOTS[0], width=2)))
    fig.add_trace(go.Scatter(x=sl.index, y=100 * sl["reconstructed"],
                             name="Reconstructed from AEMO 5-min SCADA (calibrated, re-anchored daily)",
                             line=dict(color=SLOTS[1], width=2)))
    fig.update_yaxes(title="State of charge (%)", range=[0, 110])
    return _layout(fig, f"New England BESS section 1: state of charge, {days} days of the test period")


def soc_error_by_hour(soc_res: dict) -> go.Figure:
    h = soc_res["by_hour"]
    duids = list(h["duid"].unique())
    fig = make_subplots(rows=1, cols=len(duids), shared_yaxes=True, subplot_titles=duids, horizontal_spacing=0.05)
    for j, duid in enumerate(duids, start=1):
        for k, (model, label) in enumerate((("calibrated", "Calibrated on June"),
                                            ("registered", "Registered MWh and efficiencies"))):
            g = h[(h["duid"] == duid) & (h["model"] == model)]
            fig.add_trace(go.Scatter(x=g["hour"], y=g["rmse_pp"], name=label, legendgroup=model, showlegend=j == 1,
                                     line=dict(color=SLOTS[k], width=2), mode="lines+markers",
                                     marker=dict(size=6)), row=1, col=j)
        fig.update_xaxes(title="Hours since midnight anchor", row=1, col=j)
    fig.update_yaxes(title="RMSE of SoC (percentage points)", row=1, col=1)
    fig = _layout(fig, "SoC reconstruction error grows during the day (test: July-September 2026)",
                  legend_bottom=True)
    fig.update_annotations(font=dict(color=INK2, size=12))
    return fig


def fleet_chart(fleet: pd.DataFrame) -> go.Figure:
    f = fleet[fleet["consistent_months"] >= 8].copy()
    f["label"] = f["name"].map(_short) + " (" + f["region"].str[:-1] + ")"
    dup = f["label"].duplicated(keep=False)                  # several units of one station
    f.loc[dup, "label"] = f.loc[dup, "label"].str[:-1] + ", " + f.loc[dup, "duid"] + ")"
    f = f.sort_values("rte")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Round-trip efficiency (%)", "Standby drain (% of rated MW)",
                                        "Apparent capacity trend (%/year)"))
    hover = "%{y}<br>%{x:.1f}<extra></extra>"
    fig.add_trace(go.Scatter(x=100 * f["rte"], y=f["label"], mode="markers", marker=dict(color=SLOTS[0], size=9),
                             hovertemplate=hover, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=f["standby_pct_of_mw"], y=f["label"], mode="markers",
                             marker=dict(color=SLOTS[0], size=9), hovertemplate=hover, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=f["scale_trend_pct_per_yr"], y=f["label"], mode="markers",
                             marker=dict(color=SLOTS[0], size=9), hovertemplate=hover, showlegend=False),
                  row=1, col=3)
    fig.add_vline(x=0, line=dict(color=AXIS, width=1), row=1, col=3)
    fig = _layout(fig, "What the stored energy reported to AEMO reveals, batteries of 100 MW or more",
                  height=max(420, 26 * len(f) + 140), legend=False)
    fig.update_annotations(font=dict(color=INK2, size=12))
    return fig


# ---------------------------------------------------------------------------------------------------------- cells

def cell_capacity(cell_res: dict) -> go.Figure:
    caps = cell_res["caps"]
    caps = caps[(caps["current_a"] == 50.0) & (caps["checkup"] > 0)].copy()
    c0 = cell_res["calendar"].c0.combine_first(cell_res["cycle"].c0)
    caps["retention"] = 100 * caps["q_dis_ah"] / caps["cell"].map(c0)
    m = cell_res["model"]
    temp_cell = cell_res["cell_temp"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Storage (calendar ageing)", "Full cycles at 50 A (about 1.8 a day)"))
    t = np.linspace(1, 900, 120)
    for col, mode in ((1, "calendar"), (2, "cycling")):
        for k, temp in enumerate((35, 50)):
            for soc, symbol in (((100, "circle"), (75, "diamond")) if mode == "calendar" else ((None, "circle"),)):
                g = caps[(caps["mode"] == mode) & (caps["temp_c"] == temp)]
                if soc is not None:
                    g = g[g["storage_soc"] == soc]
                label = f"{temp} °C" + (f", stored at {soc} %" if soc else "")
                fig.add_trace(go.Scatter(x=g["days"], y=g["retention"], mode="markers", name=label,
                                         marker=dict(color=SLOTS[k], size=9, symbol=symbol,
                                                     line=dict(color=SURFACE, width=1)),
                                         text=g["cell"], hovertemplate="%{text}<br>day %{x:.0f}: %{y:.1f} %<extra></extra>"),
                              row=1, col=col)
            if mode == "calendar":
                y = 100 * (1 - m.calendar_loss(t, 0.875, temp))
            else:
                tc = np.mean([temp_cell[c] for c, v in CELLS.items() if v[1] == "cycling" and v[2] == temp])
                n = t * 1500 / 840
                socs = np.linspace(0, 1, 21)
                kc = np.mean(m.k_cal_at(socs, tc) ** (1 / m.z_cal)) ** m.z_cal
                y = 100 * (1 - kc * t ** m.z_cal - m.cycle_loss(n, tc))
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=SLOTS[k], width=2), showlegend=False,
                                     hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(title="Days since the first checkup", row=1, col=col)
    fig.update_yaxes(title="Capacity, % of fitted initial", row=1, col=1)
    fig = _layout(fig, "Large-format LFP cells: measured capacity (markers) and fitted ageing model (lines)",
                  legend_bottom=True)
    fig.update_annotations(font=dict(color=INK2, size=12))
    return fig


def cell_ica(cell_res: dict, cell: str = "CA-181103-22", current: float = 18.0) -> go.Figure:
    curves = {k[1]: v for k, v in cell_res["curves"].items() if k[0] == cell and k[2] == current}
    caps = cell_res["caps"].set_index(["cell", "checkup", "current_a"])["q_dis_ah"]
    fig = go.Figure()
    steps = [BLUES[2], BLUES[4], BLUES[6]]
    for col, checkup in zip(steps, sorted(curves)):
        q = caps.get((cell, checkup, current), np.nan)
        fig.add_trace(go.Scatter(x=curves[checkup].index, y=curves[checkup].values, mode="lines",
                                 name=f"after {checkup} cycles ({q:.0f} Ah)", line=dict(color=col, width=2)))
    fig.update_xaxes(title="Cell voltage (V)")
    fig.update_yaxes(title="dQ/dV (Ah/V)")
    paper, mode, temp, _ = CELLS[cell]
    return _layout(fig, f"Incremental capacity of a cell cycled at {temp} °C (C/10 charge): "
                        "peaks shrink and shift as the cell ages")


def cell_regression(cell_res: dict) -> go.Figure:
    p = cell_res["ml"]["predictions"]
    fig = go.Figure()
    lo, hi = p[["q_dis_ah", "predicted_ah"]].min().min() - 2, p[["q_dis_ah", "predicted_ah"]].max().max() + 2
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=AXIS, width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p["q_dis_ah"], y=p["predicted_ah"], mode="markers", name="Leave-one-cell-out prediction",
                             marker=dict(color=SLOTS[0], size=9, line=dict(color=SURFACE, width=1)),
                             text=p["cell"] + ", checkup " + p["checkup"].astype(str),
                             hovertemplate="%{text}<br>measured %{x:.1f} Ah<br>predicted %{y:.1f} Ah<extra></extra>"))
    fig.update_xaxes(title="Measured capacity (Ah)", range=[lo, hi])
    fig.update_yaxes(title="Predicted from the 50 A charge curve (Ah)", range=[lo, hi])
    ml = cell_res["ml"]
    return _layout(fig, f"Capacity from voltage-curve features: MAE {ml['mae_pct']:.1f} % "
                        f"(baseline {ml['baseline_mae_pct']:.1f} %)", width=620, height=520, legend=False)


# ----------------------------------------------------------------------------------------------------- strategies

def soc_heatmaps(strat: dict, duid: str, names: list[str]) -> go.Figure:
    titles = [_style(n)[0] for n in names]
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=titles)
    scale = [[i / (len(BLUES) - 1), c] for i, c in enumerate(BLUES)]
    for i, name in enumerate(names, start=1):
        s = strat["evals"][(duid, name)]["soc"]
        frame = pd.DataFrame({"day": s.index.normalize(), "hour": s.index.hour + s.index.minute / 60,
                              "soc": 100 * s.to_numpy()})
        frame["hour"] = (frame["hour"] * 2).astype(int) / 2                   # half-hour cells
        grid = frame.pivot_table(index="hour", columns="day", values="soc", aggfunc="mean")
        fig.add_trace(go.Heatmap(z=grid.to_numpy(), x=grid.columns, y=grid.index, zmin=0, zmax=100,
                                 colorscale=scale, showscale=i == 1,
                                 colorbar=dict(title="SoC %", len=0.9, thickness=12),
                                 hovertemplate="%{x|%d %b %Y} %{y:.1f} h<br>SoC %{z:.0f} %<extra></extra>"),
                      row=i, col=1)
        fig.update_yaxes(title="Hour", autorange="reversed", dtick=6, row=i, col=1)
    name = strat["table"].set_index("duid").loc[duid, "name"]
    name = name.iloc[0] if isinstance(name, pd.Series) else name
    fig = _layout(fig, f"{_short(name)}: state of charge by hour of day (rows) and day (columns)",
                  height=230 * len(names) + 80, legend=False)
    fig.update_annotations(font=dict(color=INK2, size=12))
    return fig


def cycle_depths(strat: dict, names: list[str]) -> go.Figure:
    bins = np.linspace(0, 1, 11)
    labels = [f"{int(100 * a)}-{int(100 * b)}" for a, b in zip(bins[:-1], bins[1:])]
    fig = go.Figure()
    units = sorted({d for d, _ in strat["evals"]})
    for name in names:
        h = sum(depth_histogram(strat["evals"][(d, name)]["stress"].cycles, bins)["efc"]
                / strat["evals"][(d, name)]["stress"].years for d in units) / len(units)
        label, color, _ = _style(name)
        fig.add_trace(go.Bar(x=labels, y=h.to_numpy(), name=label, marker=dict(color=color),
                             hovertemplate="depth %{x} %<br>%{y:.0f} equivalent cycles a year<extra></extra>"))
    fig.update_layout(bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(title="Depth of rainflow cycle (% of capacity)")
    fig.update_yaxes(title="Equivalent full cycles per year")
    return _layout(fig, f"Where the throughput goes: rainflow cycle depths, mean of {len(units)} batteries",
                   legend_bottom=True)


def soh_fan_chart(fan: pd.DataFrame, duid: str, names: list[str], name: str,
                  bands: tuple[str, ...] = ("actual",)) -> go.Figure:
    """SoH paths of the strategies; uncertainty bands only for ``bands`` (overlapping bands become unreadable)."""
    fig = go.Figure()
    for n in names:
        g = fan[(fan["duid"] == duid) & (fan["strategy"] == n)]
        label, color, _ = _style(n)
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if n in bands:
            fig.add_trace(go.Scatter(x=np.r_[g["years"], g["years"][::-1]],
                                     y=100 * np.r_[g["soh_high"], g["soh_low"][::-1]], fill="toself",
                                     fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False, legendgroup=n))
        fig.add_trace(go.Scatter(x=g["years"], y=100 * g["soh"], name=label, line=dict(color=color, width=2),
                                 legendgroup=n, hovertemplate="year %{x:.1f}: SoH %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=80, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title="Years of operation")
    fig.update_yaxes(title="State of health (% of initial capacity)")
    return _layout(fig, f"{_short(name)}: state of health if the year's operation repeats "
                        "(bands: 20-30 °C, beta 1-1.5, linear or power-law cycle ageing)", height=480,
                   legend_bottom=True)


def revenue_vs_degradation(table: pd.DataFrame) -> go.Figure:
    units = list(dict.fromkeys(table["duid"]))
    fig = make_subplots(rows=1, cols=len(units), horizontal_spacing=0.06,
                        subplot_titles=[_short(table.loc[table["duid"] == d, "name"].iloc[0]) for d in units])
    seen = set()
    for j, duid in enumerate(units, start=1):
        g = table[table["duid"] == duid].copy()
        g["loss_pct_yr"] = g["loss_10y_pct"] / 10
        aware = g[g["strategy"].str.match(r"pf_aware_\d+$")].sort_values("loss_pct_yr")
        fig.add_trace(go.Scatter(x=aware["loss_pct_yr"], y=aware["revenue_k_aud_per_mw_yr"], mode="lines",
                                 line=dict(color=SLOTS[6], width=2), showlegend=False, hoverinfo="skip"),
                      row=1, col=j)
        for _, r in g.iterrows():
            label, color, symbol = _style(r["strategy"])
            key = label
            extra = f"<br>replacement cost {r['strategy'].split('_')[2]} AUD/kWh" if r["strategy"].startswith("pf_aware") else ""
            fig.add_trace(go.Scatter(
                x=[r["loss_pct_yr"]], y=[r["revenue_k_aud_per_mw_yr"]], mode="markers", name=label,
                legendgroup=key, showlegend=key not in seen,
                marker=dict(color=color, size=11, symbol=symbol, line=dict(color=SURFACE, width=2)),
                hovertemplate=(f"{label}{extra}<br>%{{y:.0f}} k AUD/MW-yr<br>%{{x:.2f}} % capacity/yr"
                               f"<br>{r['cycles_per_day']:.2f} cycles/day<extra></extra>")), row=1, col=j)
            seen.add(key)
        fig.update_xaxes(title="Capacity lost, %/yr (10-yr avg)", row=1, col=j)
    fig.update_yaxes(title="Revenue (k AUD per MW-year)", row=1, col=1)
    fig = _layout(fig, "Revenue against ageing: actual operation vs. perfect-foresight strategies",
                  height=430, width=1100, legend_bottom=True)
    fig.update_annotations(font=dict(color=INK2, size=12))
    return fig


# ------------------------------------------------------------------------------------------------------- output

def make_all(soc_res: dict, fleet_monthly: pd.DataFrame, fleet: pd.DataFrame, cell_res: dict, strat: dict,
             fan: pd.DataFrame, img_dir: Path, dashboard: Path, headline: str = "pf_aware_250",
             heatmap_unit: str = "BLYTHB1", fan_unit: str = "BLYTHB1") -> None:
    names = ["actual", "pf_max", "pf_rule", headline]
    table = strat["table"]
    unit_name = table.loc[table["duid"] == fan_unit, "name"].iloc[0]
    # soc_week plots ACEN's own SoC series, which has no stated licence: PNG for the README only, not embedded
    # (with its data) in the HTML dashboard
    png_only = {"soc_week"}
    figures = [
        ("soc_week", "SoC reconstruction validated on published operator data", soc_week(soc_res)),
        ("soc_error", None, soc_error_by_hour(soc_res)),
        ("fleet", "Stored energy reported to AEMO across the fleet", fleet_chart(fleet)),
        ("cell_capacity", "State of health from cell data", cell_capacity(cell_res)),
        ("cell_ica", None, cell_ica(cell_res)),
        ("cell_regression", None, cell_regression(cell_res)),
        ("soc_heatmap", "Strategies: SoC, cycle depth, ageing and revenue",
         soc_heatmaps(strat, heatmap_unit, ["actual", "pf_max", headline])),
        ("cycle_depths", None, cycle_depths(strat, names)),
        ("soh_fan", None, soh_fan_chart(fan, fan_unit, names, unit_name, bands=("actual", headline))),
        ("revenue_vs_degradation", None, revenue_vs_degradation(table)),
    ]
    img_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for key, section, fig in figures:
        fig.write_image(img_dir / f"{key}.png", scale=2)
        log.info("chart %s", key)
        if section:
            parts.append(f"<h2>{section}</h2>")
        if key in png_only:
            parts.append(f"<p><img src='img/{key}.png' alt='{key}' style='max-width:100%'></p>")
            continue
        first = not any("<div" in s for s in parts)
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                                 config={"displaylogo": False, "responsive": True}))
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>BESS SoC, SoH and strategy</title>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<style>body{{background:{SURFACE};color:{INK};font-family:{FONT};max-width:1150px;margin:0 auto;"
            f"padding:16px}} h1{{font-size:22px}} h2{{font-size:17px;margin-top:36px;color:{INK2}}}</style></head>"
            f"<body><h1>Grid battery: state of charge, state of health and operating strategy</h1>"
            + "\n".join(parts) + "</body></html>")
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.write_text(html, encoding="utf-8")
