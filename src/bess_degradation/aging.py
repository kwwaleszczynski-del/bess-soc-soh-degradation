"""Semi-empirical capacity-fade model of an LFP battery: calendar ageing plus cycle ageing counted by rainflow.

Capacity loss (fraction of the initial capacity) is the sum of two power laws:

    Q_cal(t) = k_cal * A(T, Ea_cal) * exp(gamma * (SoC - 0.5)) * t ** z_cal          t in days
    Q_cyc(N) = k_cyc * A(T, Ea_cyc) * N_eq ** z_cyc,   N_eq = sum_i n_i * DoD_i ** beta
    A(T, Ea) = exp(-Ea / R * (1 / T - 1 / T_ref)),    T_ref = 25 degC

N_eq counts rainflow cycles (n_i = 1 or 0.5, depth DoD_i) as equivalent 100 %-deep cycles; beta = 1 makes cycle
ageing depend on energy throughput only, beta > 1 makes deep cycles more harmful per MWh than shallow ones (the
Woehler-type cycle-life curves of cell data sheets). This is the structure of e.g. Schmalstieg et al. (J. Power
Sources 257, 2014) and Naumann et al. (J. Energy Storage 17, 2018; J. Power Sources 451, 2020) with the
rainflow-based cycle stress of Xu et al. (IEEE Trans. Smart Grid 9, 2018).

Conditions that vary in time are combined with the "virtual time" rule, which is exact for the power law: for
Q = k t**z, the quantity Q**(1/z) grows at the rate k**(1/z), so over a period with a distribution of SoC and
temperature Q(t) = (sum_i k_i**(1/z) dt_i)**z.

Parameters k_cal, Ea_cal, gamma, z_cal, k_cyc, Ea_cyc and z_cyc are fitted to large-format LFP cell data
(:mod:`bess_degradation.cellfit`); beta cannot be identified from those (all cycling at 100 % depth) and is taken as
an assumption with a sensitivity range.

Usage:
    from bess_degradation.aging import AgingModel, profile_stress, soh_path
    stress = profile_stress(soc_5min, dt_hours=5/60)
    soh = soh_path(model, stress, years=15, temp_c=25)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from bess_degradation.rainflow import count_cycles

R_GAS = 8.314
T_REF = 298.15


def arrhenius(temp_c, ea: float) -> np.ndarray:
    return np.exp(-ea / R_GAS * (1.0 / (np.asarray(temp_c, float) + 273.15) - 1.0 / T_REF))


@dataclass(frozen=True)
class AgingModel:
    k_cal: float = 2.0e-3          # fraction per day**z_cal at 25 degC and SoC 0.5
    ea_cal: float = 37_000.0       # J/mol
    gamma: float = 1.0             # SoC sensitivity of calendar ageing (per unit SoC, exponential)
    z_cal: float = 0.5
    k_cyc: float = 3.0e-5          # fraction per equivalent full cycle**z_cyc at 25 degC
    ea_cyc: float = 30_000.0       # J/mol
    z_cyc: float = 1.0
    beta: float = 1.0              # depth-of-cycle exponent (assumption, see module docstring)

    def k_cal_at(self, soc, temp_c) -> np.ndarray:
        return self.k_cal * arrhenius(temp_c, self.ea_cal) * np.exp(self.gamma * (np.asarray(soc, float) - 0.5))

    def k_cyc_at(self, temp_c) -> np.ndarray:
        return self.k_cyc * arrhenius(temp_c, self.ea_cyc)

    def calendar_loss(self, days, soc, temp_c) -> np.ndarray:
        """Calendar loss after ``days`` at constant SoC and temperature."""
        return self.k_cal_at(soc, temp_c) * np.asarray(days, float) ** self.z_cal

    def cycle_loss(self, n_eq, temp_c) -> np.ndarray:
        return self.k_cyc_at(temp_c) * np.asarray(n_eq, float) ** self.z_cyc

    def with_(self, **kw) -> "AgingModel":
        return replace(self, **kw)


@dataclass(frozen=True)
class Stress:
    """Ageing-relevant summary of an operating profile, per year of operation."""
    soc_hist: pd.Series          # share of time in each SoC bin (index: bin centres)
    cycles: pd.DataFrame         # rainflow cycles of the profile (depth, mean, count, start, end)
    years: float                 # length of the profile, years
    fec_per_year: float          # equivalent full cycles per year (sum depth * count)
    mean_soc: float

    def n_eq_per_year(self, beta: float) -> float:
        c = self.cycles
        return float((c["count"] * c["depth"] ** beta).sum() / self.years)


def profile_stress(soc, dt_hours: float = 5 / 60, bins: int = 20, min_range: float = 0.0) -> Stress:
    """SoC histogram and rainflow cycles of a SoC profile (fraction, evenly spaced every ``dt_hours``)."""
    s = np.clip(np.asarray(soc, float), 0.0, 1.0)
    s = s[~np.isnan(s)]
    b = np.minimum((s * bins).astype(int), bins - 1)
    count = np.bincount(b, minlength=bins)
    total = np.bincount(b, weights=s, minlength=bins)
    used = count > 0
    level = total[used] / count[used]                 # mean SoC within each occupied bin (exact for flat profiles)
    cycles = count_cycles(s, min_range)
    years = len(s) * dt_hours / 8760
    return Stress(soc_hist=pd.Series(count[used] / count.sum(), index=level), cycles=cycles, years=years,
                  fec_per_year=float((cycles["depth"] * cycles["count"]).sum() / years), mean_soc=float(s.mean()))


def calendar_rate(model: AgingModel, stress: Stress, temp_c) -> float:
    """Effective k of the calendar power law for a profile: (time-weighted mean of k**(1/z))**z."""
    k = model.k_cal_at(stress.soc_hist.index.to_numpy(), temp_c)
    return float(np.sum(stress.soc_hist.to_numpy() * k ** (1 / model.z_cal)) ** model.z_cal)


def soh_path(model: AgingModel, stress: Stress, years: float = 15.0, temp_c: float = 25.0,
             steps_per_year: int = 12) -> pd.DataFrame:
    """State of health over time if the profile repeats every year (operation defined in SoC terms)."""
    t = np.arange(0, int(round(years * steps_per_year)) + 1) / steps_per_year
    cal = calendar_rate(model, stress, temp_c) * (t * 365.25) ** model.z_cal
    cyc = model.cycle_loss(stress.n_eq_per_year(model.beta) * t, temp_c)
    return pd.DataFrame({"years": t, "calendar_loss": cal, "cycle_loss": cyc, "soh": 1.0 - cal - cyc})


def years_to(soh: pd.DataFrame, threshold: float = 0.8) -> float:
    below = soh.loc[soh["soh"] <= threshold, "years"]
    if below.empty:
        return float("nan")
    i = below.index[0]
    if i == 0:
        return 0.0
    a, b = soh.loc[i - 1], soh.loc[i]
    return float(a["years"] + (a["soh"] - threshold) / (a["soh"] - b["soh"]) * (b["years"] - a["years"]))


def loss_first_years(model: AgingModel, stress: Stress, temp_c: float = 25.0, years: float = 1.0) -> dict:
    """Capacity loss (fraction) after ``years`` of the profile, split into calendar and cycle ageing."""
    p = soh_path(model, stress, years=years, temp_c=temp_c, steps_per_year=1)
    last = p.iloc[-1]
    return {"calendar": float(last["calendar_loss"]), "cycle": float(last["cycle_loss"]),
            "total": float(last["calendar_loss"] + last["cycle_loss"])}


# ---------------------------------------------------------------------------------------------------------------------
# Linear cost coefficients for a dispatch optimiser
# ---------------------------------------------------------------------------------------------------------------------

def segment_costs(model: AgingModel, energy_mwh: float, fade_cost_aud: float, segments: int = 8,
                  temp_c: float = 25.0, life_cycles: float | None = None) -> np.ndarray:
    """Marginal ageing cost (AUD per MWh of stored energy discharged) of each SoC segment, shallow to deep.

    A cycle of depth DoD costs ``fade_cost_aud * dQ(DoD)`` with dQ the capacity lost per such cycle; discharging the
    top j segments of a battery split into ``segments`` equal parts costs the sum of the first j coefficients times
    the segment energy (the piecewise-linear cycle-depth cost of Xu et al., IEEE Trans. Power Syst. 33, 2018). With
    z_cyc < 1 the per-cycle loss is evaluated as the average over ``life_cycles`` equivalent cycles.
    """
    k = float(model.k_cyc_at(temp_c))
    if model.z_cyc != 1.0:
        n = life_cycles or 5000.0
        k = k * n ** model.z_cyc / n                  # average loss per equivalent cycle over n cycles
    depth = np.arange(segments + 1) / segments
    phi = k * depth ** model.beta                    # loss per cycle of depth d
    return fade_cost_aud * np.diff(phi) * segments / energy_mwh


def holding_cost(model: AgingModel, energy_mwh: float, fade_cost_aud: float, temp_c: float = 25.0,
                 life_years: float = 15.0) -> float:
    """AUD per MWh stored per hour: the extra calendar ageing of a fuller battery, linearised over SoC 0..1 and
    averaged over ``life_years`` (the power law makes the early years more expensive)."""
    days = life_years * 365.25
    rate = model.k_cal_at(np.array([0.0, 1.0]), temp_c) * days ** model.z_cal / (days * 24)   # loss per hour
    return float(fade_cost_aud * (rate[1] - rate[0]) / energy_mwh)
