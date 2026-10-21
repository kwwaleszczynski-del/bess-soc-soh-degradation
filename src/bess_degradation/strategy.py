"""Perfect-foresight arbitrage of a battery on 5-minute prices, with an optional cost of ageing in the objective.

For one trading day with prices p_t (AUD/MWh) the battery maximises

    sum_t p_t dt (d_t - c_t)  -  w * sum_t dt d_t  -  sum_j g_j * sum_t dt d_jt / eta  -  h * sum_t dt e_t

where the stored energy e_t is split into J equal SoC segments (e_t = sum_j e_jt, c_t = sum_j c_jt,
d_t = sum_j d_jt):

* w   flat cost per MWh discharged (the usual "cycling cost"),
* g_j cost per MWh of stored energy taken out of segment j; increasing in j, so shallow cycles are cheap per MWh and
      deep ones expensive (piecewise-linear cycle-depth cost, Xu et al., IEEE Trans. Power Syst. 33, 2018),
* h   cost per MWh-hour of stored energy: calendar ageing is faster at high SoC, so a full battery costs more to
      hold than an empty one.

Constraints per segment: e_jt = e_j,t-1 + eta dt c_jt - dt d_jt / eta, 0 <= e_jt <= E / J; for the whole battery:
c_t <= P_c, d_t <= P_d, c_t / P_c + d_t / P_d <= 1, the day ends with at least the stored energy it started with
(the start level is free), an optional cap on daily discharge, and no charging and discharging in the same interval.
As in the companion project the last condition gets binary variables only for intervals where the LP relaxation
violates it, repeating until none does (the result is then optimal for the MILP).

Usage:
    from bess_degradation.strategy import Battery, AgingCost, optimize_day
    res = optimize_day(prices, Battery(100, 100, 200, 0.88), AgingCost(segment=np.array([5, 10, 20, 40])))
    res.schedule   # price, charge_mw, discharge_mw, energy_mwh (end of interval)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

DT_5MIN = 5 / 60
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Battery:
    discharge_mw: float = 1.0
    charge_mw: float = 1.0
    energy_mwh: float = 2.0
    round_trip_efficiency: float = 0.88

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


@dataclass(frozen=True)
class AgingCost:
    throughput: float = 0.0                                          # AUD per MWh discharged
    segment: np.ndarray = field(default_factory=lambda: np.zeros(1))  # AUD per MWh stored energy, shallow -> deep
    holding: float = 0.0                                             # AUD per MWh stored per hour

    @property
    def segments(self) -> int:
        return len(self.segment)


@dataclass
class DayResult:
    revenue: float           # AUD, energy only
    objective: float         # revenue minus the modelled ageing cost
    charged_mwh: float
    discharged_mwh: float
    binaries: int
    schedule: pd.DataFrame   # price, charge_mw, discharge_mw, energy_mwh (stored energy at interval end)


def optimize_day(prices, battery: Battery, cost: AgingCost = AgingCost(), e_start: float | None = None,
                 max_discharge_mwh: float | None = None, dt_hours: float = DT_5MIN,
                 time_limit: float = 60.0) -> DayResult:
    p = np.asarray(prices, dtype=float)
    if np.isnan(p).any():
        raise ValueError("prices contain NaN")
    n = len(p)
    binary = np.array([], dtype=int)
    while True:
        x = _solve(p, battery, cost, e_start, max_discharge_mwh, dt_hours, binary, time_limit)
        j = cost.segments
        charge = x[:n * j].reshape(j, n).sum(axis=0)
        discharge = x[n * j:2 * n * j].reshape(j, n).sum(axis=0)
        both = np.flatnonzero((charge > 1e-6) & (discharge > 1e-6))
        new = np.setdiff1d(both, binary)
        if len(new) == 0:
            break
        binary = np.union1d(binary, new)
    energy = x[2 * n * j:3 * n * j].reshape(j, n).sum(axis=0)
    charge[charge < 1e-9] = 0.0
    discharge[discharge < 1e-9] = 0.0
    index = prices.index if isinstance(prices, pd.Series) else pd.RangeIndex(n)
    schedule = pd.DataFrame({"price": p, "charge_mw": charge, "discharge_mw": discharge, "energy_mwh": energy},
                            index=index)
    revenue = float(np.sum(p * dt_hours * (discharge - charge)))
    d_seg = x[n * j:2 * n * j].reshape(j, n)
    aging = (cost.throughput * discharge.sum() * dt_hours
             + float(cost.segment @ d_seg.sum(axis=1)) * dt_hours / battery.eta
             + cost.holding * energy.sum() * dt_hours)
    return DayResult(revenue=revenue, objective=revenue - aging, charged_mwh=float(charge.sum() * dt_hours),
                     discharged_mwh=float(discharge.sum() * dt_hours), binaries=len(binary), schedule=schedule)


def _solve(p, battery: Battery, cost: AgingCost, e_start, max_discharge_mwh, dt, binary, time_limit,
           _retry: bool = False) -> np.ndarray:
    """Variables: c[j,t], d[j,t], e[j,t] (row-major by segment), e0[j], u[binary]."""
    n, J, m = len(p), cost.segments, len(binary)
    eta, pc, pdis = battery.eta, battery.charge_mw, battery.discharge_mw
    seg_cap = battery.energy_mwh / J
    ic, idd, ie, i0, iu = 0, n * J, 2 * n * J, 3 * n * J, 3 * n * J + J
    nv = iu + m

    obj = np.zeros(nv)                                      # minimise -profit
    obj[ic:ic + n * J] = np.tile(p * dt, J)
    obj[idd:idd + n * J] = np.concatenate([-(p - cost.throughput - g / eta) * dt for g in cost.segment])
    obj[ie:ie + n * J] = cost.holding * dt

    eye_n = sparse.identity(n, format="csr")
    eye_j = sparse.identity(J, format="csr")
    shift = sparse.eye(n, k=-1, format="csr")
    first = sparse.csr_matrix(([-1.0], ([0], [0])), shape=(n, 1))
    # e_jt - e_j,t-1 - eta dt c_jt + dt/eta d_jt = 0, with e_j,-1 = e0_j
    balance = sparse.hstack([sparse.kron(eye_j, -eta * dt * eye_n), sparse.kron(eye_j, (dt / eta) * eye_n),
                             sparse.kron(eye_j, eye_n - shift), sparse.kron(eye_j, first),
                             sparse.csr_matrix((n * J, m))], format="csr")
    ones_j = sparse.csr_matrix(np.ones((1, J)))
    sum_c = sparse.kron(ones_j, eye_n)                      # (n, nJ): total charge per interval
    zeros_nnj = sparse.csr_matrix((n, n * J))
    hull = sparse.hstack([sum_c / pc, sum_c / pdis, zeros_nnj, sparse.csr_matrix((n, J)),
                          sparse.csr_matrix((n, m))], format="csr")
    cons = [LinearConstraint(balance, 0.0, 0.0), LinearConstraint(hull, -np.inf, 1.0)]
    # end of day: sum_j e_j,T-1 - sum_j e0_j >= 0
    end = np.zeros(nv)
    end[ie + np.arange(J) * n + n - 1] = 1.0
    end[i0:i0 + J] = -1.0
    cons.append(LinearConstraint(sparse.csr_matrix(end), 0.0, np.inf))
    if e_start is not None:
        start = np.zeros(nv)
        start[i0:i0 + J] = 1.0
        e0 = float(np.clip(e_start, 0, battery.energy_mwh))
        cons.append(LinearConstraint(sparse.csr_matrix(start), e0, e0))
    if max_discharge_mwh is not None:
        thr = np.zeros(nv)
        thr[idd:idd + n * J] = dt
        cons.append(LinearConstraint(sparse.csr_matrix(thr), 0.0, max(max_discharge_mwh, 0.0)))
    if m:
        rows = np.arange(m)
        sel_u = sparse.csr_matrix((np.ones(m), (rows, iu + rows)), shape=(m, nv))
        pick = sparse.csr_matrix((np.ones(m), (rows, binary)), shape=(m, n))
        tot_c = sparse.hstack([pick @ sum_c, sparse.csr_matrix((m, nv - n * J))], format="csr")
        tot_d = sparse.hstack([sparse.csr_matrix((m, n * J)), pick @ sum_c, sparse.csr_matrix((m, nv - 2 * n * J))],
                              format="csr")
        cons.append(LinearConstraint(tot_c - pc * sel_u, -np.inf, 0.0))          # c_t <= Pc u_t
        cons.append(LinearConstraint(tot_d + pdis * sel_u, -np.inf, pdis))        # d_t <= Pd (1 - u_t)

    lb = np.zeros(nv)
    ub = np.concatenate([np.full(n * J, pc), np.full(n * J, pdis), np.full(n * J, seg_cap), np.full(J, seg_cap),
                         np.ones(m)])
    integrality = np.concatenate([np.zeros(iu), np.ones(m)])
    res = milp(obj, constraints=cons, bounds=Bounds(lb, ub), integrality=integrality,
               options={"disp": False, "mip_rel_gap": 1e-6, "time_limit": time_limit})
    if res.success:
        return res.x
    if res.status == 1 and res.x is not None:
        log.warning("MILP hit the %.0f s time limit with %d binaries; using best solution", time_limit, m)
        return res.x
    if max_discharge_mwh and not _retry:            # HiGHS: rare "unknown" status on exactly attainable caps
        return _solve(p, battery, cost, e_start, max_discharge_mwh * (1 - 1e-7), dt, binary, time_limit, True)
    raise RuntimeError(f"MILP failed: {res.message}")
