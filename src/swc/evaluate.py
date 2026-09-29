"""Independent solution checker and KPI calculator (floating point, no solver involved)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .instance import Instance


@dataclass
class KPIs:
    feasible: bool
    violations: list
    cost_total: float
    cost_fixed: float
    cost_variable: float
    cost_penalty: float
    co2_kg: float
    km: float
    trucks_used: int
    bins_served: int
    bins_skipped: int
    kg_collected: float
    expected_overflows_weighted: float  # sum over skipped bins of sensitivity * risk

    def as_dict(self):
        return asdict(self)


def evaluate(inst: Instance, routes: list, penalty: np.ndarray, risk: np.ndarray | None = None,
             mandatory: set | None = None) -> KPIs:
    p = inst.params
    D = inst.disposal
    dist, tt, svc = inst.dist_m, inst.travel_s(), inst.service_s()
    q = inst.demand_kg()
    viol = []
    served = np.zeros(inst.n, dtype=int)
    fixed = var = co2_g = metres = kg = 0.0
    used = 0
    for k, r in enumerate(routes):
        if not r:
            continue
        veh = inst.vehicles[k]
        used += 1
        fixed += veh.fixed_cost
        if r[0] != 0 or r[-1] != 0 or r[-2] != D:
            viol.append(f"truck {k}: route must be 0 -> bins -> D -> 0, got {r}")
        load, dur = 0.0, 0
        for a, b in zip(r[:-1], r[1:]):
            d = dist[a, b]
            metres += d
            var += veh.var_cost_per_km * d / 1000.0
            co2_g += d * (veh.a_g_per_m + veh.b_g_per_m_per_kg * load)
            dur += tt[a, b]
            if b in inst.bins:
                served[b - 1] += 1
                load += q[b - 1]
                dur += svc[b - 1]
                if load > veh.capacity_kg + 1e-9:
                    viol.append(f"truck {k}: capacity exceeded ({load} > {veh.capacity_kg})")
            elif b == D:
                kg += load
                load = 0.0
                dur += p.ops.unload_s
        if dur > p.ops.shift_s:
            viol.append(f"truck {k}: shift exceeded ({dur} s > {p.ops.shift_s} s)")
    if (served > 1).any():
        viol.append(f"bins served twice: {list(np.where(served > 1)[0] + 1)}")
    for i in (mandatory or ()):
        if served[i - 1] == 0:
            viol.append(f"mandatory bin {i} not served")
    skipped = served == 0
    pen = float(penalty[skipped].sum())
    wexp = float((inst.sensitivity() * risk)[skipped].sum()) if risk is not None else float("nan")
    return KPIs(
        feasible=not viol, violations=viol, cost_total=fixed + var + pen, cost_fixed=fixed,
        cost_variable=var, cost_penalty=pen, co2_kg=co2_g / 1000.0, km=metres / 1000.0,
        trucks_used=used, bins_served=int((served > 0).sum()), bins_skipped=int(skipped.sum()),
        kg_collected=kg, expected_overflows_weighted=wexp,
    )
