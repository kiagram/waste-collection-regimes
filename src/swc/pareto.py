"""Bi-objective (cost vs CO2) Pareto fronts with a lexicographic epsilon-constraint
method in the spirit of AUGMECON2 (Mavrotas & Florios 2013):

* the pay-off table is computed lexicographically (no weakly efficient end points);
* every grid point solves  min cost s.t. CO2 <= eps,  then  min CO2 s.t. cost <= cost*,
  so each returned point is efficient (when both solves are proven optimal);
* the bypass rule skips grid points already dominated by the last solution.
Only points whose both stages are OPTIMAL are labelled 'exact'.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model_cpsat import DailyModel, Solution


@dataclass
class ParetoPoint:
    cost_int: int
    co2_int: int
    solution: Solution
    exact: bool
    eps: int | None


def lexicographic(model: DailyModel, first: str, time_limit: float, **kw) -> tuple[Solution, bool]:
    s1 = model.solve(objective=first, time_limit=time_limit, **kw)
    if s1.status not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"no feasible solution ({s1.status})")
    second = "co2" if first == "cost" else "cost"
    bound = {"cost_ub": s1.cost_int} if first == "cost" else {"co2_ub": s1.co2_int}
    s2 = model.solve(objective=second, time_limit=time_limit, hint=s1, **bound, **kw)
    exact = s1.status == "OPTIMAL" and s2.status == "OPTIMAL"
    return (s2 if s2.status in ("OPTIMAL", "FEASIBLE") else s1), exact


def epsilon_front(model: DailyModel, grid: int = 10, time_limit: float = 60.0, **kw) -> list[ParetoPoint]:
    s_cost, ex1 = lexicographic(model, "cost", time_limit, **kw)   # cheapest, highest CO2 end
    s_co2, ex2 = lexicographic(model, "co2", time_limit, **kw)     # greenest, most expensive end
    pts = [ParetoPoint(s_cost.cost_int, s_cost.co2_int, s_cost, ex1, None)]
    hi, lo = s_cost.co2_int, s_co2.co2_int
    if hi > lo:
        step = (hi - lo) / grid
        eps = hi - step
        last = s_cost
        while eps > lo + 0.5 * step:
            a = model.solve(objective="cost", co2_ub=int(eps), time_limit=time_limit, hint=last, **kw)
            if a.status not in ("OPTIMAL", "FEASIBLE"):
                eps -= step
                continue
            b = model.solve(objective="co2", cost_ub=a.cost_int, time_limit=time_limit, hint=a, **kw)
            s = b if b.status in ("OPTIMAL", "FEASIBLE") else a
            pts.append(ParetoPoint(s.cost_int, s.co2_int, s,
                                   a.status == "OPTIMAL" and b.status == "OPTIMAL", int(eps)))
            last = s
            # bypass: jump below the CO2 actually achieved
            eps = min(eps - step, s.co2_int - 1)
    pts.append(ParetoPoint(s_co2.cost_int, s_co2.co2_int, s_co2, ex2, None))
    return nondominated(pts)


def nondominated(pts: list[ParetoPoint]) -> list[ParetoPoint]:
    uniq = {}
    for p in pts:
        key = (p.cost_int, p.co2_int)
        if key not in uniq or (p.exact and not uniq[key].exact):
            uniq[key] = p
    pts = sorted(uniq.values(), key=lambda p: (p.cost_int, p.co2_int))
    out, best_co2 = [], float("inf")
    for p in pts:
        if p.co2_int < best_co2:
            out.append(p)
            best_co2 = p.co2_int
    return out


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """Area dominated by a minimisation front w.r.t. reference point `ref`."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv
