import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swc.config import Params
from swc.evaluate import evaluate
from swc.instance import generate_synthetic
from swc.model_cpsat import DailyModel
from swc.pareto import epsilon_front, hypervolume_2d
from swc.risk import mandatory_bins, overflow_risk, skip_penalty

W = 4  # keep tests light


def make(n=8, seed=0, ovf=60.0, bpp=(4, 10), fleet=None, **ops):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=ovf)
    if ops:
        P.ops = replace(P.ops, **ops)
    inst = generate_synthetic(n, seed=seed, params=P, bins_per_point=bpp,
                              fleet_counts=fleet or {"new_compactor": 1, "old_compactor": 1, "small_loader": 1})
    risk = overflow_risk(inst, mode="count")
    return inst, risk, skip_penalty(inst, risk), mandatory_bins(inst)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("objective", ["cost", "co2"])
def test_solver_matches_checker(seed, objective):
    inst, risk, pen, mand = make(seed=seed)
    s = DailyModel(inst, penalty=pen, mandatory=mand).solve(objective, time_limit=20, workers=W)
    assert s.status == "OPTIMAL"
    k = evaluate(inst, s.routes, pen, risk, mand)
    assert k.feasible, k.violations
    cs, es = inst.params.scale.cost_scale, inst.params.scale.co2_scale
    assert s.cost_int / cs == pytest.approx(k.cost_total, rel=2e-3, abs=0.05)
    assert s.co2_int / es / 1000 == pytest.approx(k.co2_kg, rel=2e-3, abs=0.01)


def test_all_mandatory_served_and_capacity_forces_trips():
    # every point mandatory, total waste larger than any single truckload -> several trips/trucks
    inst, risk, pen, _ = make(n=10, seed=5, bpp=(8, 12))
    mand = set(inst.bins)
    s = DailyModel(inst, penalty=pen, mandatory=mand).solve("cost", time_limit=60, workers=W)
    k = evaluate(inst, s.routes, pen, risk, mand)
    assert k.feasible, k.violations
    assert k.bins_served == inst.n
    trips = sum(r.count(inst.disposal) for r in s.routes if r)
    assert trips >= int(np.ceil(inst.demand_kg().sum() / max(v.capacity_kg for v in inst.vehicles)))


def test_drop_worthless_is_exact():
    inst, risk, pen, mand = make(n=9, seed=3)
    a = DailyModel(inst, penalty=pen, mandatory=mand, drop_worthless=True).solve("cost", time_limit=30, workers=W)
    b = DailyModel(inst, penalty=pen, mandatory=mand, drop_worthless=False).solve("cost", time_limit=60, workers=W)
    assert a.status == b.status == "OPTIMAL"
    assert a.cost_int == b.cost_int


def test_infeasible_when_shift_too_short():
    inst, risk, pen, _ = make(n=8, seed=1, shift_s=1800)
    s = DailyModel(inst, penalty=pen, mandatory=set(inst.bins)).solve("cost", time_limit=20, workers=W)
    assert s.status == "INFEASIBLE"


def test_pareto_points_are_mutually_nondominated():
    inst, risk, pen, mand = make(n=8, seed=2, bpp=(6, 10),
                                 fleet={"new_compactor": 1, "old_compactor": 2, "small_loader": 1})
    front = epsilon_front(DailyModel(inst, penalty=pen, mandatory=mand), grid=5, time_limit=20, workers=W)
    pts = [(p.cost_int, p.co2_int) for p in front]
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i != j:
                assert not (b[0] <= a[0] and b[1] <= a[1]), (a, b)
    assert hypervolume_2d([(1, 3), (2, 1)], (4, 4)) == pytest.approx(3 * 1 + 2 * 2)


def test_pyvrp_engine_feasible_and_matches_exact_on_small():
    from swc.heuristic_pyvrp import solve_pyvrp
    inst, risk, pen, mand = make(n=8, seed=4)
    ex = DailyModel(inst, penalty=pen, mandatory=mand).solve("cost", time_limit=30, workers=W)
    he = solve_pyvrp(inst, pen, mand, time_s=2, seed=0)
    ke, kh = evaluate(inst, ex.routes, pen, risk, mand), evaluate(inst, he.routes, pen, risk, mand)
    assert kh.feasible, kh.violations
    assert ex.status == "OPTIMAL"
    assert kh.cost_total == pytest.approx(ke.cost_total, rel=1e-3)


def test_pyvrp_repair_restores_feasibility_when_overloaded():
    # all points optional but hugely valuable, fleet far too small -> repair must drop points
    from swc.heuristic_pyvrp import solve_pyvrp
    inst, risk, pen, _ = make(n=12, seed=6, bpp=(10, 12), fleet={"small_loader": 1})
    he = solve_pyvrp(inst, pen + 1e4, set(), time_s=2, seed=0)
    k = evaluate(inst, he.routes, pen, risk, set())
    assert he.status == "FEASIBLE" and k.feasible, k.violations
    assert 0 < k.bins_served < inst.n
