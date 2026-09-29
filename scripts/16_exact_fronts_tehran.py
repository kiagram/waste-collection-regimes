"""Exact cost-CO2 Pareto fronts on Tehran District 6 sub-districts (CP-SAT), and how close the
large-scale heuristic gets.

Sub-districts: S compact neighbourhoods of the OSM instance (farthest-point seeds; each takes its n
nearest collection points by road distance). One night at a mid-cycle fill state (same rule as
09_tehran_sim.py), calibrated variability (CV 1.0), zone chance constraints and overflow penalties.
Fleet per sub-district: 1 new compactor, 2 old compactors, 1 small loader, up to 2 trips each.

For each sub-district:
  exact     lexicographic epsilon-constraint front (pareto.epsilon_front), every point re-checked
            by the independent evaluator; 'exact' = both stages proven optimal
  heuristic PyVRP with a carbon price lambda (0 ... 2000 USD/t, empty-truck emissions per km) --
            its non-dominated points are compared with the exact front (hypervolume ratio), and
            its lambda = 0 cost with the exact minimum cost
Two service modes:
  selective  the risk-based model: optional points may be skipped at their overflow penalty
  full       every point is collected, which isolates the fleet / sequencing trade-off
Resumable (one JSON per sub-district). usage:
  python scripts/16_exact_fronts_tehran.py [S] [n] [grid] [time_limit_s] [workers] [mode]
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import Params
from swc.evaluate import evaluate
from swc.heuristic_pyvrp import solve_pyvrp_best
from swc.model_cpsat import DailyModel
from swc.pareto import epsilon_front, hypervolume_2d, nondominated
from swc.risk import mandatory_bins, overflow_risk, skip_penalty
from swc.tehran import INSTANCE_VERSION, TEHRAN_FLEET, load

S = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
GRID = int(sys.argv[3]) if len(sys.argv) > 3 else 6
TL = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 16
MODE = sys.argv[6] if len(sys.argv) > 6 else "selective"
assert MODE in ("selective", "full")
LAMBDAS = [0, 50, 100, 200, 400, 800, 1200, 2000]
FLEET = ["new_compactor", "old_compactor", "old_compactor", "small_loader"]
OUT = ROOT / "results" / f"exact_fronts_tehran_{INSTANCE_VERSION}_{MODE}_s{S}_n{N}_g{GRID}_tl{TL:g}"
OUT.mkdir(parents=True, exist_ok=True)


def city():
    P = Params()
    P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=1.0)
    inst = load(P)
    rng = np.random.default_rng(50)                   # rep 0 of 09_tehran_sim.py
    inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
    return inst


def neighbourhoods(inst):
    """Farthest-point seeds over the collection points; each seed takes its N nearest points."""
    pts = np.array(list(inst.bins))
    xy = inst.xy_m[pts]
    d = np.minimum(inst.dist_m, inst.dist_m.T)
    seeds = [int(pts[np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1))])]
    while len(seeds) < S:
        dmin = np.min(np.linalg.norm(xy[:, None, :] - inst.xy_m[seeds][None, :, :], axis=2), axis=1)
        seeds.append(int(pts[np.argmax(dmin)]))
    return [sorted(int(j) for j in pts[np.argsort(d[s, pts])][:N]) for s in seeds]


def point(inst, routes, pen, risk, mand, **extra):
    k = evaluate(inst, routes, pen, risk, mand)
    used = [inst.vehicles[t].name for t, r in enumerate(routes) if r]
    return dict(cost=k.cost_total, co2_kg=k.co2_kg, km=k.km, served=k.bins_served, feasible=k.feasible,
                t_collected=k.kg_collected / 1000, trucks=used, **extra)


def run(idx, members, inst0):
    f = OUT / f"sub{idx}.json"
    if f.exists():
        return json.loads(f.read_text())
    inst = inst0.subset(members, name=f"tehran_sub{idx}")
    inst.vehicles = [TEHRAN_FLEET[v] for v in FLEET]
    risk = overflow_risk(inst, mode="count")
    pen = skip_penalty(inst, risk)
    mand = mandatory_bins(inst) if MODE == "selective" else set(inst.bins)
    info = dict(mode=MODE, sub=idx, points=N, bins=int(inst.bins_per_point.sum()), mandatory=len(mand),
                demand_t=float(inst.demand_kg().sum() / 1000), members=members)
    print(f"sub {idx}: {info['bins']} bins, {len(mand)} mandatory of {N}, {info['demand_t']:.1f} t", flush=True)

    model = DailyModel(inst, penalty=pen, mandatory=mand)
    front = epsilon_front(model, grid=GRID, time_limit=TL, workers=WORKERS)
    exact = [point(inst, p.solution.routes, pen, risk, mand, exact=p.exact, cost_int=p.cost_int,
                   co2_int=p.co2_int, wall_s=p.solution.wall_s) for p in front]
    for e in exact:
        print(f"   exact  cost={e['cost']:7.1f} CO2={e['co2_kg']:6.1f} km={e['km']:5.1f} served={e['served']:2d} "
              f"trucks={e['trucks']} proven={e['exact']} ok={e['feasible']}", flush=True)

    heur = []
    for lam in LAMBDAS:
        sol = solve_pyvrp_best(inst, pen, mand, starts=2, time_s=120, seed=0, max_iterations=5000, carbon_price=lam)
        heur.append(point(inst, sol.routes, pen, risk, mand, carbon_price=lam, wall_s=sol.wall_s))
        h = heur[-1]
        print(f"   pyvrp  lambda={lam:<5} cost={h['cost']:7.1f} CO2={h['co2_kg']:6.1f} trucks={h['trucks']} "
              f"ok={h['feasible']}", flush=True)
    res = dict(info=info, exact=exact, heuristic=heur)
    f.write_text(json.dumps(res, indent=1, default=str))
    return res


def summarise(results):
    rows = []
    for r in results:
        ex = pd.DataFrame(r["exact"])
        he = pd.DataFrame(r["heuristic"])
        he = he[he.feasible]
        # non-dominated heuristic points
        hs = he.sort_values(["cost", "co2_kg"])
        hnd, best = [], np.inf
        for _, x in hs.iterrows():
            if x.co2_kg < best - 1e-9:
                hnd.append((x.cost, x.co2_kg))
                best = x.co2_kg
        epts = list(zip(ex.cost, ex.co2_kg))
        ref = (max(ex.cost.max(), he.cost.max()) * 1.02, max(ex.co2_kg.max(), he.co2_kg.max()) * 1.02)
        hv_e, hv_h = hypervolume_2d(epts, ref), hypervolume_2d(hnd, ref)
        cmin, cmax = ex.loc[ex.cost.idxmin()], ex.loc[ex.co2_kg.idxmin()]
        dco2 = cmin.co2_kg - cmax.co2_kg
        rows.append(dict(
            sub=r["info"]["sub"], bins=r["info"]["bins"], mandatory=r["info"]["mandatory"],
            demand_t=round(r["info"]["demand_t"], 1), front_points=len(ex), proven=int(ex.exact.sum()),
            cost_min=round(cmin.cost, 1), co2_at_cost_min=round(cmin.co2_kg, 1),
            co2_min=round(cmax.co2_kg, 1), cost_at_co2_min=round(cmax.cost, 1),
            co2_range_pct=round(100 * dco2 / cmin.co2_kg, 1),
            cost_premium_pct=round(100 * (cmax.cost - cmin.cost) / cmin.cost, 1),
            abatement_usd_per_t=round((cmax.cost - cmin.cost) / (dco2 / 1000), 0) if dco2 > 1e-6 else np.nan,
            pyvrp_gap_cost_pct=round(100 * (he[he.carbon_price == 0].cost.iloc[0] - cmin.cost) / cmin.cost, 3),
            hv_ratio_pyvrp=round(hv_h / hv_e, 3) if hv_e > 0 else np.nan,
            trucks_cost_min="+".join(cmin.trucks), trucks_co2_min="+".join(cmax.trucks)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "summary.csv", index=False)
    print("\n" + t.to_string(index=False))
    return t


if __name__ == "__main__":
    inst0 = city()
    hoods = neighbourhoods(inst0)
    (OUT / "neighbourhoods.json").write_text(json.dumps(hoods))
    results = [run(i, h, inst0) for i, h in enumerate(hoods)]
    summarise(results)
    print("saved", OUT)
