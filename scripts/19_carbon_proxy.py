"""Does pricing carbon on empty-truck emissions (the proxy PyVRP can optimise) approximate pricing the
true load-dependent CO2? (review point R9)

On the Tehran sub-districts of 16_exact_fronts_tehran.py (full service: every point collected, which
isolates the fleet and sequencing effect), for each carbon price lambda (USD/t):
  true   CP-SAT minimum of cost + lambda x load-dependent CO2 (exact objective)
  proxy  CP-SAT minimum of cost + lambda x empty-truck CO2 (per-km costs raised by lambda x a_k),
         i.e. the proxy optimised exactly
  pyvrp  the district-scale engine with the same proxy (best of 2 x 5,000 iterations)
All plans are scored by the independent checker on the true objective. proxy - true is the
approximation error of the proxy; pyvrp - proxy is the search error of the engine.
usage: python scripts/19_carbon_proxy.py [S] [n] [time_limit_s] [workers]
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION, Params
from swc.evaluate import evaluate
from swc.heuristic_pyvrp import solve_pyvrp_best
from swc.model_cpsat import DailyModel
from swc.tehran import TEHRAN_FLEET, load

S = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TL = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LAMBDAS = [0, 50, 100, 200, 400, 800, 2000]
FLEET = ["new_compactor", "old_compactor", "old_compactor", "small_loader"]
OUT = ROOT / "results" / f"carbon_proxy_{MODEL_VERSION}_s{S}_n{N}_tl{TL:g}"
OUT.mkdir(parents=True, exist_ok=True)


def city():  # same night as 16_exact_fronts_tehran.py
    P = Params()
    P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=1.0)
    inst = load(P)
    rng = np.random.default_rng(50)
    inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
    return inst


def neighbourhoods(inst):  # same sub-districts as 16_exact_fronts_tehran.py
    pts = np.array(list(inst.bins))
    xy = inst.xy_m[pts]
    d = np.minimum(inst.dist_m, inst.dist_m.T)
    seeds = [int(pts[np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1))])]
    while len(seeds) < S:
        dmin = np.min(np.linalg.norm(xy[:, None, :] - inst.xy_m[seeds][None, :, :], axis=2), axis=1)
        seeds.append(int(pts[np.argmax(dmin)]))
    return [sorted(int(j) for j in pts[np.argsort(d[s, pts])][:N]) for s in seeds]


def proxy_instance(inst, lam):
    """Copy of `inst` whose per-km costs include lambda x empty-truck emissions (same truck identities)."""
    mapped = {}
    for v in inst.vehicles:
        if id(v) not in mapped:
            mapped[id(v)] = replace(v, var_cost_per_km=v.var_cost_per_km + lam * v.a_g_per_m / 1000.0)
    px = inst.subset(list(inst.bins), name=inst.name + f"_proxy{lam}")
    px.vehicles = [mapped[id(v)] for v in inst.vehicles]
    return px


def score(inst, routes, lam, mand, **extra):
    k = evaluate(inst, routes, np.zeros(inst.n), mandatory=mand)
    return dict(lam=lam, feasible=k.feasible, cost=k.cost_total, co2_kg=k.co2_kg, km=k.km,
                true_priced=k.cost_total + lam * k.co2_kg / 1000, trucks=[inst.vehicles[t].name for t, r in
                                                                          enumerate(routes) if r], **extra)


rows = []
inst0 = city()
for idx, members in enumerate(neighbourhoods(inst0)):
    f = OUT / f"sub{idx}.json"
    if f.exists():
        rows += json.loads(f.read_text())
        continue
    inst = inst0.subset(members, name=f"tehran_sub{idx}")
    inst.vehicles = [TEHRAN_FLEET[v] for v in FLEET]
    mand = set(inst.bins)
    zero = np.zeros(inst.n)
    true_model = DailyModel(inst, penalty=zero, mandatory=mand)
    res = []
    for lam in LAMBDAS:
        t = true_model.solve("priced", carbon_price=lam, time_limit=TL, workers=WORKERS)
        res.append(score(inst, t.routes, lam, mand, sub=idx, method="true", status=t.status))
        px = proxy_instance(inst, lam)
        p = DailyModel(px, penalty=zero, mandatory=mand).solve("cost", time_limit=TL, workers=WORKERS)
        res.append(score(inst, p.routes, lam, mand, sub=idx, method="proxy", status=p.status))
        h = solve_pyvrp_best(inst, zero, mand, starts=2, time_s=120, seed=0, max_iterations=5000, carbon_price=lam)
        res.append(score(inst, h.routes, lam, mand, sub=idx, method="pyvrp", status=h.status))
        print(" ".join(f"{r['method']}={r['true_priced']:.2f}({r['status']})" for r in res[-3:]), f"lambda={lam}",
              f"sub={idx}", flush=True)
    f.write_text(json.dumps(res, indent=1, default=str))
    rows += res

df = pd.DataFrame(rows)
df.to_csv(OUT / "runs.csv", index=False)
w = df.pivot_table(index=["sub", "lam"], columns="method", values=["true_priced", "co2_kg", "cost"])
gap = pd.DataFrame({
    "proxy_vs_true_pct": 100 * (w["true_priced"]["proxy"] / w["true_priced"]["true"] - 1),
    "pyvrp_vs_proxy_pct": 100 * (w["true_priced"]["pyvrp"] / w["true_priced"]["proxy"] - 1),
    "pyvrp_vs_true_pct": 100 * (w["true_priced"]["pyvrp"] / w["true_priced"]["true"] - 1),
    "co2_true": w["co2_kg"]["true"], "co2_proxy": w["co2_kg"]["proxy"], "co2_pyvrp": w["co2_kg"]["pyvrp"],
})
summ = gap.groupby("lam").agg(["mean", "max"]).round(2)
print(summ.to_string())
summ.to_csv(OUT / "summary.csv")
print("saved", OUT)
