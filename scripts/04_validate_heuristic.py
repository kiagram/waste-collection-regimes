"""Validate the PyVRP engine against the exact CP-SAT model (cost objective).

For each small instance: CP-SAT to proven optimality (or its best bound) vs PyVRP,
both evaluated by the independent checker. Reports the gap of PyVRP to the optimum.
usage: python scripts/04_validate_heuristic.py [n] [instances] [cpsat_tl] [pyvrp_s]
"""
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
from swc.instance import generate_synthetic
from swc.model_cpsat import DailyModel
from swc.risk import mandatory_bins, overflow_prob_within, overflow_risk, skip_penalty

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 120
hs = float(sys.argv[4]) if len(sys.argv) > 4 else 3

rows = []
for s in range(N):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=60.0)
    inst = generate_synthetic(n, seed=500 + s, params=P, bins_per_point=(4, 10),
                              fleet_counts={"new_compactor": 1, "old_compactor": 2, "small_loader": 1})
    risk = overflow_risk(inst, mode="count")
    # CFA-style penalties so that optional points matter (harder selection problem)
    pen = skip_penalty(inst, risk) + 15.0 * overflow_prob_within(inst, 48.0)
    mand = mandatory_bins(inst)
    ex = DailyModel(inst, penalty=pen, mandatory=mand).solve("cost", time_limit=tl, workers=16)
    he = solve_pyvrp_best(inst, pen, mand, starts=2, time_s=120, seed=10 * s, max_iterations=5000)
    ke = evaluate(inst, ex.routes, pen, risk, mand)
    kh = evaluate(inst, he.routes, pen, risk, mand)
    lb = ex.obj_bound / inst.params.scale.cost_scale
    rows.append(dict(seed=s, n=n, mandatory=len(mand), cpsat_status=ex.status, cpsat_cost=ke.cost_total,
                     cpsat_lb=lb, cpsat_s=ex.wall_s, pyvrp_cost=kh.cost_total, pyvrp_s=he.wall_s,
                     pyvrp_feasible=kh.feasible, gap_to_cpsat=(kh.cost_total - ke.cost_total) / ke.cost_total,
                     gap_to_lb=(kh.cost_total - lb) / kh.cost_total, cpsat_co2=ke.co2_kg, pyvrp_co2=kh.co2_kg))
    r = rows[-1]
    print(f"seed {s}: mand={len(mand):2d} CP-SAT {ex.status:8s} {ke.cost_total:8.2f} (lb {lb:8.2f}, {ex.wall_s:5.1f}s) | "
          f"PyVRP {kh.cost_total:8.2f} ({he.wall_s:4.1f}s, ok={kh.feasible}) gap={r['gap_to_cpsat']*100:+.2f}%", flush=True)

df = pd.DataFrame(rows)
out = ROOT / "results" / f"validate_pyvrp_{MODEL_VERSION}_n{n}.csv"
df.to_csv(out, index=False)
print(f"\nPyVRP vs CP-SAT: mean gap {df.gap_to_cpsat.mean()*100:+.3f}%, max {df.gap_to_cpsat.max()*100:+.3f}%, "
      f"all feasible={df.pyvrp_feasible.all()}, CP-SAT optimal {int((df.cpsat_status=='OPTIMAL').sum())}/{len(df)}; "
      f"mean time CP-SAT {df.cpsat_s.mean():.1f}s vs PyVRP {df.pyvrp_s.mean():.1f}s")
print("saved", out)
