"""Cost-CO2 trade-off across fleet compositions on a typical Tehran District 6 night.

Every point is collected (fixed-daily service) to isolate the fleet effect. For each fleet mix,
a carbon price lambda (USD/t CO2, applied to empty-truck emissions per km inside PyVRP) is swept.
True operating cost (without the carbon charge) and true load-dependent CO2 are recomputed by
the checker. Output: (cost, CO2) points per fleet and the abatement cost of fleet renewal.
usage: python scripts/10_fleet_tradeoff.py [seeds] [pyvrp_iters]
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import Params
from swc.evaluate import evaluate
from swc.heuristic_pyvrp import solve_pyvrp
from swc.tehran import INSTANCE_VERSION, load

seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
FLEETS = {  # similar total capacity per shift (~360-370 t)
    "current_ageing": {"new_compactor": 8, "old_compactor": 18, "small_loader": 8},
    "half_renewed": {"new_compactor": 17, "old_compactor": 8, "small_loader": 8},
    "all_new": {"new_compactor": 24, "old_compactor": 0, "small_loader": 8},
    "all_old": {"new_compactor": 0, "old_compactor": 28, "small_loader": 8},
}
PRICES = [0, 25, 50, 100, 200, 400]
rows = []
for fname, counts in FLEETS.items():
    counts = {k: v for k, v in counts.items() if v > 0}
    P = Params()
    P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=1.0)
    inst = load(P, fleet_counts=counts)
    inst.fill = np.clip(inst.rate_mean_per_h * 24, 0, 1)      # one day since the last collection
    mand = set(inst.bins)
    zero = np.zeros(inst.n)
    for lam in PRICES:
        for s in range(seeds):
            sol = solve_pyvrp(inst, zero, mand, time_s=600, seed=s, max_iterations=iters, carbon_price=lam)
            k = evaluate(inst, sol.routes, zero, mandatory=mand)
            used = {}
            for t, r in enumerate(sol.routes):
                if r:
                    used[inst.vehicles[t].name] = used.get(inst.vehicles[t].name, 0) + 1
            rows.append(dict(fleet=fname, carbon_price=lam, seed=s, feasible=k.feasible, cost=k.cost_total,
                             co2_kg=k.co2_kg, km=k.km, trucks=k.trucks_used, **{f"used_{a}": b for a, b in used.items()}))
            print(f"{fname:15s} lambda={lam:<4} seed={s} cost={k.cost_total:7.0f} CO2={k.co2_kg:6.0f} kg "
                  f"km={k.km:6.0f} used={used} ok={k.feasible}", flush=True)

df = pd.DataFrame(rows).fillna(0)
out = ROOT / "results" / f"fleet_tradeoff_tehran_{INSTANCE_VERSION}"
out.mkdir(exist_ok=True)
df.to_csv(out / "runs.csv", index=False)
best = df.loc[df.groupby(["fleet", "carbon_price"])["cost"].idxmin()]  # best of seeds per setting
summ = best.groupby(["fleet", "carbon_price"])[["cost", "co2_kg", "km", "trucks"]].first().round(1)
print("\n", summ.to_string())
base = summ.loc[("current_ageing", 0)]
print("\nAbatement cost of fleet change vs current fleet at lambda=0 (USD per t CO2 avoided, per night):")
for f in FLEETS:
    r = summ.loc[(f, 0)]
    dco2 = (base.co2_kg - r.co2_kg) / 1000
    dc = r.cost - base.cost
    print(f"  {f:15s} dCost={dc:+7.0f} USD/night  dCO2={-dco2 * 1000:+6.0f} kg/night  "
          f"abatement={'n/a' if abs(dco2) < 1e-6 else f'{dc / dco2:,.0f} USD/t'}  "
          f"annual: {dc * 300:+,.0f} USD, {-dco2 * 300:+,.0f} t CO2")
summ.to_csv(out / "summary.csv")
