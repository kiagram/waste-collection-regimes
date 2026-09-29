"""Pareto demo on a realistic collection-point instance (multi-trip, heterogeneous fleet)."""
import sys, json
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swc.config import Params
from swc.instance import generate_synthetic
from swc.model_cpsat import DailyModel
from swc.pareto import epsilon_front, hypervolume_2d
from swc.risk import overflow_risk, skip_penalty, mandatory_bins
from swc.evaluate import evaluate

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ovf = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
P = Params(); P.risk = replace(P.risk, overflow_cost=ovf)
inst = generate_synthetic(n, seed=seed, params=P, bins_per_point=(4, 10),
                          fleet_counts={"new_compactor": 1, "old_compactor": 2, "small_loader": 1})
risk = overflow_risk(inst, mode="count"); pen = skip_penalty(inst, risk); mand = mandatory_bins(inst)
tot_kg = inst.demand_kg().sum()
print(f"{inst.name}: points={n}, bins={inst.bins_per_point.sum()}, waste if all collected={tot_kg/1000:.1f} t, "
      f"mandatory={len(mand)}, penalty if nothing collected={pen.sum():.0f}")
m = DailyModel(inst, penalty=pen, mandatory=mand)
front = epsilon_front(m, grid=8, time_limit=tl, workers=16)
rows = []
for p in front:
    k = evaluate(inst, p.solution.routes, pen, risk, mand)
    trucks = [inst.vehicles[i].name for i, r in enumerate(p.solution.routes) if r]
    trips = [r.count(inst.disposal) for r in p.solution.routes if r]
    rows.append(dict(cost=k.cost_total, co2=k.co2_kg, exact=p.exact, **{kk: getattr(k, kk) for kk in
                ("cost_fixed", "cost_variable", "cost_penalty", "km", "bins_served", "kg_collected", "feasible")},
                trucks=trucks, trips=trips))
    print(f"cost={k.cost_total:7.1f} [fix {k.cost_fixed:4.0f} var {k.cost_variable:5.1f} pen {k.cost_penalty:5.1f}] "
          f"CO2={k.co2_kg:6.1f} kg km={k.km:5.1f} served={k.bins_served}/{n} {k.kg_collected/1000:.1f}t "
          f"trucks={trucks} trips={trips} exact={p.exact} ok={k.feasible}")
out = ROOT / "results" / f"pareto_{inst.name}_ovf{ovf:g}.json"
out.write_text(json.dumps(rows, indent=1, default=str)); print("saved", out.name)
