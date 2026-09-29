"""Policy comparison by rolling-horizon simulation (common random numbers).

usage: python scripts/03_policy_sim.py [n_points] [reps] [days] [time_limit_s] [overflow_cost] [engine] [daily_cv] [pyvrp_iters]
"""
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION, Params
from swc.instance import generate_synthetic
from swc.simulate import (DayRecord, FixedAlternate, FixedDaily, RiskCFA, RiskMyopic, SimConfig,
                          Threshold, draw_increments, simulate, summarise)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
days = int(sys.argv[3]) if len(sys.argv) > 3 else 12
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
ovf = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0

policies = [FixedDaily(), FixedAlternate(), Threshold(tau=0.70), RiskMyopic(),
            RiskCFA(name="risk_cfa_t5", theta=5.0), RiskCFA(name="risk_cfa_t15", theta=15.0)]
engine = sys.argv[6] if len(sys.argv) > 6 else "pyvrp"
cv = float(sys.argv[7]) if len(sys.argv) > 7 else 0.35
iters = int(sys.argv[8]) if len(sys.argv) > 8 else 1500
cfg = SimConfig(days=days, time_limit=tl, workers=16, engine=engine, pyvrp_s=max(tl, 120), pyvrp_iters=iters)
tag = f"{engine}_n{n}_r{reps}_d{days}_it{iters}_ovf{ovf:g}_cv{cv:g}"
out_dir = ROOT / "results" / f"policy_sim_{MODEL_VERSION}_{tag}"
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for rep in range(reps):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=ovf, fill_rate_cv=cv)
    # fleet scaled with the instance: roughly one truck-shift per 20 points on a full collection day
    fleet = {"new_compactor": max(1, n // 60), "old_compactor": max(2, n // 30), "small_loader": max(1, n // 60)}
    inst = generate_synthetic(n, seed=100 + rep, params=P, bins_per_point=(4, 10), fleet_counts=fleet,
                              area_m=5000.0 if n <= 40 else 5000.0 * (n / 40) ** 0.5 / 1.6)
    inc = draw_increments(inst, days, seed=1000 + rep)
    for pol in policies:
        f = out_dir / f"days_{pol.name}_rep{rep}.csv"
        t0 = time.perf_counter()
        if f.exists():  # resumable: reuse finished runs
            d = pd.read_csv(f)
            recs = [DayRecord(**{k: (None if pd.isna(v) else v) for k, v in row.items()}) for row in d.to_dict("records")]
        else:
            recs = simulate(inst, pol, inc, cfg)
            pd.DataFrame([asdict(r) for r in recs]).to_csv(f, index=False)
        s = summarise(recs, burn_in=2)
        s.update(policy=pol.name, rep=rep, instance=inst.name, sim_s=time.perf_counter() - t0,
                 max_gap=max((r.gap or 0.0) for r in recs[2:]))
        rows.append(s)
        print(f"rep {rep} {pol.name:16s} cost/day={s['cost_per_day']:7.1f} CO2/day={s['co2_kg_per_day']:6.1f} "
              f"overflow bins/day={s['overflow_bins_per_day']:5.2f} weighted={s['overflow_weighted_per_day']:5.2f} "
              f"trucks={s['truck_shifts_per_day']:.2f} relaxed={s['relaxed_days']} nonopt={s['non_optimal_days']} "
              f"maxgap={s['max_gap']:.3f} "
              f"({s['sim_s']:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out_dir / "summary_by_rep.csv", index=False)
metrics = ["cost_per_day", "co2_kg_per_day", "km_per_day", "truck_shifts_per_day", "points_served_per_day",
           "overflow_bins_per_day", "overflow_weighted_per_day", "kg_per_km", "relaxed_days", "non_optimal_days",
           "max_gap"]
agg = df.groupby("policy", sort=False)[metrics].agg(["mean", "std"])
print("\n", agg.round(2).to_string())
agg.to_csv(out_dir / "summary.csv")
(out_dir / "config.json").write_text(json.dumps({"n": n, "reps": reps, "days": days, "time_limit": tl,
                                                  "overflow_cost": ovf, "daily_cv": cv, "pyvrp_iters": iters, "policies": [p.name for p in policies]}, indent=1))
print("saved", out_dir)
