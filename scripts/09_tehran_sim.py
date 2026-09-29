"""Tehran District 6 (OSM case): policies x container capacity x variability, with sensor break-even.

Scenarios:
  capacity 'current' : one 1100 L bin per alley (2,852 bins)
  capacity 'double'  : two bins per alley (same waste, half the fill rate per bin, 2x sensors)
usage: python scripts/09_tehran_sim.py [reps] [days] [pyvrp_iters]
"""
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from swc.config import Params
from swc.simulate import (DayRecord, FixedDaily, RiskCFA, RiskMyopic, SimConfig, Threshold, draw_increments,
                          simulate, summarise)
from swc.tehran import INSTANCE_VERSION, load

# optional sharding for parallel runs: SHARD=i/N runs every N-th job; an unsharded run afterwards
# reuses the saved day files and writes the summary
import os
_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)
_job = -1

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
days = int(sys.argv[2]) if len(sys.argv) > 2 else 20
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
AREA_KM2 = 21.4
policies = [FixedDaily(), Threshold(tau=0.70), RiskMyopic(), RiskCFA(name="risk_cfa_t5", theta=5.0)]
cfg = SimConfig(days=days, engine="pyvrp", pyvrp_s=300, pyvrp_iters=iters)
out = ROOT / "results" / f"tehran_d6_{INSTANCE_VERSION}_r{reps}_d{days}_it{iters}"
out.mkdir(parents=True, exist_ok=True)

rows = []
for cv in (1.0, 0.35):
    for cap in ("current", "double"):
        for rep in range(reps):
            P = Params()
            P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=cv)
            inst = load(P, rate_mult=1.0 if cap == "current" else 0.5)
            if cap == "double":
                inst.bins_per_point = inst.bins_per_point * 2
            rng = np.random.default_rng(50 + rep)            # replicate-specific initial state
            inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
            inc = draw_increments(inst, days, seed=2000 + rep)
            for pol in policies:
                f = out / f"days_cv{cv:g}_{cap}_{pol.name}_rep{rep}.csv"
                _job += 1
                if SH_N > 1:
                    if _job % SH_N == SH_I and not f.exists():
                        pd.DataFrame([asdict(r) for r in simulate(inst, pol, inc, cfg)]).to_csv(f, index=False)
                        print("done", f.name, flush=True)
                    continue
                t0 = time.perf_counter()
                if f.exists():
                    d = pd.read_csv(f)
                    recs = [DayRecord(**{k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in d.to_dict("records")]
                else:
                    recs = simulate(inst, pol, inc, cfg)
                    pd.DataFrame([asdict(r) for r in recs]).to_csv(f, index=False)
                s = summarise(recs, burn_in=3)
                s.update(cv=cv, capacity=cap, bins=int(inst.bins_per_point.sum()), policy=pol.name, rep=rep,
                         sim_s=time.perf_counter() - t0)
                rows.append(s)
                print(f"cv={cv:<4} cap={cap:8s} rep={rep} {pol.name:12s} cost={s['cost_per_day']:7.0f} "
                      f"CO2={s['co2_kg_per_day']:6.0f} overflow={s['overflow_bins_per_day']:6.1f} "
                      f"trucks={s['truck_shifts_per_day']:5.1f} served={s['points_served_per_day']:5.1f} "
                      f"({s['sim_s']:.0f}s)", flush=True)

if SH_N > 1:
    sys.exit(0)
df = pd.DataFrame(rows)
df.to_csv(out / "summary_by_rep.csv", index=False)

from importlib import import_module
be = import_module("08_break_even")  # reuse the same sensor-cost assumptions
agg = []
for (cv, cap), g in df.groupby(["cv", "capacity"], sort=False):
    base = g[g.policy == "fixed_daily"].set_index("rep")
    sensor_day = be.sensor_cost_per_day(int(g.bins.iloc[0]), AREA_KM2)
    for pol, x in g.groupby("policy", sort=False):
        x = x.set_index("rep")
        sav = base.cost_per_day - x.cost_per_day
        agg.append(dict(cv=cv, capacity=cap, policy=pol, cost=round(x.cost_per_day.mean()),
                        d_cost_pct=round(((x.cost_per_day / base.cost_per_day - 1) * 100).mean(), 1),
                        co2_kg=round(x.co2_kg_per_day.mean()), overflow=round(x.overflow_bins_per_day.mean(), 1),
                        trucks=round(x.truck_shifts_per_day.mean(), 1),
                        net_vs_fixed_after_sensors=None if pol == "fixed_daily" else round((sav - sensor_day).mean())))
t = pd.DataFrame(agg)
print("\n", t.to_string(index=False))
t.to_csv(out / "summary.csv", index=False)
print("saved", out)
