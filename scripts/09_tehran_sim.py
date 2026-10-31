"""Tehran District 6 (OSM case): policy families x container capacity x variability (v4).

Scenarios:
  capacity 'current' : one 1100 L bin per alley (2,852 bins)
  capacity 'double'  : two bins per alley (same waste, half the fill rate per bin, 2x sensors)
Every policy of the tuning grid (swc.experiments.policy_grid) is simulated on common random numbers;
families are tuned leave-one-replication-out on the social cost (operating + cleanup + valued
overflow), and the value of sensing is measured against the best policy without sensors.
Horizon per scenario: burn-in of one fill cycle, then at least three fill cycles (min 20 days).
Resumable; SHARD=i/N runs every N-th job, an unsharded run afterwards writes the summaries.
usage: python scripts/09_tehran_sim.py [reps] [pyvrp_iters]
"""
import os
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from swc.config import Params
from swc.experiments import horizon, loo_tuned, policy_grid, run_or_load, social_cost, tuned_table
from swc.simulate import SimConfig, draw_increments, summarise
from swc.tehran import INSTANCE_VERSION, load

_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
AREA_KM2 = 21.4
OVERFLOW_COST = 60.0
out = ROOT / "results" / f"tehran_d6_{INSTANCE_VERSION}_r{reps}_it{iters}"
out.mkdir(parents=True, exist_ok=True)


def scenario(cv, cap, rep):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=OVERFLOW_COST, fill_rate_cv=cv)
    inst = load(P, rate_mult=1.0 if cap == "current" else 0.5)
    if cap == "double":
        inst.bins_per_point = inst.bins_per_point * 2
    rng = np.random.default_rng(50 + rep)            # replicate-specific initial state
    inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
    dtf = 1.0 / float(np.mean(inst.rate_mean_per_h * 24))
    return inst, dtf


jobs = []
for rep in range(reps):                               # replications outermost: early reps finish first
    for cv in (1.0, 0.35):
        for cap in ("current", "double"):
            inst, dtf = scenario(cv, cap, rep)
            for pol in policy_grid(dtf):
                jobs.append((cv, cap, rep, pol))

if SH_N > 1:
    for j, (cv, cap, rep, pol) in enumerate(jobs):
        if j % SH_N != SH_I:
            continue
        f = out / f"days_cv{cv:g}_{cap}_{pol.name}_rep{rep}.csv"
        if f.exists():
            continue
        inst, dtf = scenario(cv, cap, rep)
        burn, days = horizon(dtf)
        run_or_load(f, inst, pol, draw_increments(inst, days, seed=2000 + rep), SimConfig(days=days, pyvrp_s=300,
                                                                                          pyvrp_iters=iters))
        print("done", f.name, flush=True)
    sys.exit(0)

be = import_module("08_break_even")
rows = []
for cv, cap, rep, pol in jobs:
    inst, dtf = scenario(cv, cap, rep)
    burn, days = horizon(dtf)
    f = out / f"days_cv{cv:g}_{cap}_{pol.name}_rep{rep}.csv"
    recs = run_or_load(f, inst, pol, draw_increments(inst, days, seed=2000 + rep),
                       SimConfig(days=days, pyvrp_s=300, pyvrp_iters=iters))
    s = summarise(recs, burn_in=burn)
    bins = int(inst.bins_per_point.sum())
    s.update(cv=cv, capacity=cap, bins=bins, days_to_full=dtf, burn_in=burn, policy=pol.name, rep=rep,
             sensor_cost_day=be.sensor_cost_per_day(bins, AREA_KM2))
    s["social_cost"] = social_cost(s, OVERFLOW_COST)
    rows.append(s)
df = pd.DataFrame(rows)
df.to_csv(out / "summary_by_rep.csv", index=False)
tuned = loo_tuned(df, ["cv", "capacity"])
tuned.to_csv(out / "tuned_by_rep.csv", index=False)
t = tuned_table(tuned, ["cv", "capacity"])
pd.set_option("display.width", 250)
print(t.round(1).to_string(index=False))
t.to_csv(out / "summary.csv", index=False)
# capacity effect at equal policy family: double vs current, paired by replication
cap_rows = []
for (cv, fam), g in tuned.groupby(["cv", "tuned_family"]):
    a = g[g.capacity == "current"].set_index("rep")
    b = g[g.capacity == "double"].set_index("rep").loc[a.index]
    cap_rows.append(dict(cv=cv, family=fam, d_cost_double=(b.cost_per_day - a.cost_per_day).mean(),
                         d_social_double=(b.social_cost - a.social_cost).mean(),
                         d_overflow_double=(b.overflow_bins_per_day - a.overflow_bins_per_day).mean(),
                         d_overflow_vol_double=(b.overflow_volume_bins_per_day - a.overflow_volume_bins_per_day).mean()))
pd.DataFrame(cap_rows).to_csv(out / "capacity_effect.csv", index=False)
print("saved", out)
