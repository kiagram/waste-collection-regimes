"""Regime map: when does sensor-based collection pay off? (v4)

Varies the fill speed (days to full ≈ 3.2 / rate multiplier), the day-to-day variability (CV) and
the district geometry (clustered and uniform layouts, a new geometry per replication), and compares
every policy family of the tuning grid (swc.experiments.policy_grid) on common random numbers.
The fill-speed grid is dense around the crossover found in v3 (3-8 days to full). Families are
tuned leave-one-replication-out on the social cost; the value of sensing is the saving of the best
sensor policy over the best policy without sensors, net of the sensor-network cost.
Horizon: burn-in of one fill cycle, then at least three fill cycles (min 20 days).
Resumable; SHARD=i/N runs every N-th job, an unsharded run afterwards writes the summaries.
usage: python scripts/06_regime_map.py [n_points] [reps] [pyvrp_iters] [extra_cells_csv]
"""
import os
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION
from swc.experiments import (horizon, loo_tuned, policy_grid, regime_instance, run_or_load, social_cost,
                             tuned_table)
from swc.simulate import SimConfig, draw_increments, summarise

_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
# optional: replications beyond BASE_REPS only in the listed cells (rate_mult, cv, layout), e.g. the cells
# where 3 replications left the sensor decision uncertain (results/regime_map_v4_extra_cells.csv)
EXTRA = sys.argv[4] if len(sys.argv) > 4 else None
BASE_REPS = 3
extra_cells = None if EXTRA is None else {
    (float(r.rate_mult), float(r.cv), r.layout) for r in pd.read_csv(ROOT / EXTRA).itertuples()}

RATE_MULT = [1.0, 0.8, 0.65, 0.5, 0.4, 0.25]   # ~3.2, 4, 5, 6.5, 8 and 13 days to full
CVS = [1.0, 0.35]
LAYOUTS = ["clustered", "uniform"]
OVERFLOW_COST = 60.0
AREA_KM2 = (5.0 * (n / 40) ** 0.5 / 1.6) ** 2
out_dir = ROOT / "results" / f"regime_map_{MODEL_VERSION}_n{n}_r{reps}_it{iters}"
out_dir.mkdir(parents=True, exist_ok=True)


def scenario(mult, cv, layout, rep):
    return regime_instance(n, mult, cv, layout, rep, OVERFLOW_COST)


def fname(mult, cv, layout, pol, rep):
    return out_dir / f"days_m{mult:g}_cv{cv:g}_{layout}_{pol.name}_rep{rep}.csv"


jobs = []
for rep in range(reps):
    for layout in LAYOUTS:
        for cv in CVS:
            for mult in RATE_MULT:
                if extra_cells is not None and rep >= BASE_REPS and (mult, cv, layout) not in extra_cells:
                    continue
                inst, dtf = scenario(mult, cv, layout, rep)
                for pol in policy_grid(dtf):
                    jobs.append((mult, cv, layout, rep, pol))

if SH_N > 1:
    for j, (mult, cv, layout, rep, pol) in enumerate(jobs):
        f = fname(mult, cv, layout, pol, rep)
        if j % SH_N != SH_I or f.exists():
            continue
        inst, dtf = scenario(mult, cv, layout, rep)
        burn, days = horizon(dtf)
        run_or_load(f, inst, pol, draw_increments(inst, days, seed=1000 + rep), SimConfig(days=days, pyvrp_iters=iters))
        print("done", f.name, flush=True)
    sys.exit(0)

be = import_module("08_break_even")
rows = []
for mult, cv, layout, rep, pol in jobs:
    inst, dtf = scenario(mult, cv, layout, rep)
    burn, days = horizon(dtf)
    recs = run_or_load(fname(mult, cv, layout, pol, rep), inst, pol, draw_increments(inst, days, seed=1000 + rep),
                       SimConfig(days=days, pyvrp_iters=iters))
    s = summarise(recs, burn_in=burn)
    bins = int(inst.bins_per_point.sum())
    s.update(rate_mult=mult, cv=cv, layout=layout, days_to_full=dtf, burn_in=burn, policy=pol.name, rep=rep,
             bins=bins, sensor_cost_day=be.sensor_cost_per_day(bins, AREA_KM2))
    s["social_cost"] = social_cost(s, OVERFLOW_COST)
    rows.append(s)
df = pd.DataFrame(rows)
df.to_csv(out_dir / "summary_by_rep.csv", index=False)
keys = ["rate_mult", "cv", "layout"]
tuned = loo_tuned(df, keys)
tuned.to_csv(out_dir / "tuned_by_rep.csv", index=False)
t = tuned_table(tuned, keys)
t.to_csv(out_dir / "summary.csv", index=False)
pd.set_option("display.width", 250)
print(t[t.family.isin(["best_no_sensor", "best_sensor", "fixed_daily"])].round(1).to_string(index=False))
print("saved", out_dir)
