"""Regime map: when does sensor-based (risk) collection pay off?

Varies the fill-rate level (days-to-full ≈ 1 / daily fill fraction) and the day-to-day
variability (CV), and compares fixed-daily, threshold-70% and risk-based collection on the
same district-scale instance family (common random numbers, iteration-based PyVRP).
Resumable: finished runs are reused.

usage: python scripts/06_regime_map.py [n_points] [reps] [days] [pyvrp_iters]
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

# optional sharding for parallel runs: SHARD=i/N runs every N-th job; an unsharded run afterwards
# reuses the saved day files and writes the summary
import os
_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)
_job = -1
from swc.instance import generate_synthetic
from swc.simulate import (DayRecord, FixedDaily, RiskMyopic, SimConfig, Threshold, draw_increments,
                          simulate, summarise)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
days = int(sys.argv[3]) if len(sys.argv) > 3 else 20
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

RATE_MULT = [0.33, 0.5, 1.0]      # baseline ≈ 0.31 bin/day -> ~3 days to full; 0.33x -> ~10 days
CVS = [0.35, 1.0]
policies = [FixedDaily(), Threshold(tau=0.70), RiskMyopic()]
cfg = SimConfig(days=days, engine="pyvrp", pyvrp_s=120, pyvrp_iters=iters)
out_dir = ROOT / "results" / f"regime_map_{MODEL_VERSION}_n{n}_r{reps}_d{days}_it{iters}"
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for mult in RATE_MULT:
    for cv in CVS:
        for rep in range(reps):
            P = Params()
            P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=cv)
            fleet = {"new_compactor": max(1, n // 60), "old_compactor": max(2, n // 30), "small_loader": max(1, n // 60)}
            inst = generate_synthetic(n, seed=100 + rep, params=P, bins_per_point=(4, 10), fleet_counts=fleet,
                                      area_m=5000.0 * (n / 40) ** 0.5 / 1.6)
            inst.rate_mean_per_h = inst.rate_mean_per_h * mult
            inst.rate_sd_per_h = inst.rate_sd_per_h * mult
            inc = draw_increments(inst, days, seed=1000 + rep)
            days_to_full = 1.0 / float(np.mean(inst.rate_mean_per_h * 24))
            for pol in policies:
                f = out_dir / f"days_m{mult:g}_cv{cv:g}_{pol.name}_rep{rep}.csv"
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
                s.update(rate_mult=mult, cv=cv, days_to_full=days_to_full, policy=pol.name, rep=rep,
                         sim_s=time.perf_counter() - t0)
                rows.append(s)
                print(f"mult={mult:<4} cv={cv:<4} rep={rep} {pol.name:12s} cost={s['cost_per_day']:7.1f} "
                      f"overflow={s['overflow_bins_per_day']:6.1f} served={s['points_served_per_day']:5.1f} "
                      f"({s['sim_s']:.0f}s)", flush=True)

if SH_N > 1:
    sys.exit(0)
df = pd.DataFrame(rows)
df.to_csv(out_dir / "summary_by_rep.csv", index=False)
(out_dir / "config.json").write_text(json.dumps({"n": n, "reps": reps, "days": days, "iters": iters,
                                                  "rate_mult": RATE_MULT, "cvs": CVS}, indent=1))
print("saved", out_dir)
