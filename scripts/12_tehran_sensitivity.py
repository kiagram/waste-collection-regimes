"""Robustness of the Tehran District 6 conclusions (CV = 1.0, calibrated variability).

Scenarios, each compared with the base run (09_tehran_sim.py) on common random-number seeds:
  gen0.7 / gen1.3   waste generation x0.7 / x1.3 (population or per-capita uncertainty, +/-30%)
                    -> shifts days-to-full; capacity current and double
  leg3 / leg12      transfer-station distance 3 km / 12 km (base 6 km); current capacity
  alpha_tight/loose zone tolerances (0.02-0.25) / (0.10-0.70) vs base (0.05-0.50); double capacity
Resumable. usage: python scripts/12_tehran_sensitivity.py [reps] [days] [pyvrp_iters]
"""
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import Params
from swc.simulate import (DayRecord, FixedDaily, RiskCFA, SimConfig, Threshold, draw_increments, simulate,
                          summarise)
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
cfg = SimConfig(days=days, engine="pyvrp", pyvrp_s=300, pyvrp_iters=iters)
out = ROOT / "results" / f"tehran_sensitivity_{INSTANCE_VERSION}_r{reps}_d{days}_it{iters}"
out.mkdir(parents=True, exist_ok=True)
ALL = [FixedDaily(), Threshold(tau=0.70), RiskCFA(name="risk_cfa_t5", theta=5.0)]
RISK = [RiskCFA(name="risk_cfa_t5", theta=5.0)]

SCEN = []
for g in (0.7, 1.3):
    for cap in ("current", "double"):
        SCEN.append(dict(name=f"gen{g:g}", cap=cap, gen=g, leg=None, alpha=None, policies=ALL))
for leg in (3.0, 12.0):
    SCEN.append(dict(name=f"leg{leg:g}", cap="current", gen=1.0, leg=leg, alpha=None, policies=ALL))
for nm, a in (("alpha_tight", (0.02, 0.25)), ("alpha_loose", (0.10, 0.70))):
    SCEN.append(dict(name=nm, cap="double", gen=1.0, leg=None, alpha=a, policies=RISK))

rows = []
for sc in SCEN:
    for rep in range(reps):
        P = Params()
        risk = dict(overflow_cost=60.0, fill_rate_cv=1.0)
        if sc["alpha"]:
            risk.update(alpha_min=sc["alpha"][0], alpha_max=sc["alpha"][1])
        P.risk = replace(P.risk, **risk)
        mult = sc["gen"] * (0.5 if sc["cap"] == "double" else 1.0)
        inst = load(P, rate_mult=mult, transfer_leg_km=sc["leg"])
        if sc["cap"] == "double":
            inst.bins_per_point = inst.bins_per_point * 2
        rng = np.random.default_rng(50 + rep)            # same seeds as 09_tehran_sim.py
        inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
        inc = draw_increments(inst, days, seed=2000 + rep)
        for pol in sc["policies"]:
            f = out / f"days_{sc['name']}_{sc['cap']}_{pol.name}_rep{rep}.csv"
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
            s.update(scenario=sc["name"], capacity=sc["cap"], gen=sc["gen"], leg=sc["leg"] or 6.0,
                     bins=int(inst.bins_per_point.sum()), policy=pol.name, rep=rep,
                     days_to_full=float(1 / np.mean(inst.rate_mean_per_h * 24)), sim_s=time.perf_counter() - t0)
            rows.append(s)
            print(f"{sc['name']:12s} {sc['cap']:8s} rep={rep} {pol.name:12s} cost={s['cost_per_day']:7.0f} "
                  f"overflow_m3={1.1 * s['overflow_volume_bins_per_day']:6.1f} trucks={s['truck_shifts_per_day']:5.1f} "
                  f"({s['sim_s']:.0f}s)", flush=True)
if SH_N > 1:
    sys.exit(0)
pd.DataFrame(rows).to_csv(out / "summary_by_rep.csv", index=False)
print("saved", out)
