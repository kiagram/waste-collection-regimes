"""Cost-service frontiers: sensor and no-sensor policy families at matched overflow (v4, review R1).

The risk-priced policies trace a frontier when the overflow price c_o is varied; the other families
trace one through their own parameter (period k, threshold tau). On the regime-map instances and fill
realisations (clustered layout, common random numbers), this script adds the price sweep of
risk_cfa_t5 (sensors) and blind_risk (no sensors) and combines it with the grid runs of
06_regime_map.py. For each family the lower envelope of (operating + cleanup cost, overflowing
containers) is computed, and families are compared at the overflow level of today's practice.
usage: python scripts/07_overflow_price_sweep.py [reps] [pyvrp_iters]   (after 06_regime_map.py)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION
from swc.experiments import family, horizon, regime_instance, run_or_load
from swc.simulate import BlindRisk, RiskCFA, SimConfig, draw_increments, summarise

_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
n, CV, LAYOUT = 150, 1.0, "clustered"
MULTS = [1.0, 0.5]
PRICES = [15.0, 30.0, 120.0, 240.0, 480.0]     # 60 is the base run of the regime map
REG = ROOT / "results" / f"regime_map_{MODEL_VERSION}_n{n}_r{reps}_it{iters}"
out = ROOT / "results" / f"frontiers_{MODEL_VERSION}_n{n}_cv{CV:g}_r{reps}_it{iters}"
out.mkdir(parents=True, exist_ok=True)

def scenario(mult, rep, price):
    return regime_instance(n, mult, CV, LAYOUT, rep, overflow_cost=price)


POLS = [RiskCFA(name="risk_cfa_t5", theta=5.0), BlindRisk(name="blind_risk")]
jobs = [(mult, rep, price, pol) for rep in range(reps) for mult in MULTS for price in PRICES for pol in POLS]
fname = lambda mult, rep, price, pol: out / f"days_m{mult:g}_{pol.name}_price{price:g}_rep{rep}.csv"

if SH_N > 1:
    for j, (mult, rep, price, pol) in enumerate(jobs):
        f = fname(mult, rep, price, pol)
        if j % SH_N != SH_I or f.exists():
            continue
        inst, dtf = scenario(mult, rep, price)
        burn, days = horizon(dtf)
        run_or_load(f, inst, pol, draw_increments(inst, days, seed=1000 + rep), SimConfig(days=days, pyvrp_iters=iters))
        print("done", f.name, flush=True)
    sys.exit(0)

rows = []
for mult, rep, price, pol in jobs:
    inst, dtf = scenario(mult, rep, price)
    burn, days = horizon(dtf)
    recs = run_or_load(fname(mult, rep, price, pol), inst, pol, draw_increments(inst, days, seed=1000 + rep),
                       SimConfig(days=days, pyvrp_iters=iters))
    s = summarise(recs, burn_in=burn)
    s.update(rate_mult=mult, rep=rep, policy=f"{pol.name}@{price:g}", base_policy=pol.name, price=price)
    rows.append(s)
sweep = pd.DataFrame(rows)
reg = pd.read_csv(REG / "summary_by_rep.csv")
reg = reg[(reg.cv == CV) & (reg.layout == LAYOUT) & reg.rate_mult.isin(MULTS)].copy()
reg["base_policy"], reg["price"] = reg["policy"], 60.0
pts = pd.concat([reg, sweep], ignore_index=True)
pts["family"] = pts["base_policy"].map(family)
pts["op_cost"] = pts["cost_per_day"] + pts["cleanup_cost_per_day"]
# mean over replications (common random numbers) per setting
mean = pts.groupby(["rate_mult", "family", "policy"])[["op_cost", "overflow_bins_per_day",
                                                       "overflow_weighted_per_day"]].mean().reset_index()
mean.to_csv(out / "frontier_points.csv", index=False)


def envelope(g):
    """Lower-left envelope: points not dominated in (op_cost, overflow)."""
    g = g.sort_values(["overflow_bins_per_day", "op_cost"])
    keep, best = [], np.inf
    for _, r in g.iterrows():
        if r.op_cost < best - 1e-9:
            keep.append(r)
            best = r.op_cost
    return pd.DataFrame(keep)


tab = []
for mult, g in mean.groupby("rate_mult"):
    target = float(g[g.policy == "fixed_daily"]["overflow_bins_per_day"].iloc[0])
    for fam, gf in g.groupby("family"):
        env = envelope(gf)
        ok = env[env.overflow_bins_per_day <= target + 1e-9]
        tab.append(dict(rate_mult=mult, family=fam, overflow_target=round(target, 2),
                        cost_at_target=None if ok.empty else round(float(ok.op_cost.min()), 1),
                        policy_at_target=None if ok.empty else ok.loc[ok.op_cost.idxmin(), "policy"],
                        envelope=" ".join(f"{r.policy}({r.op_cost:.0f},{r.overflow_bins_per_day:.1f})"
                                          for _, r in env.iterrows())))
t = pd.DataFrame(tab)
t.to_csv(out / "matched_overflow.csv", index=False)
pd.set_option("display.width", 250)
print(t.drop(columns="envelope").to_string(index=False))
print("saved", out)
