"""Robustness of the Tehran District 6 conclusions (v4, CV = 1.0, calibrated variability).

Every scenario is re-optimised (simulated with its own parameters), on the common random numbers
of the base run (09_tehran_sim.py), with the policies tuned there: per family and capacity the most
frequently chosen member, plus today's practice (fixed_daily), the threshold rule used in practice
(threshold70) and the paper's policy (risk_cfa_t5).
  gen0.7 / gen1.3      waste generation x0.7 / x1.3 (population or per-capita uncertainty)
  leg3 / leg12         transfer-station distance 3 / 12 km (base 6 km); current capacity
  fixed0.5 / fixed1.5  truck fixed cost per shift x0.5 / x1.5 (R11: re-optimised, not re-costed)
  km0.5 / km1.5        truck cost per km x0.5 / x1.5
  spill_lost           spilled waste leaves the system (the accounting of the first draft)
  split6               collection points split into sub-points of at most 6 containers that fill
                       independently, 100 m apart (aggregation resolution, within-point dependence)
  targeted             second container only at the faster-filling half of the alleys
  large                one 2.2 m3 container per alley instead of two 1.1 m3 containers
  alpha_tight / loose  zone tolerances 0.02-0.25 / 0.10-0.70 (base 0.05-0.50)      [decisions only]
  normal               Normal instead of Gamma tails in the decision model (misspecification)  [decisions only]
  draft_prize          penalty on P_skip instead of P_skip - P_serve (first draft)     [decisions only]
'Decisions only' scenarios change what the risk-priced policies (and BlindRisk) decide but not the
simulated world, so the other policies are taken from the base run.
Container, sensor and cleanup prices do not change any decision and are varied ex post (13_cost_robustness.py).
Resumable; SHARD=i/N as in 09. usage: python scripts/12_tehran_sensitivity.py [reps] [pyvrp_iters] [base_reps]
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
from swc.experiments import family, horizon, loo_tuned, policy_grid, run_or_load, social_cost, tuned_table
from swc.simulate import SimConfig, draw_increments, summarise
from swc.tehran import INSTANCE_VERSION, load

_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
base_reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CV, OVERFLOW_COST, AREA_KM2 = 1.0, 60.0, 21.4
BASE = ROOT / "results" / f"tehran_d6_{INSTANCE_VERSION}_r{base_reps}_it{iters}"
out = ROOT / "results" / f"tehran_sensitivity_{INSTANCE_VERSION}_r{reps}_it{iters}"
out.mkdir(parents=True, exist_ok=True)

# name: (capacities, decisions_only)
SCEN = {
    "gen0.7": (("current", "double"), False), "gen1.3": (("current", "double"), False),
    "leg3": (("current",), False), "leg12": (("current",), False),
    "fixed0.5": (("current", "double"), False), "fixed1.5": (("current", "double"), False),
    "km0.5": (("current", "double"), False), "km1.5": (("current", "double"), False),
    "spill_lost": (("current", "double"), False), "split6": (("current", "double"), False),
    "targeted": (("targeted",), False), "large": (("large",), False),
    "alpha_tight": (("current", "double"), True), "alpha_loose": (("current", "double"), True),
    "normal": (("current", "double"), True), "draft_prize": (("current", "double"), True),
}
RISK_FAMILIES = ("blind_risk", "risk")


def build(scen, cap, rep):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=OVERFLOW_COST, fill_rate_cv=CV)
    kw, mult = {}, 1.0
    if scen.startswith("gen"):
        mult = float(scen[3:])
    if scen.startswith("leg"):
        kw["transfer_leg_km"] = float(scen[3:])
    if scen == "split6":
        kw.update(max_bins_per_point=6, intra_m=100)
    if scen == "alpha_tight":
        P.risk = replace(P.risk, alpha_min=0.02, alpha_max=0.25)
    if scen == "alpha_loose":
        P.risk = replace(P.risk, alpha_min=0.10, alpha_max=0.70)
    if scen == "normal":
        P.risk = replace(P.risk, dist="normal")
    if scen == "draft_prize":
        P.risk = replace(P.risk, incremental=False)
    if scen == "spill_lost":
        P.spill = replace(P.spill, mode="lost")
    if cap == "large":
        P.ops = replace(P.ops, bin_volume_m3=2.2)
    inst = load(P, rate_mult=mult * (0.5 if cap in ("double", "large") else 1.0), **kw)
    if scen.startswith(("fixed", "km")):
        f = float(scen.lstrip("fixedkm"))
        mapped = {}
        for v in inst.vehicles:
            if id(v) not in mapped:
                mapped[id(v)] = (replace(v, fixed_cost=v.fixed_cost * f) if scen.startswith("fixed")
                                 else replace(v, var_cost_per_km=v.var_cost_per_km * f))
        inst.vehicles = [mapped[id(v)] for v in inst.vehicles]
    if cap == "double":
        inst.bins_per_point = inst.bins_per_point * 2
    if cap == "targeted":
        fast = inst.rate_mean_per_h >= np.median(inst.rate_mean_per_h)
        inst.bins_per_point = np.where(fast, 2 * inst.bins_per_point, inst.bins_per_point)
        inst.rate_mean_per_h = np.where(fast, inst.rate_mean_per_h / 2, inst.rate_mean_per_h)
        inst.rate_sd_per_h = np.where(fast, inst.rate_sd_per_h / 2, inst.rate_sd_per_h)
    rng = np.random.default_rng(50 + rep)            # same initial state rule as 09
    inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
    dtf = 1.0 / float(np.mean(inst.rate_mean_per_h * 24))
    return inst, dtf


def tuned_policies(cap):
    """Policies to run: the modal tuned choice of the periodic, threshold and risk families in the base
    run, plus fixed_daily, periodic2, threshold50/70 and the three risk variants. The blind (no-sensor)
    threshold and risk policies are left out: in every base case their social cost is 2-7 times that
    of the best policy without sensors (09 summary), so they cannot become the reference."""
    t = pd.read_csv(BASE / "tuned_by_rep.csv")
    t = t[(t.cv == CV) & (t.capacity == ("double" if cap in ("targeted", "large") else cap))]
    names = [t[t.tuned_family == f]["chosen"].mode().iloc[0] for f in ("periodic", "threshold", "risk")]
    names += ["fixed_daily", "periodic2", "threshold50", "threshold70", "risk_myopic", "risk_cfa_t5", "risk_cfa_t15"]
    grid = {p.name: p for p in policy_grid(1e9)}
    return [grid[n] for n in dict.fromkeys(names)]


# where the fleet is overloaded (waste +30%) prioritisation matters; the no-sensor blind risk policy is added
# there to separate the value of prioritising from the value of the readings
EXTRA_POLICIES = {"gen1.3": ["blind_risk"]}


def jobs():
    out_jobs = []
    grid = {p.name: p for p in policy_grid(1e9)}
    for rep in range(reps):
        for scen, (caps, dec_only) in SCEN.items():
            for cap in caps:
                for pol in tuned_policies(cap) + [grid[n] for n in EXTRA_POLICIES.get(scen, [])]:
                    if dec_only and family(pol.name) not in RISK_FAMILIES:
                        base_f = BASE / f"days_cv{CV:g}_{cap}_{pol.name}_rep{rep}.csv"
                        out_jobs.append((scen, cap, rep, pol, base_f, False))
                    else:
                        out_jobs.append((scen, cap, rep, pol, out / f"days_{scen}_{cap}_{pol.name}_rep{rep}.csv", True))
    return out_jobs


J = jobs()
if SH_N > 1:
    run = [j for j in J if j[5]]
    for k, (scen, cap, rep, pol, f, _) in enumerate(run):
        if k % SH_N != SH_I or f.exists():
            continue
        inst, dtf = build(scen, cap, rep)
        burn, days = horizon(dtf)
        run_or_load(f, inst, pol, draw_increments(inst, days, seed=2000 + rep), SimConfig(days=days, pyvrp_s=300,
                                                                                          pyvrp_iters=iters))
        print("done", f.name, flush=True)
    sys.exit(0)

be = import_module("08_break_even")
rows = []
for scen, cap, rep, pol, f, own in J:
    inst, dtf = build(scen, cap, rep)
    burn, days = horizon(dtf)
    if own:
        recs = run_or_load(f, inst, pol, draw_increments(inst, days, seed=2000 + rep),
                           SimConfig(days=days, pyvrp_s=300, pyvrp_iters=iters))
    else:
        recs = run_or_load(f, None, None, None, None)        # base-run file (must exist)
    s = summarise(recs, burn_in=burn)
    bins = int(inst.bins_per_point.sum())
    s.update(scenario=scen, capacity=cap, rep=rep, policy=pol.name, bins=bins, days_to_full=dtf,
             sensor_cost_day=be.sensor_cost_per_day(bins, AREA_KM2))
    s["social_cost"] = social_cost(s, OVERFLOW_COST)
    rows.append(s)
df = pd.DataFrame(rows)
df.to_csv(out / "summary_by_rep.csv", index=False)
tuned = loo_tuned(df, ["scenario", "capacity"])
tuned.to_csv(out / "tuned_by_rep.csv", index=False)
t = tuned_table(tuned, ["scenario", "capacity"])
t.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 250)
print(t[t.family.isin(["best_no_sensor", "best_sensor", "fixed_daily"])].round(1).to_string(index=False))
print("saved", out)
