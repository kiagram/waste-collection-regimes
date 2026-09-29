"""Solution quality of the routing engine used in the simulations.

Engine (swc.simulate.SimConfig): greedy warm start + truck-type swaps, best of 2 runs x 5,000 iterations.
(a) Sub-districts (instances of 16_exact_fronts_tehran.py, both modes): engine cost vs the proven
    CP-SAT minimum cost.
(b) District scale: engine vs a long reference (best of 4 runs x 40,000 iterations: warm and cold
    starts, two seeds each) on representative nights:
    - Tehran District 6 (323 points, 34 trucks), current and doubled capacity, fixed-daily and
      risk-based policies, on a mid-cycle night (replications 0-1) and a steady-state night of daily
      collection (fill = one day's increment);
    - the synthetic district of the regime map (150 points) at the three fill speeds.
usage: python scripts/17_heuristic_quality.py [processes]
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION, Params
from swc.evaluate import evaluate
from swc.heuristic_pyvrp import solve_pyvrp, solve_pyvrp_best
from swc.instance import generate_synthetic
from swc.risk import mandatory_bins, overflow_risk, skip_penalty
from swc.simulate import FixedDaily, RiskMyopic, draw_increments
from swc.tehran import TEHRAN_FLEET, load

OUT = ROOT / "results" / f"heuristic_quality_{MODEL_VERSION}"
FRONTS = ROOT / "results" / f"exact_fronts_tehran_{MODEL_VERSION}_{{}}_s6_n12_g6_tl120"
FLEET = ["new_compactor", "old_compactor", "old_compactor", "small_loader"]
ENGINE = dict(starts=2, max_iterations=5000)
REF_ITERS = 40_000


def tehran(cap="current", rep=0, state="mid"):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=1.0)
    inst = load(P, rate_mult=1.0 if cap == "current" else 0.5)
    if cap == "double":
        inst.bins_per_point = inst.bins_per_point * 2
    return set_state(inst, rep, state, seed0=50, inc_seed=2000)


def synthetic(mult, rep=0, state="mid"):
    P = Params()
    P.risk = replace(P.risk, overflow_cost=60.0, fill_rate_cv=1.0)
    n = 150
    fleet = {"new_compactor": max(1, n // 60), "old_compactor": max(2, n // 30), "small_loader": max(1, n // 60)}
    inst = generate_synthetic(n, seed=100 + rep, params=P, bins_per_point=(4, 10), fleet_counts=fleet,
                              area_m=5000.0 * (n / 40) ** 0.5 / 1.6)
    inst.rate_mean_per_h = inst.rate_mean_per_h * mult
    inst.rate_sd_per_h = inst.rate_sd_per_h * mult
    return set_state(inst, rep, state, seed0=None, inc_seed=1000)


def set_state(inst, rep, state, seed0, inc_seed):
    if state == "steady":   # the night after a daily collection: fill = one day's increment
        inst.fill = np.minimum(draw_increments(inst, 5, seed=inc_seed + rep)[3], 1.0)
    elif seed0 is not None:  # mid-cycle initial state of the simulations
        rng = np.random.default_rng(seed0 + rep)
        inst.fill = np.clip(rng.uniform(0.0, 1.0, inst.n) * inst.rate_mean_per_h * 24 * 2, 0, 1)
    return inst


def sub_job(args):
    mode, s = args
    r = json.loads((Path(str(FRONTS).format(mode)) / f"sub{s}.json").read_text())
    inst = tehran().subset(r["info"]["members"])
    inst.vehicles = [TEHRAN_FLEET[v] for v in FLEET]
    risk = overflow_risk(inst, mode="count")
    pen = skip_penalty(inst, risk)
    mand = mandatory_bins(inst) if mode == "selective" else set(inst.bins)
    exact = min(e["cost"] for e in r["exact"])
    sol = solve_pyvrp_best(inst, pen, mand, time_s=300, seed=0, **ENGINE)
    c = evaluate(inst, sol.routes, pen, risk, mand).cost_total
    return dict(case=f"{mode}_sub{s + 1}", exact=exact, engine=c, gap_pct=100 * (c - exact) / exact)


CASES = ([("tehran", cap, pol, rep, "mid") for cap in ("current", "double") for pol in ("fd", "risk") for rep in (0, 1)]
         + [("tehran", cap, "fd", 0, "steady") for cap in ("current", "double")]
         + [("synthetic", m, pol, 0, st) for m in (1.0, 0.5, 0.33) for pol in ("fd", "risk") for st in ("mid", "steady")])


def district_job(args):
    case, run = args
    kind, a, pol, rep, state = case
    inst = tehran(a, rep, state) if kind == "tehran" else synthetic(a, rep, state)
    pen, mand = (FixedDaily() if pol == "fd" else RiskMyopic()).decide(inst, 0)
    if run == "engine":
        sol = solve_pyvrp_best(inst, pen, mand, time_s=3600, seed=0, **ENGINE)
    else:
        ws, seed = run
        sol = solve_pyvrp(inst, pen, mand, time_s=3600, seed=seed, max_iterations=REF_ITERS, max_no_improve=10**9,
                          warm_start=ws)
    k = evaluate(inst, sol.routes, pen, None, mand)
    return dict(kind=kind, scenario=str(a), policy=pol, rep=rep, state=state, run=str(run),
                cost=k.cost_total if k.feasible else np.nan, trucks=k.trucks_used, wall_s=sol.wall_s)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    subs = [(m, s) for m in ("full", "selective") for s in range(6)]
    runs = ["engine", (True, 10), (True, 11), (False, 10), (False, 11)]
    jobs = [(c, r) for c in CASES for r in runs]
    with Pool(procs) as pool:
        a = pd.DataFrame(pool.map(sub_job, subs))
        b = pd.DataFrame(pool.map(district_job, jobs))
    a.to_csv(OUT / "subdistricts.csv", index=False)
    b.to_csv(OUT / "district_runs.csv", index=False)
    print(a.round(3).to_string(index=False))
    print(f"sub-districts: engine = proven optimum in {int((a.gap_pct < 0.01).sum())}/{len(a)}, "
          f"max gap {a.gap_pct.max():.2f}%\n")
    key = ["kind", "scenario", "policy", "rep", "state"]
    eng = b[b.run == "engine"].set_index(key)[["cost", "trucks"]]
    ref = b[b.run != "engine"].groupby(key).cost.min().rename("ref")
    g = eng.join(ref)
    g["gap_pct"] = 100 * (g.cost - g.ref) / g.ref
    g = g.reset_index()
    g.to_csv(OUT / "district_gaps.csv", index=False)
    print(g.round(2).to_string(index=False))
    for kind, x in g.groupby("kind"):
        print(f"{kind}: engine vs best of 4 x {REF_ITERS:,}-iteration runs: mean gap {x.gap_pct.mean():.2f}%, "
              f"max {x.gap_pct.max():.2f}%, min {x.gap_pct.min():.2f}%")
    print("saved", OUT)
