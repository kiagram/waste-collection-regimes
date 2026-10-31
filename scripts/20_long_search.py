"""Do the sensing decisions survive a much longer routing search on complete trajectories? (review R5)

Single-night gap measurements (17_heuristic_quality.py) say how far one plan is from a better one, but
routing decisions also change the next days' fill levels. Here the decisive comparison of each cell -
the best policy without sensors against the best sensor policy (modal tuned choices, 06 / 09) - is
re-simulated over the full horizon with an 8x longer search (best of 4 warm-started runs x 20,000
iterations instead of 2 x 5,000), on the same instances and fill realisations, and the net saving of
sensing is compared with the standard engine.
Cells: regime map at CV 1.0 (both layouts, 2.8-8 days to full), Tehran CV 1.0 (current and doubled).
Resumable; SHARD=i/N. usage: python scripts/20_long_search.py [iters] [starts]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION as V
from swc.experiments import horizon, paired_ci, policy_grid, regime_instance, run_or_load, social_cost
from swc.simulate import SimConfig, draw_increments, summarise

_sh = os.environ.get("SHARD")
SH_I, SH_N = (int(x) for x in _sh.split("/")) if _sh else (0, 1)
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
STARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
REG = ROOT / "results" / f"regime_map_{V}_n150_r8_it5000"
TEH = ROOT / "results" / f"tehran_d6_{V}_r5_it5000"
OUT = ROOT / "results" / f"long_search_{V}_it{ITERS}x{STARTS}"
OUT.mkdir(parents=True, exist_ok=True)
GRID = {p.name: p for p in policy_grid(1e9)}


def modal(t, fam):
    return t[t.tuned_family == fam]["chosen"].mode().iloc[0]


def tehran_case(cap, rep):
    src = (ROOT / "scripts" / "09_tehran_sim.py").read_text(encoding="utf-8")
    src = src[src.index("def scenario"):src.index("jobs = []")]
    from dataclasses import replace
    from swc.config import Params
    from swc.tehran import load
    g = {"replace": replace, "Params": Params, "load": load, "np": np, "OVERFLOW_COST": 60.0}
    exec(compile(src, "09", "exec"), g)
    return g["scenario"](1.0, cap, rep)


CASES = []   # (tag, kind, key, rep, policy, standard-run file)
rt = pd.read_csv(REG / "tuned_by_rep.csv")
for layout in ("clustered", "uniform"):
    for mult in (1.0, 0.8, 0.65, 0.5, 0.4):
        t = rt[(rt.rate_mult == mult) & (rt.cv == 1.0) & (rt.layout == layout)]
        for fam in ("best_no_sensor", "best_sensor"):
            pol = modal(t, fam)
            for rep in range(3):
                CASES.append((f"m{mult:g}_cv1_{layout}", "regime", (mult, layout), rep, pol,
                              REG / f"days_m{mult:g}_cv1_{layout}_{pol}_rep{rep}.csv"))
tt = pd.read_csv(TEH / "tuned_by_rep.csv")
for cap in ("current", "double"):
    t = tt[(tt.cv == 1.0) & (tt.capacity == cap)]
    for fam in ("best_no_sensor", "best_sensor"):
        pol = modal(t, fam)
        for rep in range(2):
            CASES.append((f"tehran_cv1_{cap}", "tehran", cap, rep, pol, TEH / f"days_cv1_{cap}_{pol}_rep{rep}.csv"))


def build(kind, key, rep):
    if kind == "regime":
        inst, dtf = regime_instance(150, key[0], 1.0, key[1], rep)
        return inst, dtf, 1000 + rep, 120.0
    inst, dtf = tehran_case(key, rep)
    return inst, dtf, 2000 + rep, 300.0


def fname(tag, pol, rep):
    return OUT / f"days_{tag}_{pol}_rep{rep}.csv"


if SH_N > 1:
    for j, (tag, kind, key, rep, pol, _) in enumerate(CASES):
        f = fname(tag, pol, rep)
        if j % SH_N != SH_I or f.exists():
            continue
        inst, dtf, seed, cap_s = build(kind, key, rep)
        burn, days = horizon(dtf)
        cfg = SimConfig(days=days, pyvrp_iters=ITERS, pyvrp_starts=STARTS, pyvrp_s=cap_s * 8)
        run_or_load(f, inst, GRID[pol], draw_increments(inst, days, seed=seed), cfg)
        print("done", f.name, flush=True)
    sys.exit(0)

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
be = import_module("08_break_even")
rows = []
for tag, kind, key, rep, pol, std_file in CASES:
    inst, dtf, seed, cap_s = build(kind, key, rep)
    burn, days = horizon(dtf)
    bins = int(inst.bins_per_point.sum())
    area = 21.4 if kind == "tehran" else (5.0 * (150 / 40) ** 0.5 / 1.6) ** 2
    for engine, f in (("standard", std_file), ("long", fname(tag, pol, rep))):
        s = summarise(run_or_load(f, None, None, None, None), burn_in=burn)
        s.update(cell=tag, rep=rep, policy=pol, engine=engine,
                 sensing=GRID[pol].uses_sensors, sensor_cost_day=be.sensor_cost_per_day(bins, area))
        s["social_cost"] = social_cost(s, 60.0)
        s["net_cost"] = s["social_cost"] + (s["sensor_cost_day"] if s["sensing"] else 0.0)
        rows.append(s)
df = pd.DataFrame(rows)
df.to_csv(OUT / "summary_by_rep.csv", index=False)
out = []
for (cell, engine), g in df.groupby(["cell", "engine"], sort=False):
    ns, se = g[~g.sensing].set_index("rep"), g[g.sensing].set_index("rep")
    d, hw = paired_ci(ns.net_cost - se.loc[ns.index].net_cost)
    out.append(dict(cell=cell, engine=engine, no_sensor=ns.policy.iloc[0], sensor=se.policy.iloc[0],
                    no_sensor_op=ns.cost_per_day.mean(), sensor_op=se.cost_per_day.mean(),
                    net_saving=d, net_saving_hw=hw))
t = pd.DataFrame(out)
w = df.pivot_table(index=["cell", "policy", "rep"], columns="engine", values="cost_per_day")
w["op_change_pct"] = 100 * (w["long"] / w["standard"] - 1)
pd.set_option("display.width", 250)
print(t.round(1).to_string(index=False))
print("\noperating cost, long vs standard engine (%):")
print(w.groupby(["cell", "policy"]).op_change_pct.mean().round(2).to_string())
t.to_csv(OUT / "summary.csv", index=False)
w.to_csv(OUT / "operating_cost_change.csv")
print("saved", OUT)
