"""Price robustness of the Tehran District 6 conclusions (v4): prices that change no routing decision.

Sensor-network, container and cleanup prices, and the overflow valuation used for scoring, do not enter
any nightly decision (the risk policies price overflow at their own c_o = 60), so they can be varied
ex post on the simulated runs. Truck prices do change decisions and are re-optimised in
12_tehran_sensitivity.py. For every price scenario the families are re-tuned leave-one-replication-out on
the re-priced social cost, and two conclusions are re-checked per variability level:
  S  sensing does not pay: best sensor policy (incl. sensor network) vs best policy without sensors
     (value = net saving of sensing; the claim holds when the whole 95% interval is below zero)
  K  doubling container capacity lowers the social cost, container capital cost included
usage: python scripts/13_cost_robustness.py
"""
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION as V
from swc.experiments import loo_tuned, paired_ci

be = import_module("08_break_even")
res = ROOT / "results" / f"tehran_d6_{V}_r5_it5000"
df0 = pd.read_csv(res / "summary_by_rep.csv")
BIN_USD, BIN_LIFE_Y, EXTRA_BINS = 150.0, 8.0, 2852
BASE = dict(sensor=1.0, container=1.0, cleanup=30.0, overflow=60.0)
SCEN = {"base": {}, "sensor x0.5": dict(sensor=0.5), "sensor x2": dict(sensor=2.0),
        "container x0.5": dict(container=0.5), "container x1.5": dict(container=1.5),
        "cleanup 0": dict(cleanup=0.0), "cleanup 100": dict(cleanup=100.0),
        "overflow 30": dict(overflow=30.0), "overflow 120": dict(overflow=120.0)}

rows = []
for name, ch in SCEN.items():
    p = {**BASE, **ch}
    d = df0.copy()
    d["sensor_cost_day"] = d["sensor_cost_day"] * p["sensor"]
    d["social_cost"] = (d.cost_per_day + p["cleanup"] * d.spill_t_per_day
                        + p["overflow"] * d.overflow_weighted_per_day)
    t = loo_tuned(d, ["cv", "capacity"])
    t["net"] = t.social_cost + np.where(t.sensing.astype(bool), t.sensor_cost_day, 0.0)
    t.loc[t.capacity == "double", "net"] += p["container"] * EXTRA_BINS * BIN_USD / (BIN_LIFE_Y * 365)
    for cv, g in t.groupby("cv"):
        for cap in ("current", "double"):
            c = g[g.capacity == cap]
            ns = c[c.tuned_family == "best_no_sensor"].set_index("rep")
            se = c[c.tuned_family == "best_sensor"].set_index("rep").loc[ns.index]
            s, hw = paired_ci(ns.net - se.net)
            rows.append(dict(scenario=name, cv=cv, claim=f"S sensing does not pay ({cap})", value=round(s, 1),
                             hw=round(hw, 1), holds_original=s + hw < 0,
                             detail=f"{ns.chosen.mode().iloc[0]} vs {se.chosen.mode().iloc[0]}"))
        best = {cap: g[(g.capacity == cap) & g.tuned_family.isin(["best_no_sensor", "best_sensor"])]
                .groupby("rep").net.min() for cap in ("current", "double")}
        k, hw = paired_ci(best["double"] - best["current"])
        rows.append(dict(scenario=name, cv=cv, claim="K doubling capacity lowers social cost", value=round(k, 1),
                         hw=round(hw, 1), holds_original=k + hw < 0, detail="best option per capacity"))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
out.to_csv(res / "cost_robustness.csv", index=False)
print("saved", res / "cost_robustness.csv")
