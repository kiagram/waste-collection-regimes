"""Are the v4 conclusions robust to the routing engine's optimality gap?

17_heuristic_quality.py measures how far the simulation engine's nightly cost lies above a much longer
search (gap range [gmin, gmax] per instance family and policy class). Here each sensing decision is
recomputed under the adversarial correction: the engine gap affects operating cost only, true operating
cost = reported / (1 + g). To test a claim that sensing does NOT pay, the sensor option is corrected by
the LARGEST gap and the no-sensor option by the SMALLEST (the sensor saving grows as much as the
measurements allow); to test a claim that sensing DOES pay, the reverse. Policy classes without their own
measurement (periodic, threshold, blind) take the full measured range of their instance family.
Overflow, cleanup and sensor costs are not affected by the routing gap.
usage: python scripts/18_engine_bias.py [regime_reps]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION as V
from swc.experiments import family

REG_REPS = sys.argv[1] if len(sys.argv) > 1 else "8"
OUT = ROOT / "results" / f"heuristic_quality_{V}"
g = pd.read_csv(OUT / "district_gaps.csv").dropna(subset=["gap_pct"])
g["gap"] = g.gap_pct.clip(lower=0) / 100


def gap_range(kind, scenario, pol):
    """[gmin, gmax] for a policy on an instance family/scenario."""
    sub = g[g.kind == kind]
    cls = "fd" if pol == "fixed_daily" else ("risk" if family(pol) == "risk" else None)
    if kind == "synthetic":       # nearest measured fill speed
        mults = sorted(sub.scenario.astype(float).unique())
        m = min(mults, key=lambda x: abs(x - float(scenario)))
        sub = sub[sub.scenario.astype(float) == m]
    else:
        sub = sub[sub.scenario == ("double" if scenario in ("double", "targeted", "large") else "current")]
    s = sub[sub.policy == cls] if cls is not None and (sub.policy == cls).any() else g[g.kind == kind]
    return float(s.gap.min()), float(s.gap.max())


def corrected(row, kind, scenario, which):
    lo, hi = gap_range(kind, scenario, row["policy"])
    gap = hi if which == "max" else lo
    op = row["cost_per_day"] / (1 + gap)
    sens = row["sensor_cost_day"] if row["sensing"] else 0.0
    return op + (row["social_cost"] - row["cost_per_day"]) + sens


def verdicts(tuned, keys, kind, scen_key):
    out = []
    for cell, c in tuned.groupby(keys, sort=False):
        ns = c[c.tuned_family == "best_no_sensor"].set_index("rep")
        se = c[c.tuned_family == "best_sensor"].set_index("rep").loc[ns.index]
        scen = cell[keys.index(scen_key)]
        rep_net = (ns.social_cost - se.social_cost - se.sensor_cost_day).mean()
        # most favourable to sensing / least favourable, over the measured gap ranges
        fav = np.mean([corrected(ns.loc[r], kind, scen, "min") - corrected(se.loc[r], kind, scen, "max")
                       for r in ns.index])
        unfav = np.mean([corrected(ns.loc[r], kind, scen, "max") - corrected(se.loc[r], kind, scen, "min")
                         for r in ns.index])
        out.append(dict(zip(keys, cell), best_no_sensor="/".join(sorted(set(ns.chosen))),
                        best_sensor="/".join(sorted(set(se.chosen))), net_saving_reported=round(rep_net, 1),
                        net_saving_worst_for_sensing=round(unfav, 1), net_saving_best_for_sensing=round(fav, 1),
                        verdict=("sensing pays under any gap" if unfav > 0 else
                                 "sensing does not pay under any gap" if fav < 0 else "depends on the gap")))
    return pd.DataFrame(out)


pd.set_option("display.width", 250)
T = pd.read_csv(ROOT / "results" / f"tehran_d6_{V}_r5_it5000" / "tuned_by_rep.csv")
t1 = verdicts(T, ["cv", "capacity"], "tehran", "capacity")
print("Tehran District 6 (net saving of the best sensor policy over the best no-sensor policy, USD/day):")
print(t1.to_string(index=False))
S = pd.read_csv(ROOT / "results" / f"tehran_sensitivity_{V}_r3_it5000" / "tuned_by_rep.csv")
t2 = verdicts(S, ["scenario", "capacity"], "tehran", "capacity")
print("\nTehran sensitivity scenarios:")
print(t2.to_string(index=False))
R = pd.read_csv(ROOT / "results" / f"regime_map_{V}_n150_r{REG_REPS}_it5000" / "tuned_by_rep.csv")
t3 = verdicts(R, ["rate_mult", "cv", "layout"], "synthetic", "rate_mult")
print("\nRegime map:")
print(t3.to_string(index=False))
t1.to_csv(OUT / "bias_tehran.csv", index=False)
t2.to_csv(OUT / "bias_tehran_sensitivity.csv", index=False)
t3.to_csv(OUT / "bias_regime.csv", index=False)
print("saved", OUT)
