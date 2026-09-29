"""Are the conclusions robust to the routing engine's optimality gap?

17_heuristic_quality.py measures, per policy class, how far the simulation engine's cost lies above a
much longer search on representative nights (gap range [gmin, gmax]). Here every headline comparison is
recomputed under the adversarial correction: the option claimed to be better is corrected by its
SMALLEST measured gap (true cost = reported / (1 + gmin)) and the option it is compared with by its
LARGEST (reported / (1 + gmax)), or the reverse when the claim is that a saving is small.
Capital costs (sensors, containers) are not affected. Classes without a measurement (threshold rule)
are left uncorrected, which is conservative for every claim below.
usage: python scripts/18_engine_bias.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module

import pandas as pd

from swc.config import MODEL_VERSION as V

be = import_module("08_break_even")
g = pd.read_csv(ROOT / "results" / f"heuristic_quality_{V}" / "district_gaps.csv").dropna(subset=["gap_pct"])
rng = g.groupby(["kind", "scenario", "policy"]).gap_pct.agg(["min", "max"]) / 100
print("measured gap ranges (fraction):\n", rng.round(3))

T = pd.read_csv(ROOT / "results" / f"tehran_d6_{V}_r5_d20_it5000" / "summary_by_rep.csv")
T = T[T.cv == 1.0].groupby(["capacity", "policy"]).cost_per_day.mean()
SENS = {"current": be.sensor_cost_per_day(2852, 21.4), "double": be.sensor_cost_per_day(5704, 21.4)}
BINS = 2852 * 150.0 / (8 * 365)


def teh(cap, pol, bound):
    cls = "fd" if pol == "fixed_daily" else ("risk" if pol.startswith("risk") else None)
    gap = 0.0 if cls is None else rng.loc[("tehran", cap, cls), bound]
    capex = (BINS if cap == "double" else 0.0) + (SENS[cap] if pol != "fixed_daily" else 0.0)
    return T[(cap, pol)] / (1 + gap) + capex


rows = []
# C1: sensors alone cost more than today (claim: RC > FD). Adversarial: RC small, FD large.
rows.append(("C1 sensors alone cost more than daily collection", teh("current", "risk_cfa_t5", "max"),
             teh("current", "fixed_daily", "min"), ">"))
# C3: with doubled bins, sensors cheaper than daily (claim: double RC < double FD). Adversarial: RC large, FD small.
rows.append(("C3 doubled bins: sensors cheaper than daily collection", teh("double", "risk_cfa_t5", "min"),
             teh("double", "fixed_daily", "max"), "<"))
# C4: threshold cheapest (claim: TH < FD). TH uncorrected; FD corrected by its largest gap.
rows.append(("C4 threshold rule cheaper than daily collection", teh("current", "threshold70", "min"),
             teh("current", "fixed_daily", "max"), "<"))
out = []
for name, a, b, op in rows:
    holds = a > b if op == ">" else a < b
    out.append(dict(claim=name, option=round(a), compared_with=round(b), diff_pct=round(100 * (a / b - 1), 1), holds=holds))
t1 = pd.DataFrame(out)
print("\nTehran District 6 under adversarial gap correction (USD/day incl. capital cost):\n", t1.to_string(index=False))

R = pd.read_csv(ROOT / "results" / f"regime_map_{V}_n150_r5_d20_it5000" / "summary_by_rep.csv")
R = R[R.cv == 1.0].groupby(["rate_mult", "policy"]).cost_per_day.mean()
out = []
for m in (1.0, 0.5, 0.33):
    fd, rk = R[(m, "fixed_daily")], R[(m, "risk_myopic")]
    gf, gr = rng.loc[("synthetic", str(m), "fd")], rng.loc[("synthetic", str(m), "risk")]
    lo = 100 * (rk / (1 + gr["min"]) / (fd / (1 + gf["max"])) - 1)   # saving shrinks most
    hi = 100 * (rk / (1 + gr["max"]) / (fd / (1 + gf["min"])) - 1)   # saving grows most
    out.append(dict(days_to_full={1.0: 2.8, 0.5: 5.5, 0.33: 8.3}[m], reported=round(100 * (rk / fd - 1), 1),
                    corrected_range=f"{min(lo, hi):+.1f} to {max(lo, hi):+.1f}"))
t2 = pd.DataFrame(out)
print("\nRegime map (CV 1.0), risk-based vs daily collection, operating-cost change (%):\n", t2.to_string(index=False))
d = ROOT / "results" / f"heuristic_quality_{V}"
t1.to_csv(d / "bias_tehran.csv", index=False)
t2.to_csv(d / "bias_regime.csv", index=False)
print("saved", d)
