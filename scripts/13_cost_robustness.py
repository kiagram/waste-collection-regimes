"""Cost robustness of the Tehran District 6 conclusions (ex-post re-costing).

Each cost component is varied one at a time by -50% / +50% (truck fixed cost per shift, cost
per km, sensor-network cost, container price). The simulated plans (optimised at base prices)
are re-costed; the paper states this approximation. For every scenario the four headline
claims of R13 are re-checked:
  C1 sensors alone (current bins, risk policy) cost more than today's practice
  C2 doubling capacity reduces overflow volume (cost-independent; checked once)
  C3 with doubled bins, sensors + risk policy is cheaper than collecting every point daily
  C4 the threshold rule is the cheapest option but overflows more than today
usage: python scripts/13_cost_robustness.py [results_dir_name]
"""
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

be = import_module("08_break_even")
from swc.config import MODEL_VERSION
res_dir = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else f"tehran_d6_{MODEL_VERSION}_r5_d20_it5000")
BURN, AREA, BINS = 3, 21.4, {"current": 2852, "double": 5704}
BIN_USD, BIN_LIFE_Y = 150.0, 8.0

rows = []
for f in sorted(res_dir.glob("days_cv1_*_rep*.csv")):
    _, cv, cap, *pol, rep = f.stem.split("_")
    pol = "_".join(pol)
    d = pd.read_csv(f).iloc[BURN:]
    rows.append(dict(capacity=cap, policy=pol, rep=int(rep[3:]), fixed=d.cost_fixed.mean(),
                     variable=d.cost_variable.mean(), overflow_m3=1.1 * d.overflow_volume_bins.mean()))
base = pd.DataFrame(rows).groupby(["capacity", "policy"])[["fixed", "variable", "overflow_m3"]].mean()
reps = pd.DataFrame(rows).rep.nunique()
print(f"source: {res_dir.name} ({reps} replications, CV 1.0)\n")


def totals(f_fix=1.0, f_var=1.0, f_sensor=1.0, f_bin=1.0):
    t = {}
    for (cap, pol), r in base.iterrows():
        capex = 0.0
        if cap == "double":
            capex += f_bin * 2852 * BIN_USD / (BIN_LIFE_Y * 365)
        if pol != "fixed_daily":
            capex += f_sensor * be.sensor_cost_per_day(BINS[cap], AREA)
        t[(cap, pol)] = f_fix * r.fixed + f_var * r.variable + capex
    return t


def claims(t):
    today = t[("current", "fixed_daily")]
    thr = t[("current", "threshold70")]
    return {
        "C1 sensors alone cost more": t[("current", "risk_cfa_t5")] > today,
        "C3 double+sensors < double+daily": t[("double", "risk_cfa_t5")] < t[("double", "fixed_daily")],
        "C4 threshold cheapest": thr <= min(t.values()) + 1e-9,
    }


scen = [("base", {})]
for comp in ("f_fix", "f_var", "f_sensor", "f_bin"):
    for m in (0.5, 1.5):
        scen.append((f"{comp}x{m:g}", {comp: m}))
out = []
for name, kw in scen:
    t = totals(**kw)
    c = claims(t)
    out.append(dict(scenario=name, today=round(t[("current", "fixed_daily")]),
                    sensors_only=round(t[("current", "risk_cfa_t5")]), double_daily=round(t[("double", "fixed_daily")]),
                    double_sensors=round(t[("double", "risk_cfa_t5")]), threshold=round(t[("current", "threshold70")]),
                    **c))
tab = pd.DataFrame(out)
print(tab.to_string(index=False))
ov = base["overflow_m3"]
print(f"\nC2 overflow m3/day: today {ov[('current', 'fixed_daily')]:.1f} -> double+daily "
      f"{ov[('double', 'fixed_daily')]:.1f}, double+sensors {ov[('double', 'risk_cfa_t5')]:.1f}; "
      f"threshold {ov[('current', 'threshold70')]:.1f}")
held = {k: bool(tab[k].all()) for k in tab.columns if k.startswith("C")}
print("claims holding in ALL cost scenarios:", held)
tab.to_csv(res_dir / "cost_robustness.csv", index=False)
print("saved", res_dir / "cost_robustness.csv")


def breakeven(comp, lo=0.1, hi=5.0, tol=1e-3):
    """Multiplier of one cost component at which C3 flips (None if it never flips in [lo, hi])."""
    f = lambda m: (lambda t: t[("double", "fixed_daily")] - t[("double", "risk_cfa_t5")])(totals(**{comp: m}))
    a, b = f(lo), f(hi)
    if a * b > 0:
        return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) * a > 0:
            lo, a = mid, f(mid)
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


print("\nC3 break-even multipliers (sensors pay after capacity expansion only on one side of these):")
for comp, label in (("f_fix", "truck fixed cost/shift"), ("f_var", "cost per km"), ("f_sensor", "sensor-network cost"),
                    ("f_bin", "container price")):
    m = breakeven(comp)
    print(f"  {label:24s}: {'never flips in 0.1-5x' if m is None else f'flips at {m}x base'}")
