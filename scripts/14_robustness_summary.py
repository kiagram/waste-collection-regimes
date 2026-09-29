"""Summaries for the robustness round: Tehran (5 reps, 95% CI), Tehran sensitivity scenarios,
and the regime map (5 reps). Writes CSV tables next to each result folder."""
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy import stats

be = import_module("08_break_even")
sys.path.insert(0, str(ROOT / "src"))
from swc.config import MODEL_VERSION as V
BIN_CAPEX = 2852 * 150.0 / (8 * 365)
BINS = {"current": 2852, "double": 5704}


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def capex(cap, pol):
    c = BIN_CAPEX if cap == "double" else 0.0
    return c + (be.sensor_cost_per_day(BINS[cap], 21.4) if pol != "fixed_daily" else 0.0)


# ---------------------------------------------------------------- Tehran, 5 replications
T = ROOT / "results" / f"tehran_d6_{V}_r5_d20_it5000"
d = pd.read_csv(T / "summary_by_rep.csv")
d["total"] = d.cost_per_day + [capex(c, p) for c, p in zip(d.capacity, d.policy)]
d["ovf_m3"] = 1.1 * d.overflow_volume_bins_per_day
rows = []
for cv, g in d.groupby("cv", sort=False):
    today = g[(g.capacity == "current") & (g.policy == "fixed_daily")].set_index("rep")
    for (cap, pol), x in g.groupby(["capacity", "policy"], sort=False):
        x = x.set_index("rep")
        dt = (x.total / today.total - 1) * 100
        do = (x.ovf_m3 - today.ovf_m3)
        rows.append(dict(cv=cv, capacity=cap, policy=pol, total_per_day=round(x.total.mean()),
                         d_total_pct=f"{dt.mean():+.1f} ± {ci(dt):.1f}", co2_kg=round(x.co2_kg_per_day.mean()),
                         overflow_m3=f"{x.ovf_m3.mean():.1f} ± {ci(x.ovf_m3):.1f}",
                         d_overflow_m3=f"{do.mean():+.1f} ± {ci(do):.1f}", trucks=round(x.truck_shifts_per_day.mean(), 1)))
t1 = pd.DataFrame(rows)
t1.to_csv(T / "headline_ci.csv", index=False)
print("== Tehran District 6, 5 replications (95% CI, paired vs today's practice)\n", t1.to_string(index=False))

# ---------------------------------------------------------------- Tehran sensitivity scenarios
S = ROOT / "results" / f"tehran_sensitivity_{V}_r3_d20_it5000"
s = pd.read_csv(S / "summary_by_rep.csv")
s["total"] = s.cost_per_day + [capex(c, p) for c, p in zip(s.capacity, s.policy)]
s["ovf_m3"] = 1.1 * s.overflow_volume_bins_per_day
base = d[(d.cv == 1.0) & (d.rep < 3)].assign(scenario="base", days_to_full=np.nan)
rows = []
for sc, g in pd.concat([base, s]).groupby("scenario", sort=False):
    for (cap, pol), x in g.groupby(["capacity", "policy"], sort=False):
        rows.append(dict(scenario=sc, capacity=cap, policy=pol, total=round(x.total.mean()),
                         overflow_m3=round(x.ovf_m3.mean(), 1), trucks=round(x.truck_shifts_per_day.mean(), 1),
                         days_to_full=round(x.days_to_full.mean(), 2) if x.days_to_full.notna().any() else None))
t2 = pd.DataFrame(rows)
t2.to_csv(S / "scenario_table.csv", index=False)
print("\n== Tehran sensitivity scenarios (3 reps; base = first 3 reps of the main run)\n", t2.to_string(index=False))

# claims per scenario
print("\n== Claims per scenario")
for sc, g in t2.groupby("scenario", sort=False):
    v = {(r.capacity, r.policy): r for r in g.itertuples()}
    msg = []
    if ("current", "fixed_daily") in v and ("current", "risk_cfa_t5") in v:
        msg.append(f"C1 sensors-alone dearer: {v[('current', 'risk_cfa_t5')].total > v[('current', 'fixed_daily')].total}")
    if ("current", "fixed_daily") in v and ("double", "fixed_daily") in v:
        a, b = v[("current", "fixed_daily")].overflow_m3, v[("double", "fixed_daily")].overflow_m3
        msg.append(f"C2 capacity cuts overflow: {b < a} ({a:.0f}->{b:.0f} m3)")
    if ("double", "fixed_daily") in v and ("double", "risk_cfa_t5") in v:
        msg.append(f"C3 double+sensors cheaper than double+daily: {v[('double', 'risk_cfa_t5')].total < v[('double', 'fixed_daily')].total}")
    if ("current", "threshold70") in v:
        cur = [r.total for (c, p), r in v.items() if c == "current"]
        msg.append(f"C4 threshold cheapest (current bins): {v[('current', 'threshold70')].total <= min(cur)}")
    print(f"  {sc:12s} " + " | ".join(msg))

# ---------------------------------------------------------------- regime map, 5 replications
R = ROOT / "results" / f"regime_map_{V}_n150_r5_d20_it5000"
r = pd.read_csv(R / "summary_by_rep.csv")
rows = []
for (m, cv), g in r.groupby(["rate_mult", "cv"]):
    b = g[g.policy == "fixed_daily"].set_index("rep")
    row = dict(days_to_full=round(g.days_to_full.mean(), 1), cv=cv, fixed_overflow=round(b.overflow_bins_per_day.mean(), 1))
    for pol in ("threshold70", "risk_myopic"):
        x = g[g.policy == pol].set_index("rep")
        dc = (x.cost_per_day / b.cost_per_day - 1) * 100
        row[f"{pol}_dcost"] = f"{dc.mean():+.1f} ± {ci(dc):.1f}"
        row[f"{pol}_overflow"] = f"{x.overflow_bins_per_day.mean():.1f} ± {ci(x.overflow_bins_per_day):.1f}"
    rows.append(row)
t3 = pd.DataFrame(rows).sort_values(["cv", "days_to_full"])
t3.to_csv(R / "regime_table_ci.csv", index=False)
print("\n== Regime map, 5 replications (95% CI, paired vs fixed daily)\n", t3.to_string(index=False))
