"""Manuscript tables (v4), written as Markdown to results/tables_v4/ from the result files only.

  table3_regime.md       value of sensing across fill speeds, variability and layouts (06, 08)
  table4_tehran.md       District 6 options (09)
  table5_forecast.md     forecasting on the Wyndham data (05)
  table6_fronts.md       exact cost-CO2 fronts on the six sub-districts (16)
  table7_robustness.md   robustness of the District 6 conclusions (12, 13, 18, 20)
usage: python scripts/21_tables.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION as V
from swc.experiments import paired_ci

sys.stdout.reconfigure(encoding="utf-8")

RES = ROOT / "results"
OUT = RES / "tables_v4"
OUT.mkdir(exist_ok=True)
CAPEX = 2852 * 150.0 / (8 * 365)       # second container in every alley, USD/day
NAMES = {"fixed_daily": "daily", "periodic2": "every 2 days", "periodic3": "every 3 days", "periodic4": "every 4 days",
         "periodic6": "every 6 days", "threshold50": "threshold 50%", "threshold70": "threshold 70%",
         "threshold90": "threshold 90%", "risk_myopic": "risk, myopic", "risk_cfa_t5": "risk, look-ahead θ = 5",
         "risk_cfa_t15": "risk, look-ahead θ = 15", "blind_risk": "risk model without readings",
         "blind_threshold50": "elapsed time 50%", "blind_threshold70": "elapsed time 70%",
         "blind_threshold90": "elapsed time 90%"}


def nm(choice):
    return " / ".join(NAMES.get(c, c) for c in str(choice).split("/"))


def fmt(x, d=0):
    return f"{x:,.{d}f}".replace("-", "−")


def pm(m, h, d=0):
    return f"{fmt(m, d)} ± {h:,.{d}f}"


def cell(v):
    import re
    v = str(v).replace("_compactor", " compactor").replace("small_loader", "small loader")
    return re.sub(r"(?<![\w.])-(?=\d)", "−", v)      # typographic minus for negative numbers


def write(name, title, df, note=None):
    lines = [f"Table: {title}", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    if note:
        lines += ["", note]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:3 + len(df) + 1]), "\n")


# ------------------------------------------------------------------ Table 3
ABBR = {"fixed_daily": "daily", "periodic2": "P2", "periodic3": "P3", "periodic4": "P4", "periodic6": "P6",
        "threshold50": "TH50", "threshold70": "TH70", "threshold90": "TH90", "risk_myopic": "RM",
        "risk_cfa_t5": "RC5", "risk_cfa_t15": "RC15", "blind_risk": "RB", "blind_threshold50": "E50",
        "blind_threshold70": "E70", "blind_threshold90": "E90"}


def ab(choice):
    return "/".join(ABBR.get(c, c) for c in str(choice).split("/"))


def table3():
    t = pd.read_csv(RES / f"regime_map_{V}_n150_r8_it5000" / "tuned_by_rep.csv")
    rows = []
    for (m, cv, lay), g in t.groupby(["rate_mult", "cv", "layout"]):
        ns = g[g.tuned_family == "best_no_sensor"].set_index("rep")
        se = g[g.tuned_family == "best_sensor"].set_index("rep").loc[ns.index]
        fd = g[g.tuned_family == "fixed_daily"].set_index("rep").loc[ns.index]
        sav, hw = paired_ci(ns.social_cost - se.social_cost)
        op, hwo = paired_ci(ns.cost_per_day - se.cost_per_day)
        sc = float(se.sensor_cost_day.mean())
        verdict = "pays" if sav - hw > sc else ("does not pay" if sav + hw < sc else "inconclusive")
        rows.append(dict(cv=cv, layout=lay, days=g.days_to_full.mean(), reps=len(ns),
                         fd=f"{fmt(fd.cost_per_day.mean())} ({fmt(fd.overflow_bins_per_day.mean(), 1)})",
                         ns=f"{ab(ns.chosen.mode().iloc[0])} ({fmt(ns.overflow_bins_per_day.mean(), 1)})",
                         se=f"{ab(se.chosen.mode().iloc[0])} ({fmt(se.overflow_bins_per_day.mean(), 1)})",
                         op=pm(op, hwo), soc=pm(sav, hw), net=fmt(sav - sc), verdict=verdict))
    d = pd.DataFrame(rows).sort_values(["cv", "layout", "days"], ascending=[False, True, True])
    d["days"] = [f"{x:.1f}" + ("†" if r == 3 else "") for x, r in zip(d.days, d.reps)]
    d["cv"] = d.cv.map(lambda x: f"{x:g}")
    d["layout"] = d.layout.map({"clustered": "clust.", "uniform": "unif."})
    d = d.drop(columns="reps")
    d.columns = ["CV", "District", "Days to full", "Daily: cost (overflows)", "Best without sensors (overflows)",
                 "Best with sensors (overflows)", "Operating saving", "Social-cost saving", "Net of sensors",
                 "Sensing"]
    write("table3_regime.md", "Table 3. Value of sensing across fill regimes on the synthetic district (150 points, "
          "1,028–1,087 containers; USD per day; overflows = overflowing containers per day; ± 95% interval of paired "
          "replication differences; 8 replications, † 3). Savings are those of the best sensor policy over the best "
          "policy without sensors, each tuned leave-one-replication-out; the sensor network costs 66–68 USD per day. "
          "Policies: daily; Pk every k days; THτ threshold τ%; RM risk-based, myopic; RCθ risk-based with look-ahead θ.",
          d)


# ------------------------------------------------------------------ Table 4
def table4():
    t = pd.read_csv(RES / f"tehran_d6_{V}_r5_it5000" / "tuned_by_rep.csv")
    t["money"] = (t.cost_per_day + t.cleanup_cost_per_day + np.where(t.sensing.astype(bool), t.sensor_cost_day, 0.0)
                  + np.where(t.capacity == "double", CAPEX, 0.0))
    t["net_social"] = t.social_cost + np.where(t.sensing.astype(bool), t.sensor_cost_day, 0.0) \
        + np.where(t.capacity == "double", CAPEX, 0.0)
    rows = []
    opts = [("current", "fixed_daily", "Today"), ("current", "best_sensor", "Sensors, best"),
            ("current", "threshold70", "Sensors, TH70"),
            ("double", "best_no_sensor", "2× containers, best without sensors"),
            ("double", "best_sensor", "2× containers + sensors, best"),
            ("double", "threshold70", "2× containers + sensors, TH70")]
    for cv in (1.0, 0.35):
        g = t[t.cv == cv]
        base = g[(g.capacity == "current") & (g.tuned_family == "fixed_daily")].set_index("rep")
        for cap, fam, lab in opts:
            x = g[(g.capacity == cap) & (g.tuned_family == fam)].set_index("rep").loc[base.index]
            d, h = paired_ci(x.net_social - base.net_social)
            rows.append({"CV": f"{cv:g}", "Option": lab, "Policy": ab(x.chosen.mode().iloc[0]),
                         "Operating cost": fmt(x.cost_per_day.mean()), "Truck-shifts": fmt(x.truck_shifts_per_day.mean(), 1),
                         "Cost with cleanup, sensors, containers": fmt(x.money.mean()),
                         "Overflows per day": fmt(x.overflow_bins_per_day.mean(), 1),
                         "Spill (t/day)": fmt(x.spill_t_per_day.mean(), 1),
                         "Social cost": fmt(x.net_social.mean()),
                         "Δ social cost vs today": "–" if (cap, fam) == ("current", "fixed_daily") else pm(d, h)})
    write("table4_tehran.md", "Table 4. Tehran District 6 options (5 replications; USD per day; ± 95% interval of "
          "the paired difference against today's practice, daily collection with current containers). Overflows = "
          "overflowing containers per day. Social cost = operating cost + cleanup (30 USD/t spilled) + 60 USD per "
          "zone-weighted overflowing container + sensors and additional containers. Policies as in Table 3.", pd.DataFrame(rows))


# ------------------------------------------------------------------ Table 5
def table5():
    rows = []
    for thr in (0.8, 1.0):
        for h in (1, 2, 3):
            d = RES / f"forecast_benchmark_{V}_h{h}_thr{thr:g}"
            m = pd.read_csv(d / "metrics.csv", index_col=0)
            ts = pd.read_csv(d / "tests.csv").set_index("base")
            e = json.loads((d / "extra.json").read_text())
            ex = pd.read_csv(d / "exclusions_total.csv")["share"].iloc[0]
            g = ts.loc["gamma_bin"]
            pf = lambda x: "<0.001" if x < 0.001 else f"{x:.2g}"
            rows.append({"Event; horizon": f"≥ {thr:g}; {h} d", "Test rows (excluded)": f"{e['n']['test']:,} ({100 * ex:.0f}%)",
                         "Persistence": f"{m.loc['persistence', 'Brier']:.3f}",
                         "Empirical": f"{m.loc['bin_empirical', 'Brier']:.3f} / {m.loc['bin_empirical', 'ECE']:.3f}",
                         "Gamma": f"{m.loc['gamma_bin', 'Brier']:.3f} / {m.loc['gamma_bin', 'ECE']:.3f}",
                         "LightGBM": f"{m.loc['lgbm', 'Brier']:.3f} / {m.loc['lgbm', 'ECE']:.3f}",
                         "Gamma − LightGBM [95% CI]": f"{g.brier_diff:+.4f} [{g.ci_lo:+.4f}, {g.ci_hi:+.4f}]",
                         "p (bootstrap; DM)": f"{pf(g.p_block_bootstrap)}; {pf(g.dm_panel_p)}",
                         "80% coverage": f"{e['conformal_coverage']:.3f}"})
    write("table5_forecast.md", "Table 5. Forecasts of P(fill ≥ threshold at the horizon) on the Wyndham test period "
          "(July 2020–May 2021), on observed paths without an inferred emptying (event rates 0.17–0.22 for ≥ 0.8 and "
          "0.05–0.07 for full). Cells: Brier score / expected calibration error. Differences in Brier score: moving-block "
          "bootstrap over dates (14-day blocks, all bins of a date kept) and Diebold–Mariano (DM) test on date-averaged "
          "losses with Newey–West variance. Coverage: conformal 80% intervals.", pd.DataFrame(rows))


# ------------------------------------------------------------------ Table 6
def table6():
    base = RES / f"exact_fronts_tehran_{V}_full_s6_n12_g6_tl120"
    rows = []
    for i in range(6):
        d = json.loads((base / f"sub{i}.json").read_text())
        f = sorted(d["exact"], key=lambda e: e["cost"])
        a = f[0]
        r = {"Sub-district": i + 1, "Containers; waste (t)": f"{d['info']['bins']}; {d['info']['demand_t']:.1f}",
             "Cost-minimal plan: cost; CO₂ (kg); trucks": f"{a['cost']:.1f}; {a['co2_kg']:.1f}; {' + '.join(a['trucks'])}",
             "Points proven optimal": f"{sum(e['exact'] for e in f)}/{len(f)}"}
        if len(f) > 1:
            b = f[1]
            dco2, dcost = b["co2_kg"] / a["co2_kg"] - 1, b["cost"] / a["cost"] - 1
            r.update({"First step towards lower CO₂": f"{' + '.join(a['trucks'])} → {' + '.join(b['trucks'])}",
                      "Δ CO₂": f"{100 * dco2:+.1f}%", "Δ cost": f"{100 * dcost:+.1f}%",
                      "Abatement (USD/t)": f"{(b['cost'] - a['cost']) / ((a['co2_kg'] - b['co2_kg']) / 1000):,.0f}"})
        else:
            r.update({"First step towards lower CO₂": "none (cheapest plan is also the cleanest)", "Δ CO₂": "–",
                      "Δ cost": "–", "Abatement (USD/t)": "–"})
        rows.append(r)
    write("table6_fronts.md", "Table 6. Exact cost–CO₂ fronts on six District 6 sub-districts (12 collection points "
          "each; every point served). The first step is the first efficient point after the cost-minimal plan.",
          pd.DataFrame(rows))


# ------------------------------------------------------------------ Table 7
def table7():
    S = pd.read_csv(RES / f"tehran_sensitivity_{V}_r3_it5000" / "summary.csv")
    B = pd.read_csv(RES / f"heuristic_quality_{V}" / "bias_tehran_sensitivity.csv").set_index(["scenario", "capacity"])
    s = S[S.family == "best_sensor"].set_index(["scenario", "capacity"])
    labels = {"gen0.7": "Waste generation −30%", "gen1.3": "Waste generation +30% (fleet saturated)",
              "leg3": "Transfer distance 3 km", "leg12": "Transfer distance 12 km",
              "fixed0.5": "Truck fixed cost −50%", "fixed1.5": "Truck fixed cost +50%",
              "km0.5": "Truck cost per km −50%", "km1.5": "Truck cost per km +50%",
              "spill_lost": "Spilled waste leaves the system (draft accounting)",
              "split6": "Finer aggregation: ≤ 6 containers per point, independent filling, 100 m apart",
              "targeted": "Second container only at the faster-filling half", "large": "One 2.2 m³ container per alley",
              "alpha_tight": "Tolerances 0.02–0.25", "alpha_loose": "Tolerances 0.10–0.70",
              "normal": "Normal instead of Gamma tails", "draft_prize": "Penalty on full skip risk (draft formulation)"}
    rows = []
    for (scen, cap), r in s.iterrows():
        v = "sensing worse" if r.d_net - r.d_net_hw > 0 else ("sensing better" if r.d_net + r.d_net_hw < 0 else "inconclusive")
        g = {"sensing does not pay under any gap": "does not pay", "sensing pays under any gap": "pays",
             "depends on the gap": "depends"}[B.loc[(scen, cap), "verdict"]]
        rows.append({"Scenario (re-optimised, CV 1.0)": labels[scen], "Containers": cap,
                     "Best without sensors": ab(S[(S.scenario == scen) & (S.capacity == cap) & (S.family == "best_no_sensor")].chosen.iloc[0]),
                     "Net cost of sensing": pm(r.d_net, r.d_net_hw), "Verdict": v,
                     "With worst-case engine gap": g})
    write("table7_robustness.md", "Table 7. Robustness of the District 6 sensing verdict (3 replications per scenario; "
          "net cost of the best sensor policy, sensors included, minus that of the best policy without sensors, USD per "
          "day, ± 95% interval; positive = sensing worse). Last column: verdict when operating costs are corrected by "
          "the measured engine gaps in the direction least favourable to it (Section 6.1).", pd.DataFrame(rows))


if __name__ == "__main__":
    table3()
    table4()
    table5()
    table6()
    table7()
