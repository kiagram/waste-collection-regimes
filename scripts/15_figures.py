"""Publication figures for the article / supervisor meeting pack.

Reads only files in results/ and data/processed/ (no re-simulation). Writes PNG (300 dpi) and
PDF (vector) to results/figures/. Colour-blind-safe palette (Okabe-Ito).
  fig1_regime_map         savings and overflows vs days-to-full, by variability (synthetic district)
  fig2_break_even         maximum justified sensor spend per bin-day vs the assumed network cost
  fig3_tehran_d6          District 6 instance: bins by zone, collection points by days-to-full
  fig4_tehran_options     total cost vs overflow volume for the District 6 options (CV 1.0 and 0.35)
  fig5_fleet_tradeoff     fleet cost-CO2 trade-off and the (in)effect of a routing carbon price
  fig6_forecast_calib     reliability diagrams on the Wyndham sensor data, h = 1 and 3 days
usage: python scripts/15_figures.py [--no-roads]
"""
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

be = import_module("08_break_even")
from swc.config import MODEL_VERSION as V
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = dict(fixed="#000000", threshold="#E69F00", risk="#0072B2", double="#009E73", double_s="#56B4E9",
         double_t="#D55E00", grey="#999999", red="#CC79A7")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
W2, W1 = 7.48, 3.54   # double / single column width (inches), Elsevier 190 / 90 mm
D6_DAYS = 2.28        # District 6 days-to-full = 1 / mean daily fill fraction (the regime-map definition)


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def tag(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


# ------------------------------------------------------------------ 1. regime map
def regime_stats():
    r = pd.read_csv(RES / f"regime_map_{V}_n150_r5_d20_it5000" / "summary_by_rep.csv")
    rows = []
    for (m, cv), g in r.groupby(["rate_mult", "cv"]):
        b = g[g.policy == "fixed_daily"].set_index("rep")
        for pol in ("fixed_daily", "threshold70", "risk_myopic"):
            x = g[g.policy == pol].set_index("rep")
            dc = (x.cost_per_day / b.cost_per_day - 1) * 100
            sav = (b.cost_per_day - x.cost_per_day) / 1039      # USD per bin-day (1,039 bins, see 08_break_even)
            rows.append(dict(days=g.days_to_full.mean(), cv=cv, policy=pol, dcost=dc.mean(), dcost_ci=ci(dc),
                             ovf=x.overflow_bins_per_day.mean(), ovf_ci=ci(x.overflow_bins_per_day),
                             sav=sav.mean(), sav_ci=ci(sav)))
    return pd.DataFrame(rows).sort_values("days")


def fig1():
    s = regime_stats()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
    pols = [("threshold70", "Threshold 70%", C["threshold"]), ("risk_myopic", "Risk-based (proposed)", C["risk"]),
            ("fixed_daily", "Fixed daily (today)", C["fixed"])]
    for ax in axs:
        ax.axvspan(2.0, 3.6, color="#F2F2F2", zorder=0)
        ax.axvline(D6_DAYS, color=C["red"], lw=1, ls=":", zorder=1)
    for pol, lab, col in pols:
        for cv, ls, mk in ((1.0, "-", "o"), (0.35, "--", "s")):
            d = s[(s.policy == pol) & (s.cv == cv)]
            if pol != "fixed_daily":
                axs[0].errorbar(d.days, -d.dcost, d.dcost_ci, color=col, ls=ls, marker=mk, ms=4, capsize=2, lw=1.3,
                                mfc=col if cv == 1.0 else "white")
            axs[1].errorbar(d.days, d.ovf, d.ovf_ci, color=col, ls=ls, marker=mk, ms=4, capsize=2, lw=1.3,
                            mfc=col if cv == 1.0 else "white")
    axs[0].set(xlabel="Days for a bin to fill (mean)", ylabel="Cost saving vs fixed daily (%)", ylim=(-2, 52),
               title="Operating-cost saving")
    axs[1].set(xlabel="Days for a bin to fill (mean)", ylabel="Overflowing bins per day (of 1,039)",
               title="Service: overflows")
    axs[0].text(2.8, 49, "fast-filling\nregime", ha="center", va="top", fontsize=7, color="#555555")
    axs[0].text(D6_DAYS - 0.08, 30, "Tehran District 6", rotation=90, color=C["red"], fontsize=7, ha="right", va="center")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=1.5, label=l) for _, l, c in pols]
    h += [Line2D([], [], color="k", ls="-", marker="o", ms=4, label="high variability (CV 1.0, calibrated)"),
          Line2D([], [], color="k", ls="--", marker="s", mfc="white", ms=4, label="low variability (CV 0.35)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    for ax, t in zip(axs, "ab"):
        tag(ax, t)
        ax.set_xlim(2.0, 8.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig1_regime_map")


# ------------------------------------------------------------------ 2. break-even
def tehran_runs():
    d = pd.read_csv(RES / f"tehran_d6_{V}_r5_d20_it5000" / "summary_by_rep.csv")
    bins = {"current": 2852, "double": 5704}
    capex = []
    for cap, pol in zip(d.capacity, d.policy):
        c = 2852 * 150.0 / (8 * 365) if cap == "double" else 0.0
        capex.append(c + (be.sensor_cost_per_day(bins[cap], 21.4) if pol != "fixed_daily" else 0.0))
    d["total"] = d.cost_per_day + np.array(capex)
    d["ovf_m3"] = 1.1 * d.overflow_volume_bins_per_day
    return d


def fig2():
    s = regime_stats()
    unit = be.sensor_cost_per_day(1039, (5.0 * (150 / 40) ** 0.5 / 1.6) ** 2) / 1039
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 2.9))
    ax.axhspan(0.5 * unit, 1.5 * unit, color="#F2F2F2", zorder=0)
    ax.axhline(unit, color="k", lw=1, ls="--")
    ax.text(8.7, 0.5 * unit - 0.004, f"assumed sensor-network cost {unit:.3f} USD/bin-day\n(dashed; grey band ±50%)",
            ha="right", va="top", fontsize=6.3)
    for pol, lab, col in (("risk_myopic", "Risk-based", C["risk"]), ("threshold70", "Threshold 70%", C["threshold"])):
        for cv, ls, mk in ((1.0, "-", "o"), (0.35, "--", "s")):
            d = s[(s.policy == pol) & (s.cv == cv)]
            ax.errorbar(d.days, d.sav, d.sav_ci, color=col, ls=ls, marker=mk, ms=4, capsize=2, lw=1.3,
                        mfc=col if cv == 1.0 else "white", label=f"{lab}, CV {cv:g}")
    # District 6: operating saving of the risk policy (look-ahead) per bin-day, current bins, CV 1.0
    t = tehran_runs()
    t = t[(t.cv == 1.0) & (t.capacity == "current")]
    b = t[t.policy == "fixed_daily"].set_index("rep").cost_per_day
    x = t[t.policy == "risk_cfa_t5"].set_index("rep").cost_per_day
    sv = (b - x) / 2852
    ax.errorbar([D6_DAYS], [sv.mean()], [ci(sv)], color=C["red"], marker="*", ms=10, capsize=2, ls="none",
                label="Tehran District 6, risk-based")
    ax.set(xlabel="Days for a bin to fill (mean)", ylabel="Break-even sensor cost (USD per bin-day)",
           xlim=(2.0, 8.8), ylim=(-0.05, 0.34))
    ax.legend(loc="upper left", frameon=False, fontsize=6.3, title="Sensors pay off above the dashed line",
              title_fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig2_break_even")


# ------------------------------------------------------------------ 3. District 6 map
def fig3(roads=True):
    z = np.load(ROOT / "data" / "processed" / "tehran_d6.npz", allow_pickle=True)
    bxy, bz, xy, m, rate = z["bins_xy"], z["bins_zone"], z["xy"], z["m"], z["rate_day"]
    edges = None
    if roads:
        try:
            from swc.tehran import CRS, _ox
            ox = _ox()
            poly = ox.geocode_to_gdf("District 6, Tehran, Iran").geometry.iloc[0]
            G = ox.graph_from_polygon(poly, network_type="drive", simplify=True, truncate_by_edge=True)
            edges = ox.graph_to_gdfs(ox.project_graph(G, to_crs=CRS), nodes=False)
            import geopandas as gpd
            border = gpd.GeoSeries([poly], crs=4326).to_crs(CRS)
        except Exception as e:  # offline: draw without the road network
            print("road network unavailable:", e)
            edges = None
    fig, axs = plt.subplots(1, 2, figsize=(W2, 4.3))
    zc = {"residential": "#0072B2", "recreational": "#009E73", "political": "#D55E00", "historical": "#CC79A7",
          "commercial": "#E69F00"}
    for ax in axs:
        if edges is not None:
            edges.plot(ax=ax, color="#D9D9D9", lw=0.4, zorder=0)
            border.boundary.plot(ax=ax, color="k", lw=0.8, zorder=1)
        ax.set_aspect("equal")
        ax.set_axis_off()
    for zn, col in zc.items():
        k = bz == zn
        axs[0].scatter(bxy[k, 0], bxy[k, 1], s=2.2, color=col, lw=0, label=f"{zn} ({k.sum():,})", zorder=2)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, markerscale=4,
                  fontsize=6.5, title="Bins by zone (OSM)", title_fontsize=7)
    axs[0].set_title(f"(a) One 1,100 L bin per alley: {len(bxy):,} bins")
    pts = xy[1:-1]
    sc = axs[1].scatter(pts[:, 0], pts[:, 1], s=4 + 1.6 * m, c=1 / rate, cmap="viridis_r", vmin=1.5, vmax=4.0,
                        edgecolor="k", lw=0.25, zorder=2)
    axs[1].scatter(*xy[0], marker="s", s=70, color="#E69F00", edgecolor="k", zorder=3, label="Depot (municipality)")
    axs[1].scatter(*xy[-1], marker="^", s=80, color="#CC0000", edgecolor="k", zorder=3,
                   label="Exit to transfer station (+6 km)")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, frameon=False, fontsize=6.5)
    axs[1].set_title(f"(b) {len(pts)} collection points (size = bins)")
    cb = fig.colorbar(sc, ax=axs[1], shrink=0.7, pad=0.01)
    cb.set_label("Days to fill (mean of bins at point)", fontsize=7)
    fig.text(0.01, 0.01, "Map data © OpenStreetMap contributors (ODbL). District area 21.4 km², 235 km of alleys.",
             fontsize=6, color="#555555")
    fig.tight_layout()
    save(fig, "fig3_tehran_d6")


# ------------------------------------------------------------------ 4. District 6 options
def fig4():
    d = tehran_runs()
    opts = [("current", "fixed_daily", "Today: daily collection", C["fixed"], "o"),
            ("current", "risk_cfa_t5", "Sensors, risk-based", C["risk"], "o"),
            ("current", "threshold70", "Sensors, threshold 70%", C["threshold"], "o"),
            ("double", "fixed_daily", "Double bins, daily", C["double"], "D"),
            ("double", "risk_cfa_t5", "Double bins + sensors, risk-based", C["double_s"], "D"),
            ("double", "threshold70", "Double bins + sensors, threshold", C["double_t"], "D")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.2))
    for ax, cv, title in ((axs[0], 1.0, "High variability (CV 1.0, calibrated)"),
                          (axs[1], 0.35, "Low variability (CV 0.35)")):
        g = d[d.cv == cv]
        for cap, pol, lab, col, mk in opts:
            x = g[(g.capacity == cap) & (g.policy == pol)]
            ax.errorbar(x.total.mean(), x.ovf_m3.mean(), xerr=ci(x.total), yerr=ci(x.ovf_m3), color=col,
                        marker=mk, ms=6, capsize=2, ls="none", label=lab, mec="k", mew=0.4)
        ax.set(title=title, xlabel="Total cost incl. sensors and bins (USD/day)",
               ylabel="Overflow volume (m³/day)")
        ax.grid(alpha=0.25, lw=0.5)
    g = d[d.cv == 1.0]
    mean = lambda cap, pol: g[(g.capacity == cap) & (g.policy == pol)][["total", "ovf_m3"]].mean()
    t0, dd, ss = mean("current", "fixed_daily"), mean("double", "fixed_daily"), mean("current", "risk_cfa_t5")
    axs[0].annotate(f"double capacity:\n{100 * (dd.ovf_m3 / t0.ovf_m3 - 1):+.0f}% overflow",
                    xy=(dd.total - 15, dd.ovf_m3 + 6), xytext=(t0.total - 250, 60), fontsize=6.5,
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    axs[0].annotate(f"sensors alone:\n{100 * (ss.total / t0.total - 1):+.0f}% cost, no gain",
                    xy=(ss.total, ss.ovf_m3 + 5), xytext=(ss.total + 20, ss.ovf_m3 + 50), fontsize=6.5,
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    tag(axs[0], "a")
    tag(axs[1], "b")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig4_tehran_options")


# ------------------------------------------------------------------ 5. fleet trade-off
def fig5():
    s = pd.read_csv(RES / f"fleet_tradeoff_tehran_{V}" / "summary.csv")
    names = {"all_old": "All old", "current_ageing": "Current (ageing)", "half_renewed": "Half renewed",
             "all_new": "All new"}
    cols = {"all_old": "#555555", "current_ageing": C["fixed"], "half_renewed": C["risk"], "all_new": C["double"]}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
    s0 = s[s.carbon_price == 0].set_index("fleet").loc[list(names)]
    axs[0].plot(s0.co2_kg, s0.cost, color=C["grey"], lw=0.8, zorder=1)
    off = {"all_old": (-8, -4, "right"), "current_ageing": (-8, -4, "right"), "half_renewed": (8, -4, "left"),
           "all_new": (8, 4, "left")}
    for f, r in s0.iterrows():
        axs[0].scatter(r.co2_kg, r.cost, s=45, color=cols[f], edgecolor="k", lw=0.4, zorder=2)
        dx, dy, ha = off[f]
        axs[0].annotate(f"{names[f]}\n{int(r.trucks)} trucks", (r.co2_kg, r.cost), xytext=(dx, dy), ha=ha,
                        va="top" if dy < 0 else "bottom", textcoords="offset points", fontsize=6.5)
    axs[0].set(xlabel="CO₂ per night (kg)", ylabel="Operating cost per night (USD)",
               title="Fleet composition (full collection)")
    pad = 0.08 * (s0.co2_kg.max() - s0.co2_kg.min())
    axs[0].set_xlim(s0.co2_kg.min() - pad, s0.co2_kg.max() + 2 * pad)
    axs[0].set_ylim(s0.cost.min() - 0.25 * (s0.cost.max() - s0.cost.min()), s0.cost.max() + 0.15 * (s0.cost.max() - s0.cost.min()))
    for f in ("current_ageing", "all_new"):
        g = s[s.fleet == f].sort_values("carbon_price")
        axs[1].plot(g.carbon_price, g.co2_kg, marker="o", ms=3.5, color=cols[f], label=names[f])
    axs[1].set(xlabel="Carbon price inside routing (USD/t CO₂)", ylabel="CO₂ per night (kg)",
               title="Routing-level carbon price", ylim=(0, 2300))
    axs[1].legend(frameon=False, loc="center right")
    axs[1].text(0.03, 0.06, "Flat: the carbon lever is the fleet, not the route", transform=axs[1].transAxes,
                fontsize=6.5, color="#555555")
    tag(axs[0], "a")
    tag(axs[1], "b")
    fig.tight_layout()
    save(fig, "fig5_fleet_tradeoff")


# ------------------------------------------------------------------ 6. forecast calibration
def reliability(y, p, nb=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    k = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return (np.array([p[k == i].mean() for i in range(len(q) - 1) if (k == i).any()]),
            np.array([y[k == i].mean() for i in range(len(q) - 1) if (k == i).any()]))


def fig6():
    from swc.forecast import brier, ece
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 3.0), sharey=True)
    for ax, h, folder in ((axs[0], 1, "forecast_benchmark"), (axs[1], 3, "forecast_benchmark_h3")):
        d = pd.read_csv(RES / folder / "test_probs.csv")
        y = d.y.to_numpy()
        txt = ["Brier / ECE"]
        ax.plot([0, 1], [0, 1], color=C["grey"], lw=0.8, ls=":")
        for col, lab, short, c, ls in (("bin_empirical", "Per-bin statistical", "Per-bin", C["risk"], "-"),
                                       ("lgbm", "LightGBM + isotonic", "LightGBM", C["threshold"], "-"),
                                       ("lgbm_uncalibrated", "LightGBM, uncalibrated", "LightGBM raw",
                                        C["threshold"], ":")):
            p = d[col].to_numpy()
            mp, fy = reliability(y, p)
            ax.plot(mp, fy, marker="o", ms=3, color=c, ls=ls, label=lab)
            txt.append(f"{short}: {brier(y, p):.3f} / {ece(y, p):.3f}")
        ax.set(xlabel="Predicted P(bin ≥ 80% full)", title=f"Horizon {h} day{'s' if h > 1 else ''} (n = {len(y):,})",
               xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
        ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=6)
    axs[0].set_ylabel("Observed frequency")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    tag(axs[0], "a")
    tag(axs[1], "b")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig6_forecast_calib")


# ------------------------------------------------------------------ 7. exact fronts, District 6 sub-districts
def fig7():
    import json
    base = RES / ("exact_fronts_tehran_" + V + "_{}_s6_n12_g6_tl120")
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.6))
    for i, ax in enumerate(axs.flat):
        full = json.loads((Path(str(base).format("full")) / f"sub{i}.json").read_text())
        sel = json.loads((Path(str(base).format("selective")) / f"sub{i}.json").read_text())
        s = pd.DataFrame(sel["exact"]).sort_values("co2_kg")
        f = pd.DataFrame(full["exact"]).sort_values("co2_kg")
        h = pd.DataFrame(sel["heuristic"])
        ax.step(s.co2_kg, s.cost, where="post", color=C["grey"], lw=1, ls="--", zorder=1)
        ax.scatter(s.co2_kg, s.cost, s=14, facecolor="white", edgecolor=C["grey"], zorder=2,
                   label="Exact, selective (may skip points)")
        ax.step(f.co2_kg, f.cost, where="post", color=C["risk"], lw=1.3, zorder=3)
        ax.scatter(f.co2_kg, f.cost, s=22, color=C["risk"], edgecolor="k", lw=0.3, zorder=4,
                   label="Exact, full collection")
        # merge heuristic runs that give (almost) the same plan; plot their mean position
        h = h.assign(kc=h.cost.round(0), ke=h.co2_kg.round(0))
        hh = h.groupby(["kc", "ke"]).agg(cost=("cost", "mean"), co2_kg=("co2_kg", "mean"),
                                           min=("carbon_price", "min"), max=("carbon_price", "max")).reset_index()
        ax.scatter(hh.co2_kg, hh.cost, marker="x", s=40, color=C["threshold"], lw=1.2, zorder=5,
                   label="Heuristic (PyVRP) with carbon price")
        for j, (_, r) in enumerate(hh.sort_values("co2_kg").iterrows()):
            lab = f"λ {r['min']:g}" + (f"–{r['max']:g}" if r["max"] > r["min"] else "")
            ax.annotate(lab, (r.co2_kg, r.cost), xytext=(4, 4 + 9 * (j % 2)), textcoords="offset points", fontsize=5.5,
                        color=C["threshold"])
        ax.set_yscale("log")
        ax.set_title(f"Sub-district {i + 1} ({full['info']['bins']} bins, {full['info']['demand_t']:.1f} t)",
                     fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.tick_params(labelsize=6)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks([100, 150, 200, 300, 500])
        ax.set_ylim(95, 560)
    for ax in axs[1]:
        ax.set_xlabel("CO₂ per night (kg)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Cost per night (USD, log)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig7_exact_fronts")


if __name__ == "__main__":
    fig7()
    fig1()
    fig2()
    fig4()
    fig5()
    fig6()
    fig3(roads="--no-roads" not in sys.argv)
