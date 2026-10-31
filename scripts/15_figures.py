"""Publication figures (v4).

Reads only files in results/ and data/processed/ (no re-simulation). Writes PNG (300 dpi) and
PDF (vector) to results/figures/. Colour-blind-safe palette (Okabe-Ito).
  fig1_tehran_d6          District 6 instance: containers by zone, collection points by days-to-full (--map)
  fig2_value_of_sensing   saving of the best sensor policy over the best no-sensor policy per container-day,
                          against the sensor-network cost, across fill speeds (synthetic district)
  fig3_frontiers          cost-overflow frontiers of the policy families at 2.8 and 5.6 days to full
  fig4_tehran_options     cost vs overflowing containers for the District 6 options (CV 1.0 and 0.35)
  fig5_forecast_calib     reliability diagrams on the Wyndham data, thresholds 0.8 and 1.0, h = 1 and 3
  fig6_fleet_tradeoff     fleet cost-CO2 trade-off and the (in)effect of a routing carbon price
  fig7_exact_fronts       exact cost-CO2 fronts on six District 6 sub-districts
usage: python scripts/15_figures.py [--map] [--no-roads]
"""
import json
import sys
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


# ------------------------------------------------------------------ 1. District 6 map
def fig1(roads=True):
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
                  fontsize=6.5, title="Modelled containers by zone (OSM)", title_fontsize=7)
    axs[0].set_title(f"(a) One modelled 1,100 L container per alley: {len(bxy):,}")
    pts = xy[1:-1]
    sc = axs[1].scatter(pts[:, 0], pts[:, 1], s=4 + 1.6 * m, c=1 / rate, cmap="viridis_r", vmin=1.5, vmax=4.0,
                        edgecolor="k", lw=0.25, zorder=2)
    axs[1].scatter(*xy[0], marker="s", s=70, color="#E69F00", edgecolor="k", zorder=3, label="Depot (municipality)")
    axs[1].scatter(*xy[-1], marker="^", s=80, color="#CC0000", edgecolor="k", zorder=3,
                   label="Exit to transfer station (+6 km)")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, frameon=False, fontsize=6.5)
    axs[1].set_title(f"(b) {len(pts)} k-means clusters (size = containers)")
    cb = fig.colorbar(sc, ax=axs[1], shrink=0.7, pad=0.01)
    cb.set_label("Days to fill (mean of containers in cluster)", fontsize=7)
    fig.text(0.01, 0.01, "Map data © OpenStreetMap contributors (ODbL). District area 21.4 km², 235 km of alleys. "
             "Clusters with more than 15 containers are split into co-located routing points (323 in total).",
             fontsize=6, color="#555555")
    fig.tight_layout()
    save(fig, "fig1_tehran_d6")


# ------------------------------------------------------------------ 2. value of sensing across regimes
REG = RES / f"regime_map_{V}_n150_r8_it5000"


def regime_value():
    """Per cell: saving in social cost of the best sensor policy over the best policy without sensors,
    per container-day (before the sensor network), with the paired 95% interval; and the network cost."""
    t = pd.read_csv(REG / "tuned_by_rep.csv")
    rows = []
    for (m, cv, lay), g in t.groupby(["rate_mult", "cv", "layout"]):
        ns = g[g.tuned_family == "best_no_sensor"].set_index("rep")
        se = g[g.tuned_family == "best_sensor"].set_index("rep").loc[ns.index]
        sav = (ns.social_cost - se.social_cost) / se.bins
        rows.append(dict(days=g.days_to_full.mean(), cv=cv, layout=lay, sav=sav.mean(), sav_ci=ci(sav),
                         unit=float((se.sensor_cost_day / se.bins).mean()), reps=len(sav)))
    return pd.DataFrame(rows).sort_values("days")


def fig2():
    s = regime_value()
    unit = s.unit.mean()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    for ax, cv, title in ((axs[0], 1.0, "High variability (CV 1.0, calibrated)"),
                          (axs[1], 0.35, "Low variability (CV 0.35)")):
        ax.axhspan(0.5 * unit, 1.5 * unit, color="#F2F2F2", zorder=0)
        ax.axhline(unit, color="k", lw=1, ls="--", zorder=1)
        ax.axhline(0, color=C["grey"], lw=0.6, zorder=1)
        for lay, mk, col in (("clustered", "o", C["risk"]), ("uniform", "s", C["threshold"])):
            d = s[(s.cv == cv) & (s.layout == lay)]
            ax.errorbar(d.days, d.sav, d.sav_ci, color=col, marker=mk, ms=4, capsize=2, lw=1.3,
                        label=f"{lay} district")
        ax.axvline(D6_DAYS, color=C["red"], lw=1, ls=":", zorder=1)
        ax.set(title=title, xlabel="Days for a container to fill (mean)", xlim=(2.0, 13.5))
    axs[0].set_ylabel("Saving of the best sensor policy over the\nbest no-sensor policy (USD per container-day)")
    lo, hi = axs[0].get_ylim()
    axs[0].text(D6_DAYS + 0.12, hi - 0.05 * (hi - lo), "Tehran\nDistrict 6", color=C["red"], fontsize=6.5, va="top")
    axs[1].text(13.3, unit * 1.55, f"sensor-network cost {unit:.3f} USD per container-day\n(dashed; grey band ±50%)",
                ha="right", va="bottom", fontsize=6.3)
    axs[1].legend(frameon=False, loc="lower right")
    for ax, t in zip(axs, "ab"):
        tag(ax, t)
    fig.tight_layout()
    save(fig, "fig2_value_of_sensing")


# ------------------------------------------------------------------ 3. cost-overflow frontiers
def fig3():
    p = pd.read_csv(RES / f"frontiers_{V}_n150_cv1_r3_it5000" / "frontier_points.csv")
    fams = [("periodic", "Daily / every k days (no sensors)", C["fixed"], "o"),
            ("blind_risk", "Risk model without readings", C["grey"], "v"),
            ("threshold", "Threshold rule (sensors)", C["threshold"], "s"),
            ("risk", "Risk-priced routing (sensors)", C["risk"], "D")]
    days = {1.0: 2.8, 0.5: 5.6}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
    for ax, m in zip(axs, (1.0, 0.5)):
        g = p[p.rate_mult == m]
        for fam, lab, col, mk in fams:
            x = g[g.family == fam].sort_values("overflow_bins_per_day")
            ax.plot(x.overflow_bins_per_day, x.op_cost, color=col, marker=mk, ms=4, lw=0.9, label=lab,
                    mec="k", mew=0.3)
        fd = g[g.policy == "fixed_daily"].iloc[0]
        ax.annotate("daily", (fd.overflow_bins_per_day, fd.op_cost), xytext=(6, 4), textcoords="offset points",
                    fontsize=6.5)
        ax.set(title=f"{days[m]} days to full (clustered district, CV 1.0)", xlabel="Overflowing containers per day",
               ylabel="Operating and cleanup cost (USD/day)", xscale="log")
        ax.grid(alpha=0.25, lw=0.5, which="both")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    for ax, t in zip(axs, "ab"):
        tag(ax, t)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig3_frontiers")


# ------------------------------------------------------------------ 4. District 6 options
def tehran_options():
    t = pd.read_csv(RES / f"tehran_d6_{V}_r5_it5000" / "tuned_by_rep.csv")
    capex = 2852 * 150.0 / (8 * 365)
    t["money"] = (t.cost_per_day + t.cleanup_cost_per_day + np.where(t.sensing.astype(bool), t.sensor_cost_day, 0.0)
                  + np.where(t.capacity == "double", capex, 0.0))
    return t


def fig4():
    t = tehran_options()
    opts = [("current", "fixed_daily", "Today: daily collection", C["fixed"], "o"),
            ("current", "best_sensor", "Best sensor policy", C["risk"], "o"),
            ("current", "threshold70", "Threshold 70% (sensors)", C["threshold"], "o"),
            ("double", "best_no_sensor", "Double containers, best without sensors", C["double"], "D"),
            ("double", "best_sensor", "Double containers, best sensor policy", C["double_s"], "D"),
            ("double", "threshold70", "Double containers, threshold 70%", C["double_t"], "D")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.2))
    for ax, cv, title in ((axs[0], 1.0, "High variability (CV 1.0, calibrated)"),
                          (axs[1], 0.35, "Low variability (CV 0.35)")):
        g = t[t.cv == cv]
        for cap, fam, lab, col, mk in opts:
            x = g[(g.capacity == cap) & (g.tuned_family == fam)]
            ax.errorbar(x.money.mean(), x.overflow_bins_per_day.mean(), xerr=ci(x.money),
                        yerr=ci(x.overflow_bins_per_day), color=col, marker=mk, ms=6, capsize=2, ls="none",
                        label=lab, mec="k", mew=0.4)
        ax.set(title=title, xlabel="Cost incl. cleanup, sensors and containers (USD/day)",
               ylabel="Overflowing containers per day")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks([0, 5, 10, 20, 50, 100, 200, 500])
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=0.25, lw=0.5)
    for ax, s_ in zip(axs, "ab"):
        tag(ax, s_)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig4_tehran_options")


# ------------------------------------------------------------------ 5. forecast calibration
def reliability(y, p, nb=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    k = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return (np.array([p[k == i].mean() for i in range(len(q) - 1) if (k == i).any()]),
            np.array([y[k == i].mean() for i in range(len(q) - 1) if (k == i).any()]))


def fig5():
    from swc.forecast import brier, ece
    fig, axs = plt.subplots(2, 2, figsize=(W2 * 0.8, 5.6), sharex=True, sharey=True)
    for r, thr in enumerate((0.8, 1.0)):
        for c, h in enumerate((1, 3)):
            ax = axs[r, c]
            d = pd.read_csv(RES / f"forecast_benchmark_{V}_h{h}_thr{thr:g}" / "test_probs.csv")
            y = d.y.to_numpy()
            txt = ["Brier / ECE"]
            ax.plot([0, 1], [0, 1], color=C["grey"], lw=0.8, ls=":")
            for col, lab, short, cc, ls in (("gamma_bin", "Operational Gamma model", "Gamma", C["risk"], "-"),
                                            ("bin_empirical", "Per-bin empirical", "Empirical", C["double"], "--"),
                                            ("lgbm", "LightGBM + isotonic", "LightGBM", C["threshold"], "-")):
                p_ = d[col].to_numpy()
                mp, fy = reliability(y, p_)
                ax.plot(mp, fy, marker="o", ms=3, color=cc, ls=ls, label=lab)
                txt.append(f"{short}: {brier(y, p_):.3f} / {ece(y, p_):.3f}")
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.set_title(f"P(fill ≥ {thr:g}), {h} day{'s' if h > 1 else ''} ahead (n = {len(y):,})", fontsize=8, loc="right")
            ax.set_aspect("equal")
            ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=6)
            tag(ax, "abcd"[2 * r + c])
    for ax in axs[1]:
        ax.set_xlabel("Predicted probability")
    for ax in axs[:, 0]:
        ax.set_ylabel("Observed frequency")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig5_forecast_calib")


# ------------------------------------------------------------------ 6. fleet trade-off
def fig6():
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
    axs[0].set(xlabel="Operational CO₂ per night (kg)", ylabel="Operating cost per night (USD)",
               title="Fleet composition (daily collection)")
    pad = 0.08 * (s0.co2_kg.max() - s0.co2_kg.min())
    axs[0].set_xlim(s0.co2_kg.min() - pad, s0.co2_kg.max() + 2 * pad)
    span = s0.cost.max() - s0.cost.min()
    axs[0].set_ylim(s0.cost.min() - 0.25 * span, s0.cost.max() + 0.15 * span)
    for f in ("current_ageing", "all_new"):
        g = s[s.fleet == f].sort_values("carbon_price")
        axs[1].plot(g.carbon_price, g.co2_kg, marker="o", ms=3.5, color=cols[f], label=names[f])
    axs[1].set(xlabel="Carbon price inside routing (USD/t CO₂)", ylabel="Operational CO₂ per night (kg)",
               title="Routing-level carbon price", ylim=(0, 2300))
    axs[1].legend(frameon=False, loc="center right")
    tag(axs[0], "a")
    tag(axs[1], "b")
    fig.tight_layout()
    save(fig, "fig6_fleet_tradeoff")


# ------------------------------------------------------------------ 7. exact fronts, District 6 sub-districts
def fig7():
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
        h = h.assign(kc=h.cost.round(0), ke=h.co2_kg.round(0))
        hh = h.groupby(["kc", "ke"]).agg(cost=("cost", "mean"), co2_kg=("co2_kg", "mean"),
                                           min=("carbon_price", "min"), max=("carbon_price", "max")).reset_index()
        ax.scatter(hh.co2_kg, hh.cost, marker="x", s=40, color=C["threshold"], lw=1.2, zorder=5,
                   label="Heuristic (PyVRP) with carbon price")
        for j, (_, r) in enumerate(hh.sort_values("co2_kg").iterrows()):
            lab = f"λ {r['min']:g}" + (f"–{r['max']:g}" if r["max"] > r["min"] else "")
            ax.annotate(lab, (r.co2_kg, r.cost), xytext=(4, 4 + 9 * (j % 2)), textcoords="offset points",
                        fontsize=5.5, color=C["threshold"])
        ax.set_yscale("log")
        ax.set_title(f"Sub-district {i + 1} ({full['info']['bins']} containers, {full['info']['demand_t']:.1f} t)",
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
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    if "--map" in sys.argv:   # the District 6 instance does not change with the simulations
        fig1(roads="--no-roads" not in sys.argv)
