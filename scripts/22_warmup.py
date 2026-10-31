"""Warm-up adequacy (review R6): MSER-5 truncation points of the simulated day series.

For every run of the Tehran base experiment and the regime map, the daily series of social cost (operating
cost + cleanup + 60 USD per zone-weighted overflowing container) is batched in groups of 5 days and the
MSER-5 rule (White, 1997) picks the truncation point that minimises the squared standard error of the mean of
the remaining batches, searched over the first half of the run. The script compares it with the burn-in
actually discarded (one nominal fill cycle, at least 3 days).
usage: python scripts/22_warmup.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.config import MODEL_VERSION as V

RES = ROOT / "results"


def mser5(x):
    """Truncation point (days) of the MSER-5 rule, searched over the first half of the series."""
    n = len(x) // 5
    b = np.asarray(x[: 5 * n], float).reshape(n, 5).mean(axis=1)
    best, arg = np.inf, 0
    for d in range(0, n // 2 + 1):
        y = b[d:]
        v = ((y - y.mean()) ** 2).sum() / len(y) ** 2
        if v < best - 1e-12:
            best, arg = v, d
    return 5 * arg


def runs():
    t = pd.read_csv(RES / f"tehran_d6_{V}_r5_it5000" / "summary_by_rep.csv")
    for r in t.itertuples():
        yield "Tehran", RES / f"tehran_d6_{V}_r5_it5000" / f"days_cv{r.cv:g}_{r.capacity}_{r.policy}_rep{r.rep}.csv", r.burn_in
    g = pd.read_csv(RES / f"regime_map_{V}_n150_r8_it5000" / "summary_by_rep.csv")
    for r in g.itertuples():
        yield "regime map", RES / f"regime_map_{V}_n150_r8_it5000" / \
            f"days_m{r.rate_mult:g}_cv{r.cv:g}_{r.layout}_{r.policy}_rep{r.rep}.csv", r.burn_in


rows = []
for exp, f, burn in runs():
    d = pd.read_csv(f)
    x = (d.cost_fixed + d.cost_variable + d.cleanup_cost + 60.0 * d.overflow_weighted).to_numpy()
    rows.append(dict(experiment=exp, file=f.name, days=len(d), burn_in=int(burn), mser5=mser5(x)))
df = pd.DataFrame(rows)
df["covered"] = df.mser5 <= df.burn_in
df["excess"] = (df.mser5 - df.burn_in).clip(lower=0)
out = df.groupby("experiment").agg(runs=("file", "size"), mser5_zero=("mser5", lambda s: (s == 0).mean()),
                                   covered_by_burn_in=("covered", "mean"), mser5_median=("mser5", "median"),
                                   mser5_max=("mser5", "max"), excess_max=("excess", "max")).reset_index()
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
df.to_csv(RES / f"warmup_mser5_{V}.csv", index=False)
out.to_csv(RES / f"warmup_mser5_{V}_summary.csv", index=False)


# ------------------------------------------------------------------ do the conclusions change with a longer warm-up?
# Per cell, the warm-up is extended to the largest MSER-5 point of any run in it (a common window keeps the policies
# paired), leaving at least 10 evaluated days; families are re-tuned and the value of sensing recomputed.
from swc.experiments import loo_tuned, paired_ci, social_cost
from swc.simulate import DayRecord, summarise

df["cell"] = df.file.str.replace(r"_(?:fixed_daily|periodic\d|blind_threshold\d+|blind_risk|threshold\d+|risk_myopic|risk_cfa_t\d+)_rep\d+\.csv$",
                                 "", regex=True)
cell_burn = df.groupby("cell").apply(lambda g: int(max(g.burn_in.max(), g.mser5.max()))).to_dict()
res = []
for exp, folder, keys in (("Tehran", f"tehran_d6_{V}_r5_it5000", ["cv", "capacity"]),
                          ("regime map", f"regime_map_{V}_n150_r8_it5000", ["rate_mult", "cv", "layout"])):
    s = pd.read_csv(RES / folder / "summary_by_rep.csv")
    new = []
    for r in s.itertuples():
        f = (f"days_cv{r.cv:g}_{r.capacity}_{r.policy}_rep{r.rep}.csv" if exp == "Tehran" else
             f"days_m{r.rate_mult:g}_cv{r.cv:g}_{r.layout}_{r.policy}_rep{r.rep}.csv")
        d = pd.read_csv(RES / folder / f)
        recs = [DayRecord(**{k: (None if pd.isna(v) else v) for k, v in row.items()}) for row in d.to_dict("records")]
        cell = df.loc[df.file == f, "cell"].iloc[0]
        b = min(cell_burn[cell], len(recs) - 10)
        x = summarise(recs, burn_in=b)
        x.update({k: getattr(r, k) for k in keys}, policy=r.policy, rep=r.rep, sensor_cost_day=r.sensor_cost_day, burn=b)
        x["social_cost"] = social_cost(x, 60.0)
        new.append(x)
    new = pd.DataFrame(new)
    for tag, dd in (("as used", s), ("MSER-5 warm-up", new)):
        t = loo_tuned(dd, keys)
        for cell, g in t.groupby(keys):
            ns = g[g.tuned_family == "best_no_sensor"].set_index("rep")
            se = g[g.tuned_family == "best_sensor"].set_index("rep").loc[ns.index]
            v, h = paired_ci(ns.social_cost - se.social_cost - se.sensor_cost_day)
            res.append(dict(experiment=exp, cell=str(cell), warmup=tag, net=v, hw=h,
                            verdict="pays" if v - h > 0 else ("does not pay" if v + h < 0 else "inconclusive")))
R = pd.DataFrame(res)
W = R.pivot_table(index=["experiment", "cell"], columns="warmup", values="verdict", aggfunc="first")
W["same"] = W["as used"] == W["MSER-5 warm-up"]
print("\nverdicts unchanged:", int(W.same.sum()), "of", len(W))
print(W[~W.same].to_string())
R.to_csv(RES / f"warmup_mser5_{V}_verdicts.csv", index=False)
