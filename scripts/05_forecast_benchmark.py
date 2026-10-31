"""Forecast benchmark on the Wyndham smart-bin data (real sensors).

Compares persistence, the bin-empirical statistical model, the optimiser's operational Gamma/Normal
model, and LightGBM (calibrated classifier + conformalised quantile regression) for the h-day-ahead
fill and P(fill >= thr). thr = 0.8 is Wyndham's service threshold; thr = 1.0 (full) is the closest
observable proxy for overflow.

Tests of forecast accuracy account for the panel: Diebold-Mariano on date-averaged Brier
differences with a Newey-West variance (lag >= h - 1), and a moving-block bootstrap over dates
(all bins of a date kept together) for the mean Brier difference with a 95% interval.
usage: python scripts/05_forecast_benchmark.py [h] [thr] [drop_tol]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from swc.forecast import (FEATURES, add_bin_rate, bin_empirical, block_bootstrap_ci, brier, diebold_mariano,
                          dm_panel, ece, exclusion_report, logloss, make_features, parametric_bin, persistence,
                          split)
from swc.wyndham import clean, load_raw, panel

H = int(sys.argv[1]) if len(sys.argv) > 1 else 1
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
p = panel(clean(load_raw()))
allrows = make_features(p, h=H, drop_tol=TOL, return_all=True)
excl = exclusion_report(allrows)
print("excluded as emptied before t+h:", excl["total"].round(3).to_dict("records"))
print(excl["by_fill"].round(3).to_string())
df = make_features(p, h=H, drop_tol=TOL)
tr0 = split(df, "train")
df = add_bin_rate(df, tr0)
tr, ca, te = split(df, "train"), split(df, "calib"), split(df, "test")
print(f"horizon h={H} days; rows: train {len(tr)}, calib {len(ca)}, test {len(te)}; test event rate {np.mean(te.target >= THR):.3f}")

y_te = (te["target"] >= THR - 1e-9).astype(float).to_numpy()
res, preds = {}, {}

preds["persistence"] = persistence(te, THR)
preds["bin_empirical"] = bin_empirical(te, tr, THR)
preds["gamma_bin"] = parametric_bin(te, tr, THR, "gamma")
preds["normal_bin"] = parametric_bin(te, tr, THR, "normal")

# LightGBM classifier, isotonic calibration on the calibration period
params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
              n_jobs=1, deterministic=True, force_row_wise=True)
clf = lgb.LGBMClassifier(**params).fit(tr[FEATURES], (tr["target"] >= THR - 1e-9).astype(int))
iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(ca[FEATURES])[:, 1],
                                                   (ca["target"] >= THR - 1e-9).astype(float))
p_raw = clf.predict_proba(te[FEATURES])[:, 1]
p_cal = iso.predict(p_raw)

# LightGBM median + conformalised 10-90% interval (CQR, Romano et al. 2019)
qs = {}
for a in (0.1, 0.5, 0.9):
    qs[a] = lgb.LGBMRegressor(objective="quantile", alpha=a, **params).fit(tr[FEATURES], tr["target"])
lo_c, hi_c = qs[0.1].predict(ca[FEATURES]), qs[0.9].predict(ca[FEATURES])
score = np.maximum(lo_c - ca["target"].to_numpy(), ca["target"].to_numpy() - hi_c)
qhat = np.quantile(score, np.ceil((len(score) + 1) * 0.8) / len(score))
lo, hi = qs[0.1].predict(te[FEATURES]) - qhat, qs[0.9].predict(te[FEATURES]) + qhat
cover = float(np.mean((te["target"] >= lo) & (te["target"] <= hi)))
preds["lgbm"] = (np.clip(qs[0.5].predict(te[FEATURES]), 0, 1), p_cal)
preds["lgbm_uncalibrated"] = (preds["lgbm"][0], p_raw)

yt = te["target"].to_numpy()
for name, (pt, pp) in preds.items():
    res[name] = dict(MAE=float(np.mean(np.abs(pt - yt))), Brier=brier(y_te, pp), LogLoss=logloss(y_te, pp),
                     AUC=float(roc_auc_score(y_te, pp)) if len(np.unique(pp)) > 1 else float("nan"),
                     ECE=ece(y_te, pp))
tab = pd.DataFrame(res).T
print(tab.round(4).to_string())
print(f"\nconformal 80% interval coverage on test: {cover:.3f} (target 0.80), mean width {np.mean(hi - lo):.3f}")

sq = lambda pp: (pp - y_te) ** 2
tests = []
dates = te["date"].to_numpy()
for base in ("persistence", "bin_empirical", "gamma_bin", "normal_bin"):
    l1, l2 = sq(preds[base][1]), sq(preds["lgbm"][1])
    s_ts, p_ts = diebold_mariano(l1, l2, h=H)                    # pooled rows, kept for comparison only
    s_pn, p_pn, dbar, lag, ndates = dm_panel(l1, l2, dates, h=H)
    est, lo, hi, p_bs = block_bootstrap_ci(l1, l2, dates)
    tests.append(dict(base=base, brier_diff=est, ci_lo=lo, ci_hi=hi, p_block_bootstrap=p_bs,
                      dm_panel_stat=s_pn, dm_panel_p=p_pn, dm_panel_lag=lag, dates=ndates,
                      dm_pooled_stat=s_ts, dm_pooled_p=p_ts))
    print(f"Brier {base} - lgbm = {est:+.4f} [{lo:+.4f}, {hi:+.4f}] (block bootstrap p={p_bs:.3g}); "
          f"panel DM stat={s_pn:.2f} p={p_pn:.2g} (lag {lag}, {ndates} dates); pooled DM p={p_ts:.2g}")
tests = pd.DataFrame(tests)

imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nfeature importance (splits):", imp.head(8).to_dict())
out = ROOT / "results" / (f"forecast_benchmark_v4_h{H}_thr{THR:g}" + ("" if TOL == 0 else f"_tol{TOL:g}"))
out.mkdir(exist_ok=True)
tab.to_csv(out / "metrics.csv")
tests.to_csv(out / "tests.csv", index=False)
for k, v in excl.items():
    v.to_csv(out / f"exclusions_{k}.csv")
(out / "extra.json").write_text(json.dumps({"conformal_coverage": cover, "conformal_width": float(np.mean(hi - lo)),
                                            "n": {"train": len(tr), "calib": len(ca), "test": len(te)},
                                            "event_rate_test": float(y_te.mean()),
                                            "importance": imp.to_dict()}, indent=1))
# test-set probabilities (reliability diagrams)
pd.DataFrame({"bin": te["bin"].to_numpy(), "date": te["date"].to_numpy(), "y": y_te,
              **{k: v[1] for k, v in preds.items()}}).to_csv(out / "test_probs.csv", index=False)
print("saved", out)
