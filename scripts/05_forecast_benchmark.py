"""Forecast benchmark on the Wyndham smart-bin data (real sensors).

Compares persistence, the bin-empirical statistical model, and LightGBM (calibrated classifier
+ conformalised quantile regression) for next-day fill and P(fill >= 0.8).
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

from swc.forecast import (FEATURES, THR, add_bin_rate, bin_empirical, brier, diebold_mariano, ece, logloss,
                          make_features, persistence, split)
from swc.wyndham import clean, load_raw, panel

H = int(sys.argv[1]) if len(sys.argv) > 1 else 1
p = panel(clean(load_raw()))
df = make_features(p, h=H)
tr0 = split(df, "train")
df = add_bin_rate(df, tr0)
tr, ca, te = split(df, "train"), split(df, "calib"), split(df, "test")
print(f"horizon h={H} days; rows: train {len(tr)}, calib {len(ca)}, test {len(te)}; test event rate {np.mean(te.target >= THR):.3f}")

y_te = (te["target"] >= THR - 1e-9).astype(float).to_numpy()
res, preds = {}, {}

pt, pp = persistence(te)
preds["persistence"] = (pt, pp)
pt, pp = bin_empirical(te, tr)
preds["bin_empirical"] = (pt, pp)

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
for base in ("persistence", "bin_empirical"):
    s, pv = diebold_mariano(sq(preds[base][1]), sq(preds["lgbm"][1]))
    print(f"DM test Brier {base} vs lgbm: stat={s:.2f}, p={pv:.2g} (positive -> lgbm better)")

imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nfeature importance (splits):", imp.head(8).to_dict())
out = ROOT / "results" / ("forecast_benchmark" if H == 1 else f"forecast_benchmark_h{H}")
out.mkdir(exist_ok=True)
tab.to_csv(out / "metrics.csv")
(out / "extra.json").write_text(json.dumps({"conformal_coverage": cover, "conformal_width": float(np.mean(hi - lo)),
                                            "n": {"train": len(tr), "calib": len(ca), "test": len(te)},
                                            "event_rate_test": float(y_te.mean()),
                                            "importance": imp.to_dict()}, indent=1))
# test-set probabilities (reliability diagrams)
pd.DataFrame({"y": y_te, **{k: v[1] for k, v in preds.items()}}).to_csv(out / "test_probs.csv", index=False)
print("saved", out)
