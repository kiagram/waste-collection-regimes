"""Next-day fill forecasting and 'needs service' probability on real sensor data (Wyndham).

Task (evaluated only on days without an emptying between t and t+1, i.e. the counterfactual
'if not collected tonight', which is what the collection decision needs):
    given history up to night t, predict
    (a) the fill level at t+1 (point and quantiles), and
    (b) P(fill_{t+1} >= threshold), the event that the bin needs service / is at overflow risk.

Models, from simplest to AI:
    persistence      f_{t+1} = f_t ; P = 1[f_t >= thr]
    bin_empirical    f_t + increment drawn from the bin's own training-period empirical
                     distribution; this is the statistical fill model used by the optimiser
    lgbm             gradient-boosted trees with lags, calendar, spatial-neighbour and bin
                     features; classifier for (b) calibrated on a held-out period, quantile
                     regressors for (a) with split-conformal adjustment
Chronological split: train / calibration / test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THR = 0.8  # Wyndham 'fullnessThreshold' = 8 (tenths) for all but one bin

SPLITS = {"train": ("2018-06-26", "2019-12-31"), "calib": ("2020-01-01", "2020-06-30"),
          "test": ("2020-07-01", "2021-05-03")}


def make_features(p: pd.DataFrame, k_neighbours: int = 3, h: int = 1) -> pd.DataFrame:
    """One row per (bin, night t) with the target at t+h (no emptying in between). `p` = daily panel."""
    p = p.sort_values(["bin", "date"]).copy()
    g = p.groupby("bin")["fill"]
    for L in range(0, 8):
        p[f"lag{L}"] = g.shift(L)
    p["target"] = g.shift(-h)
    p["target_date_ok"] = p.groupby("bin")["date"].shift(-h) - p["date"] == pd.Timedelta(days=h)
    # 'no emptying before t+h': every intermediate reading observed and non-decreasing
    path_ok = pd.Series(True, index=p.index)
    prev = p["fill"]
    for j in range(1, h + 1):
        nxt = g.shift(-j)
        path_ok &= nxt.notna() & (nxt >= prev)
        prev = nxt
    p["collected_next"] = ~path_ok
    p["d1"] = p["lag0"] - p["lag1"]
    p["d2"] = p["lag1"] - p["lag2"]
    p["mean7"] = p[[f"lag{L}" for L in range(7)]].mean(axis=1)
    # days since last observed drop (proxy for days since emptying)
    drop = (p["lag0"] < p["lag1"]).astype(float).where(p["lag0"].notna() & p["lag1"].notna())
    p["_drop"] = drop.fillna(0.0)
    p["_grp"] = p.groupby("bin")["_drop"].cumsum()
    p["days_since_drop"] = p.groupby(["bin", "_grp"]).cumcount()
    p["dow"] = p["date"].dt.dayofweek
    p["month"] = p["date"].dt.month
    p["is_recycling"] = (p["stream"] == "recycling").astype(int)
    # spatial neighbours: mean current fill of the k nearest other bins (same night)
    xy = p.groupby("bin")[["lon", "lat"]].first()
    lat0 = np.deg2rad(xy["lat"].mean())
    P = np.c_[xy["lon"] * np.cos(lat0), xy["lat"]] * 111_320.0
    dm = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(dm, np.inf)
    nn = {b: list(xy.index[np.argsort(dm[i])[:k_neighbours]]) for i, b in enumerate(xy.index)}
    wide = p.pivot(index="date", columns="bin", values="lag0")
    nbr = {b: wide[nn[b]].mean(axis=1) for b in wide.columns}
    p["nbr_fill"] = [nbr[b].get(d, np.nan) for b, d in zip(p["bin"], p["date"])]
    p = p.drop(columns=["_drop", "_grp"])
    keep = p["target"].notna() & p["lag0"].notna() & p["target_date_ok"] & ~p["collected_next"]
    return p[keep].reset_index(drop=True)


FEATURES = ["lag0", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "d1", "d2", "mean7",
            "days_since_drop", "dow", "month", "is_recycling", "nbr_fill", "bin_rate"]


def split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = SPLITS[name]
    return df[(df["date"] >= a) & (df["date"] <= b)]


def add_bin_rate(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Bin mean daily increment estimated on the training period only (no leakage)."""
    inc = (train["target"] - train["lag0"]).groupby(train["bin"]).mean()
    df = df.copy()
    df["bin_rate"] = df["bin"].map(inc)
    return df


# ------------------------------------------------------------------ baseline models
def persistence(df):
    return df["lag0"].to_numpy(), (df["lag0"] >= THR).astype(float).to_numpy()


def bin_empirical(df, train):
    inc = (train["target"] - train["lag0"]).groupby(train["bin"]).apply(lambda s: s.to_numpy())
    point = np.empty(len(df))
    prob = np.empty(len(df))
    for i, (b, f) in enumerate(zip(df["bin"], df["lag0"])):
        x = inc.get(b, np.array([0.0]))
        nxt = np.minimum(f + x, 1.0)
        point[i] = np.median(nxt)
        prob[i] = np.mean(nxt >= THR - 1e-9)
    return point, prob


# ------------------------------------------------------------------ metrics
def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logloss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def diebold_mariano(l1, l2, h: int = 1):
    """DM test on loss series l1 - l2 (positive mean -> model 2 better). HAC variance, lag h-1."""
    from scipy.stats import norm
    d = np.asarray(l1) - np.asarray(l2)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        var += 2 * np.cov(d[k:], d[:-k], ddof=0)[0, 1]
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * norm.sf(abs(stat)))
