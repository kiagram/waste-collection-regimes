"""Next-day fill forecasting and 'needs service' probability on real sensor data (Wyndham).

Task (evaluated only on observed paths without an emptying between t and t+h; these approximate the
counterfactual 'if not collected tonight' only for the bins the historical process left alone,
see make_features and exclusion_report):
    given history up to night t, predict
    (a) the fill level at t+1 (point and quantiles), and
    (b) P(fill_{t+1} >= threshold), the event that the bin needs service / is at overflow risk.

Models, from simplest to AI:
    persistence      f_{t+1} = f_t ; P = 1[f_t >= thr]
    bin_empirical    f_t + increment drawn from the bin's own training-period empirical
                     distribution (nonparametric statistical model)
    gamma_bin        the optimiser's operational model: per-bin Gamma increment with the training
                     mean and sd (risk.py); normal_bin is its Normal variant
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


def make_features(p: pd.DataFrame, k_neighbours: int = 3, h: int = 1, drop_tol: float = 0.0,
                  return_all: bool = False) -> pd.DataFrame:
    """One row per (bin, night t) with the target at t+h (no emptying in between). `p` = daily panel.

    Rows whose path t..t+h is not fully observed and non-decreasing (up to `drop_tol`, to absorb sensor
    noise) are treated as emptied and excluded. These are selected outcomes of the historical
    collection process, not the counterfactual 'not collected' paths of every bin; `exclusion_report`
    describes what is dropped. return_all=True returns every row with the flags instead.
    """
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
        path_ok &= nxt.notna() & (nxt >= prev - drop_tol)
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
    p["observed"] = p["target"].notna() & p["lag0"].notna() & p["target_date_ok"]
    keep = p["observed"] & ~p["collected_next"]
    if return_all:
        return p.reset_index(drop=True)
    return p[keep].reset_index(drop=True)


def exclusion_report(all_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Share of observed (bin, night) rows excluded as 'emptied before t+h', by bin and by current fill."""
    o = all_rows[all_rows["observed"]].copy()
    o["excluded"] = o["collected_next"].astype(bool)
    o["fill_band"] = pd.cut(o["lag0"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0-0.2", "0.2-0.4", "0.4-0.6",
                                                                               "0.6-0.8", "0.8-1"])
    agg = lambda g: g["excluded"].agg(["size", "sum", "mean"]).rename(columns={"size": "rows", "sum": "excluded",
                                                                             "mean": "share"})
    return {"by_bin": agg(o.groupby("bin")), "by_fill": agg(o.groupby("fill_band", observed=False)),
            "total": agg(o.assign(all="all").groupby("all"))}


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
def persistence(df, thr: float = THR):
    return df["lag0"].to_numpy(), (df["lag0"] >= thr - 1e-9).astype(float).to_numpy()


def bin_empirical(df, train, thr: float = THR):
    inc = (train["target"] - train["lag0"]).groupby(train["bin"]).apply(lambda s: s.to_numpy())
    point = np.empty(len(df))
    prob = np.empty(len(df))
    for i, (b, f) in enumerate(zip(df["bin"], df["lag0"])):
        x = inc.get(b, np.array([0.0]))
        nxt = np.minimum(f + x, 1.0)
        point[i] = np.median(nxt)
        prob[i] = np.mean(nxt >= thr - 1e-9)
    return point, prob


def _bin_increments(train: pd.DataFrame) -> pd.Series:
    return (train["target"] - train["lag0"]).groupby(train["bin"])


def parametric_bin(df, train, thr: float, dist: str = "gamma"):
    """The optimiser's operational fill model: per-bin increment over the horizon with the training
    mean and sd, Gamma (as simulated and as used by risk.py) or Normal; P(f + X >= thr)."""
    from .risk import prob_exceeds
    g = _bin_increments(train)
    mu_b, sd_b = g.mean().clip(lower=1e-3), g.std().fillna(0.0).clip(lower=1e-3)
    mu = df["bin"].map(mu_b).fillna(float(mu_b.mean())).to_numpy()
    sd = df["bin"].map(sd_b).fillna(float(sd_b.mean())).to_numpy()
    f = df["lag0"].to_numpy()
    prob = prob_exceeds(thr - 1e-9 - f, mu, sd, dist)
    point = np.minimum(f + mu, 1.0)
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
    """DM test on one loss series l1 - l2 (positive mean -> model 2 better). HAC variance, lag h-1.

    Valid for a single time series. For the bin panel use `dm_panel`, which handles the
    within-date (spatial) dependence across bins; this function ignores it.
    """
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


def _date_series(d, dates):
    s = pd.DataFrame({"d": np.asarray(d, dtype=float), "date": pd.to_datetime(np.asarray(dates))})
    g = s.groupby("date")["d"].agg(["sum", "size"]).sort_index()
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")   # keep calendar spacing for the HAC lags
    return g.reindex(full)


def dm_panel(l1, l2, dates, h: int = 1, lag: int | None = None):
    """Panel Diebold-Mariano test: loss differences are averaged over bins on each date and the date
    series is tested with a Newey-West (Bartlett) variance. Averaging within a date absorbs the
    spatial correlation between bins; the HAC lag covers the serial correlation of overlapping
    h-step forecasts. Default lag: max(h - 1, floor(4 (T/100)^(2/9))) (Newey and West, 1994).
    Returns (stat, p, mean difference per row, lag used, number of dates)."""
    from scipy.stats import norm
    g = _date_series(np.asarray(l1) - np.asarray(l2), dates)
    x = (g["sum"] / g["size"]).dropna().to_numpy()   # gaps (dates without rows) are skipped
    T = len(x)
    L = lag if lag is not None else max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    u = x - x.mean()
    var = u @ u / T
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * (u[k:] @ u[:-k]) / T
    stat = x.mean() / np.sqrt(var / T)
    d_row = float(g["sum"].sum() / g["size"].sum())
    return float(stat), float(2 * norm.sf(abs(stat))), d_row, L, T


def block_bootstrap_ci(l1, l2, dates, block: int = 14, B: int = 2000, level: float = 0.95, seed: int = 0):
    """Moving-block bootstrap over calendar dates of the mean loss difference per row; every bin of a
    resampled date is kept, so within-date dependence is preserved. Returns (mean, lo, hi, p two-sided)."""
    g = _date_series(np.asarray(l1) - np.asarray(l2), dates).fillna(0.0)
    S, N = g["sum"].to_numpy(), g["size"].to_numpy()
    T = len(S)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :T]
    stat = S[idx].sum(axis=1) / np.maximum(N[idx].sum(axis=1), 1)
    est = S.sum() / N.sum()
    a = (1 - level) / 2
    lo, hi = np.quantile(stat, [a, 1 - a])
    p = float(np.mean(np.abs(stat - est) >= abs(est)))   # centred bootstrap: H0 mean difference = 0
    return float(est), float(lo), float(hi), p
