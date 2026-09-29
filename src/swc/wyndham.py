"""Wyndham City (Melbourne) smart-bin fill data: cleaning, collection inference, calibration.

Source: Wyndham City Council, data.gov.au, CC BY 2.5 AU (see data/raw/SOURCE.md).
Facts established by profiling (see Paper_notes_2025_2026.md):
* 33 bins, daily snapshots 2018-06-26 .. 2021-05-03, date only (no time of day);
* fill reported on ordinal levels 0, 2, 4, 6, 8, 10 (tenths of capacity);
* bin 1511202 is permanently in ALERT and is excluded;
* missed days are back-filled in bulk -> duplicate (bin, date) rows;
* no collection events are recorded; they are inferred from drops in fill level.
Cleaning rules used here (document them in the paper):
* ALERT with level 0 is treated as a sensor fault (missing); ALERT with level 10 is kept (full);
* duplicates (bin, date): keep the last record.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
EXCLUDE = {1511202}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW / "wyndham_smartbin_filllevel.json"
    f = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(**ft["properties"], lon=ft["geometry"]["coordinates"][0], lat=ft["geometry"]["coordinates"][1])
            for ft in f["features"]]
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["timestamp"])
    d = d.rename(columns={"serialNumber": "bin", "latestFullness": "level"})
    d["stream"] = np.where(d["description"].str.contains(" R ", regex=False), "recycling", "general")
    return d


def clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d[~d["bin"].isin(EXCLUDE)].copy()
    d = d.drop_duplicates(["bin", "date"], keep="last")
    fault = (d["reason"] == "ALERT") & (d["level"] == 0)
    d["fill"] = np.where(fault, np.nan, d["level"] / 10.0)
    return d.sort_values(["bin", "date"]).reset_index(drop=True)


def panel(d: pd.DataFrame) -> pd.DataFrame:
    """Complete daily grid (bin x date) with NaN for missing days; static attributes attached."""
    dates = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    bins = sorted(d["bin"].unique())
    idx = pd.MultiIndex.from_product([bins, dates], names=["bin", "date"])
    p = d.set_index(["bin", "date"])[["fill"]].reindex(idx).reset_index()
    static = d.groupby("bin").agg(lon=("lon", "first"), lat=("lat", "first"), stream=("stream", "first"),
                                  description=("description", "first"))
    return p.merge(static, left_on="bin", right_index=True)


def transitions(p: pd.DataFrame, drop_tol: float = 0.0) -> pd.DataFrame:
    """Consecutive-day pairs (both observed) with inferred collection and increment.

    collected : fill dropped (f_t < f_{t-1} - drop_tol) -> the bin was emptied between readings
    increment : f_t - f_{t-1} if not collected, else f_t (growth since the emptying; a lower bound)
    censored  : f_{t-1} == 1 or f_t == 1 (the true increment is unobservable above capacity)
    """
    p = p.sort_values(["bin", "date"]).copy()
    g = p.groupby("bin")
    p["prev"] = g["fill"].shift(1)
    p["prev_date"] = g["date"].shift(1)
    t = p[(p["date"] - p["prev_date"] == pd.Timedelta(days=1)) & p["fill"].notna() & p["prev"].notna()].copy()
    t["collected"] = t["fill"] < t["prev"] - drop_tol
    t["increment"] = np.where(t["collected"], t["fill"], t["fill"] - t["prev"])
    t["censored"] = (t["prev"] >= 1.0) | (t["fill"] >= 1.0)
    return t


def calibration(t: pd.DataFrame) -> pd.DataFrame:
    """Per-bin daily increment statistics from uncensored, non-collection transitions."""
    u = t[~t["collected"] & ~t["censored"]]
    s = u.groupby("bin")["increment"].agg(n="size", mean="mean", sd="std", p_zero=lambda x: (x == 0).mean())
    s["cv"] = s["sd"] / s["mean"].where(s["mean"] > 0)
    c = t.groupby("bin").agg(days=("date", "size"), collections=("collected", "sum"))
    s = s.join(c)
    s["days_between_collections"] = s["days"] / s["collections"].clip(lower=1)
    return s


def latent_dispersion(p: pd.DataFrame, max_k: int = 6, min_windows: int = 30) -> pd.DataFrame:
    """Separate true day-to-day variability from sensor quantisation (0.2 steps).

    For windows of k consecutive observed days with no inferred collection and no censoring,
    the observed change is  sum of k true increments + (rounding error at end - at start), so
        E[change_k]   = k * mu
        Var[change_k] = k * sigma^2 + c      (c ~ 2 * 0.2^2 / 12 if rounding errors are uniform)
    Regressing Var[change_k] on k per bin gives sigma^2 (slope) and c (intercept).
    """
    out = []
    for b, g in p.sort_values("date").groupby("bin"):
        f = g["fill"].to_numpy()
        n = len(f)
        rows = []
        for k in range(1, max_k + 1):
            ch = []
            for s in range(n - k):
                w = f[s:s + k + 1]
                if np.isnan(w).any() or (w >= 1.0).any():
                    continue
                if (np.diff(w) < 0).any():          # an emptying inside the window
                    continue
                ch.append(w[-1] - w[0])
            if len(ch) >= min_windows:
                rows.append((k, np.mean(ch), np.var(ch, ddof=1), len(ch)))
        if len(rows) < 3:
            continue
        k, m, v, cnt = map(np.array, zip(*rows))
        mu = np.polyfit(k, m, 1)[0]
        slope, icpt = np.polyfit(k, v, 1)
        out.append(dict(bin=b, mu=mu, sigma=np.sqrt(max(slope, 0.0)), quant_c=icpt, k_max=int(k.max()),
                        windows=int(cnt.sum())))
    r = pd.DataFrame(out).set_index("bin")
    r["cv_latent"] = r["sigma"] / r["mu"].where(r["mu"] > 0)
    return r
