"""Shared machinery of the v4 simulation experiments: horizons, the policy grid, resumable runs,
social cost and leave-one-out tuning.

Policy families (review point R1). The value of sensing is measured against the best policy that
does not read fill levels, not against daily collection:
    no sensors : fixed_daily, periodic{k}, blind_threshold{tau}, blind_risk
    sensors    : threshold{tau}, risk_myopic, risk_cfa_t{theta}
Every family's parameter is tuned on the same objective, the social cost per day
    operating cost + cleanup of spilled waste + c_o x zone-weighted overflowing containers,
with leave-one-replication-out selection, so no family is tuned on the runs it is scored on.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Params
from .instance import generate_synthetic
from .simulate import (BlindRisk, BlindThreshold, DayRecord, FixedDaily, FixedPeriodic, RiskCFA, RiskMyopic,
                       Threshold, simulate)

FAMILY = {"fixed_daily": "periodic", "periodic": "periodic", "blind_threshold": "blind_threshold",
          "blind_risk": "blind_risk", "threshold": "threshold", "risk_myopic": "risk", "risk_cfa": "risk"}
SENSING = {"periodic": False, "blind_threshold": False, "blind_risk": False, "threshold": True, "risk": True}


def family(policy_name: str) -> str:
    for prefix, fam in sorted(FAMILY.items(), key=lambda kv: -len(kv[0])):
        if policy_name.startswith(prefix):
            return fam
    raise KeyError(policy_name)


def regime_instance(n: int, mult: float, cv: float, layout: str, rep: int, overflow_cost: float = 60.0):
    """Synthetic district of the regime map: n points, fill speed x mult, a new geometry per replication.
    Returns (instance, nominal days to full)."""
    from dataclasses import replace
    P = Params()
    P.risk = replace(P.risk, overflow_cost=overflow_cost, fill_rate_cv=cv)
    fleet = {"new_compactor": max(1, n // 60), "old_compactor": max(2, n // 30), "small_loader": max(1, n // 60)}
    seed = (100 if layout == "clustered" else 200) + rep
    inst = generate_synthetic(n, seed=seed, params=P, layout=layout, bins_per_point=(4, 10), fleet_counts=fleet,
                              area_m=5000.0 * (n / 40) ** 0.5 / 1.6)
    inst.rate_mean_per_h = inst.rate_mean_per_h * mult
    inst.rate_sd_per_h = inst.rate_sd_per_h * mult
    return inst, 1.0 / float(np.mean(inst.rate_mean_per_h * 24))


def horizon(days_to_full: float, cycles: float = 3.0, min_measured: int = 20) -> tuple[int, int]:
    """(burn-in days, total days): burn-in of one nominal fill cycle (at least 3 days), then at least
    `cycles` fill cycles (and at least `min_measured` days) of measurement."""
    burn = max(3, math.ceil(days_to_full))
    return burn, burn + max(min_measured, math.ceil(cycles * days_to_full))


def policy_grid(days_to_full: float) -> list:
    """The tuning grid. Periods longer than 1.5 nominal fill cycles are left out (they overflow
    almost every point and cannot be competitive)."""
    pols = [FixedDaily()]
    pols += [FixedPeriodic(name=f"periodic{k}", k=k) for k in (2, 3, 4, 6) if k <= max(2.0, 1.5 * days_to_full)]
    pols += [BlindThreshold(name=f"blind_threshold{round(t * 100)}", tau=t) for t in (0.5, 0.7, 0.9)]
    pols += [BlindRisk(name="blind_risk")]
    pols += [Threshold(name=f"threshold{round(t * 100)}", tau=t) for t in (0.5, 0.7, 0.9)]
    pols += [RiskMyopic(), RiskCFA(name="risk_cfa_t5", theta=5.0), RiskCFA(name="risk_cfa_t15", theta=15.0)]
    return pols


def run_or_load(f: Path, inst, pol, inc, cfg) -> list[DayRecord]:
    """Simulate and save the day records, or read them back if the file exists (resumable runs)."""
    if f.exists():
        d = pd.read_csv(f)
        return [DayRecord(**{k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in d.to_dict("records")]
    recs = simulate(inst, pol, inc, cfg)
    tmp = f.with_suffix(".tmp")
    pd.DataFrame([asdict(r) for r in recs]).to_csv(tmp, index=False)
    tmp.replace(f)
    return recs


def social_cost(s: dict | pd.Series, overflow_cost: float) -> float:
    return s["cost_per_day"] + s["cleanup_cost_per_day"] + overflow_cost * s["overflow_weighted_per_day"]


def loo_tuned(df: pd.DataFrame, keys: list[str], score: str = "social_cost",
              fixed: tuple = ("fixed_daily", "threshold70", "risk_cfa_t5")) -> pd.DataFrame:
    """Leave-one-replication-out tuning.

    df: one row per (keys..., policy, rep) with `score`. For every family, cell and held-out rep,
    the policy with the lowest mean score on the other reps is selected and scored on the held-out
    rep. Adds the overall best no-sensor and sensor choices (families 'best_no_sensor',
    'best_sensor'), and the untuned reference policies in `fixed` (today's practice, the threshold
    rule used in practice and the paper's policy) under their own names.
    Returns one row per (keys..., family, rep) with the chosen policy and its row.
    """
    df = df.copy()
    df["family"] = df["policy"].map(family)
    df["sensing"] = df["family"].map(SENSING)
    out = []
    for cell, g in df.groupby(keys, sort=False):
        cell = cell if isinstance(cell, tuple) else (cell,)
        reps = sorted(g["rep"].unique())
        groups = [(fam, g[g.family == fam]) for fam in g["family"].unique()]
        groups += [("best_no_sensor", g[~g.sensing]), ("best_sensor", g[g.sensing])]
        groups += [(p, g[g.policy == p]) for p in fixed if (g.policy == p).any()]
        for fam, gf in groups:
            for r in reps:
                train = gf[gf.rep != r] if len(reps) > 1 else gf
                choice = train.groupby("policy")[score].mean().idxmin()
                row = gf[(gf.policy == choice) & (gf.rep == r)].iloc[0].to_dict()
                row.update(dict(zip(keys, cell)), tuned_family=fam, chosen=choice)
                out.append(row)
    return pd.DataFrame(out)


def paired_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Mean and t-interval half-width of paired (common-random-number) replication differences."""
    from scipy.stats import t
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def tuned_table(tuned: pd.DataFrame, keys: list[str], base: str = "best_no_sensor") -> pd.DataFrame:
    """Per cell and tuned family: means over replications and paired differences against `base`
    (default: the best policy without sensors), with 95% t-intervals. If the rows carry
    `sensor_cost_day`, the net social cost of sensing families includes it."""
    rows = []
    for cell, g in tuned.groupby(keys, sort=False):
        cell = cell if isinstance(cell, tuple) else (cell,)
        b = g[g.tuned_family == base].set_index("rep")
        for fam, x in g.groupby("tuned_family", sort=False):
            x = x.set_index("rep").loc[b.index]
            sens = x["sensing"].astype(bool)
            sc = x["sensor_cost_day"] if "sensor_cost_day" in x else 0.0
            net = x["social_cost"] + np.where(sens, sc, 0.0)
            net_b = b["social_cost"] + np.where(b["sensing"].astype(bool), sc, 0.0)
            d_op, h_op = paired_ci(x["cost_per_day"] - b["cost_per_day"])
            d_soc, h_soc = paired_ci(x["social_cost"] - b["social_cost"])
            d_net, h_net = paired_ci(net - net_b)
            d_ovf, h_ovf = paired_ci(x["overflow_bins_per_day"] - b["overflow_bins_per_day"])
            rows.append(dict(zip(keys, cell), family=fam, chosen="/".join(sorted(set(x["chosen"]))),
                             reps=len(x), cost=x["cost_per_day"].mean(), cleanup=x["cleanup_cost_per_day"].mean(),
                             overflow_bins=x["overflow_bins_per_day"].mean(), spill_t=x["spill_t_per_day"].mean(),
                             social=x["social_cost"].mean(), sensor_cost=float(np.mean(np.where(sens, sc, 0.0))),
                             d_cost=d_op, d_cost_hw=h_op, d_social=d_soc, d_social_hw=h_soc,
                             d_net=d_net, d_net_hw=h_net, d_overflow=d_ovf, d_overflow_hw=h_ovf,
                             relaxed_days=x["relaxed_days"].mean(), mand_unserved=x["mand_unserved_per_day"].mean()))
    return pd.DataFrame(rows)
