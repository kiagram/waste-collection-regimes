"""Overflow risk of skipping a bin tonight, and the resulting zone-weighted penalty.

This is the interface between the AI forecaster and the optimisation model: any
forecaster that returns, per bin, the predictive distribution of the fill increment
over the horizon can replace `normal_increment` (e.g. conformal quantile regression).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .instance import Instance


def normal_increment(inst: Instance, horizon_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of the fill increment over `horizon_h` under i.i.d. hourly increments."""
    mu = inst.rate_mean_per_h * horizon_h
    sd = inst.rate_sd_per_h * np.sqrt(horizon_h)
    return mu, np.maximum(sd, 1e-9)


def overflow_risk(inst: Instance, mu: np.ndarray | None = None, sd: np.ndarray | None = None,
                  mode: str | None = None) -> np.ndarray:
    """Risk measure per bin if it is NOT collected before the next opportunity.

    The m bins of a collection point are assumed to fill in step (same users, same street),
    i.e. comonotone, so the point overflows as a whole.
    'prob'   : P(the point overflows)
    'count'  : m * P(overflow)          (expected number of overflowing bins)
    'excess' : m * E[(fill + increment - 1)^+]  (expected overflowing volume, in bins)
    """
    rp = inst.params.risk
    mode = mode or rp.mode
    if mu is None or sd is None:
        mu, sd = normal_increment(inst, rp.horizon_h)
    room = 1.0 - inst.fill - mu          # remaining headroom after the mean increment
    z = room / sd
    m = inst.bins_per_point
    p1 = norm.sf(z)
    if mode == "prob":
        return p1
    if mode == "count":
        return m * p1
    if mode == "excess":
        return m * (sd * norm.pdf(z) - room * norm.sf(z))
    raise ValueError(mode)


def overflow_prob_within(inst: Instance, hours: float) -> np.ndarray:
    """P(the point overflows within `hours` if it is not collected), under the fill model."""
    mu = inst.rate_mean_per_h * hours
    sd = np.maximum(inst.rate_sd_per_h * np.sqrt(hours), 1e-9)
    return norm.sf((1.0 - inst.fill - mu) / sd)


def risk_tolerance(inst: Instance) -> np.ndarray:
    """Zone-dependent maximum acceptable overflow probability alpha_i."""
    rp = inst.params.risk
    s = inst.sensitivity()
    return rp.alpha_max - (rp.alpha_max - rp.alpha_min) * s


def mandatory_bins(inst: Instance, prob: np.ndarray | None = None) -> set[int]:
    """Node indices of bins whose overflow probability exceeds their zone tolerance."""
    prob = overflow_risk(inst, mode="prob") if prob is None else prob
    return {i + 1 for i in np.where(prob > risk_tolerance(inst))[0]}


def skip_penalty(inst: Instance, risk: np.ndarray | None = None) -> np.ndarray:
    """Monetised, zone-weighted penalty (cost units) for leaving each bin uncollected."""
    risk = overflow_risk(inst, mode="count") if risk is None else risk
    return inst.params.risk.overflow_cost * inst.sensitivity() * risk
