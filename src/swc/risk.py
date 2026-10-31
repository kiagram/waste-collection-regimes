"""Overflow risk of skipping a point tonight, and the resulting zone-weighted penalty.

This is the interface between the forecaster and the optimisation model: any forecaster that
returns, per point, the predictive distribution of the fill increment over the horizon can replace
`increment_moments` (e.g. conformal quantile regression).

The increment X over the horizon is Gamma by default, the same law the simulator draws from
(simulate.draw_increments), so the decision model and the simulated world agree; 'normal' keeps
the earlier Normal approximation as a misspecification sensitivity.

Serving tonight does not remove tomorrow's overflow risk: an emptied point still overflows if
X > 1. The penalty therefore prices only the avoidable part, P_skip - P_serve, with
    P_skip  = P(f + X > 1)   (point skipped tonight)
    P_serve = P(X > 1)       (point emptied tonight).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma, norm

from .instance import Instance


def increment_moments(inst: Instance, horizon_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of the fill increment over `horizon_h` under i.i.d. hourly increments."""
    mu = inst.rate_mean_per_h * horizon_h
    sd = inst.rate_sd_per_h * np.sqrt(horizon_h)
    return mu, np.maximum(sd, 1e-9)


normal_increment = increment_moments  # backwards-compatible name


def _shape_scale(mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.maximum(mu, 1e-12)
    k = (mu / sd) ** 2
    return k, mu / k


def prob_exceeds(room: np.ndarray, mu: np.ndarray, sd: np.ndarray, dist: str = "gamma") -> np.ndarray:
    """P(X > room) for an increment X with mean mu and sd sd."""
    room = np.asarray(room, dtype=float)
    if dist == "normal":
        return norm.sf((room - mu) / sd)
    if dist != "gamma":
        raise ValueError(dist)
    k, th = _shape_scale(mu, sd)
    return np.where(room <= 0.0, 1.0, gamma.sf(np.maximum(room, 0.0), k, scale=th))


def expected_excess(room: np.ndarray, mu: np.ndarray, sd: np.ndarray, dist: str = "gamma") -> np.ndarray:
    """E[(X - room)^+], the expected overflowing volume (fraction of a container)."""
    room = np.asarray(room, dtype=float)
    if dist == "normal":
        z = (room - mu) / sd
        return sd * norm.pdf(z) - (room - mu) * norm.sf(z)
    k, th = _shape_scale(mu, sd)
    r = np.maximum(room, 0.0)
    # E[(X - r)^+] = mu * P(X' > r) - r * P(X > r), X' ~ Gamma(k + 1, th); for r < 0 it is mu - r
    pos = mu * gamma.sf(r, k + 1, scale=th) - r * gamma.sf(r, k, scale=th)
    return np.where(room < 0.0, mu - room, pos)


def skip_serve_probs(inst: Instance, mu: np.ndarray | None = None, sd: np.ndarray | None = None,
                     fill: np.ndarray | None = None, dist: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(P_skip, P_serve) per point: overflow before the next opportunity if skipped / emptied tonight."""
    rp = inst.params.risk
    dist = dist or rp.dist
    if mu is None or sd is None:
        mu, sd = increment_moments(inst, rp.horizon_h)
    fill = inst.fill if fill is None else fill
    return prob_exceeds(1.0 - fill, mu, sd, dist), prob_exceeds(np.ones_like(fill), mu, sd, dist)


def overflow_risk(inst: Instance, mu: np.ndarray | None = None, sd: np.ndarray | None = None,
                  mode: str | None = None, fill: np.ndarray | None = None) -> np.ndarray:
    """Overflow risk per point that collecting it tonight avoids.

    The m bins of a collection point are assumed to fill in step (same users, same street),
    i.e. comonotone, so the point overflows as a whole.
    'prob'   : P_skip - P_serve                          (avoidable overflow probability)
    'count'  : m * (P_skip - P_serve)                    (avoidable expected overflowing bins)
    'excess' : m * (E[(f + X - 1)^+] - E[(X - 1)^+])     (avoidable expected volume, in bins)
    With risk.incremental = False the serve terms are dropped (the formulation of the first draft).
    """
    rp = inst.params.risk
    mode = mode or rp.mode
    if mu is None or sd is None:
        mu, sd = increment_moments(inst, rp.horizon_h)
    fill = inst.fill if fill is None else fill
    w = 1.0 if rp.incremental else 0.0  # weight of the post-service term
    m = inst.bins_per_point
    if mode in ("prob", "count"):
        ps, pv = skip_serve_probs(inst, mu, sd, fill)
        p = np.maximum(ps - w * pv, 0.0)
        return p if mode == "prob" else m * p
    if mode == "excess":
        es = expected_excess(1.0 - fill, mu, sd, rp.dist)
        ev = expected_excess(np.ones_like(fill), mu, sd, rp.dist)
        return m * np.maximum(es - w * ev, 0.0)
    raise ValueError(mode)


def overflow_prob_within(inst: Instance, hours: float, fill: np.ndarray | None = None) -> np.ndarray:
    """P(the point overflows within `hours` if it is not collected), under the fill model."""
    mu, sd = increment_moments(inst, hours)
    fill = inst.fill if fill is None else fill
    return prob_exceeds(1.0 - fill, mu, sd, inst.params.risk.dist)


def risk_tolerance(inst: Instance) -> np.ndarray:
    """Zone-dependent maximum acceptable overflow probability alpha_i."""
    rp = inst.params.risk
    s = inst.sensitivity()
    return rp.alpha_max - (rp.alpha_max - rp.alpha_min) * s


def mandatory_bins(inst: Instance, prob: np.ndarray | None = None) -> set[int]:
    """Node indices of points that must be served tonight.

    Chance constraint (1 - z_i) P_skip,i + z_i P_serve,i <= alpha_i: service is forced when
    P_skip,i > alpha_i. If also P_serve,i > alpha_i the constraint cannot hold whatever is done;
    the point is still served (it lowers the risk) and the violation is counted by
    `unavoidable_violations`. `prob` overrides P_skip (e.g. a forecaster's probabilities).
    """
    prob = skip_serve_probs(inst)[0] if prob is None else prob
    return {i + 1 for i in np.where(prob > risk_tolerance(inst))[0]}


def unavoidable_violations(inst: Instance, p_serve: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of points whose overflow probability exceeds alpha_i even if emptied tonight."""
    p_serve = skip_serve_probs(inst)[1] if p_serve is None else p_serve
    return p_serve > risk_tolerance(inst)


def skip_penalty(inst: Instance, risk: np.ndarray | None = None) -> np.ndarray:
    """Monetised, zone-weighted penalty (cost units) for leaving each point uncollected:
    pi_i = c_o * omega_z(i) * m_i * (P_skip,i - P_serve,i) with the default 'count' measure."""
    risk = overflow_risk(inst, mode="count") if risk is None else risk
    return inst.params.risk.overflow_cost * inst.sensitivity() * risk
