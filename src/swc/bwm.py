"""Best-Worst Method: input validation, input-based consistency, and Bayesian group BWM.

References
    Rezaei (2015) Omega 53:49-57 (BWM); Liang, Brunelli & Rezaei (2020) Omega 96:102175
    (input-based consistency ratio and thresholds); Mohammadi & Rezaei (2020) Omega 96:102075
    (Bayesian BWM for group decisions).

Conventions: criteria are indexed 0..n-1. For each expert k:
    best, worst : indices chosen by the expert
    bo[k]       : Best-to-Others vector, bo[best] = 1, values in 1..9
    ow[k]       : Others-to-Worst vector, ow[worst] = 1, values in 1..9

The Bayesian model (Mohammadi & Rezaei 2020):
    A_B^k ~ Multinomial(1 / w^k),  A_W^k ~ Multinomial(w^k),
    w^k | w* ~ Dirichlet(gamma * w*),  w* ~ Dirichlet(1),  gamma ~ Gamma(0.01, 0.01)
sampled here with an adaptive random-walk Metropolis-within-Gibbs on softmax logits
(self-contained numpy; validated by weight recovery in tests/test_bwm.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

# Liang, Brunelli & Rezaei (2020) input-based CR thresholds, by number of criteria and by a_BW,
# as implemented in the official BWM Excel solver (BWM-Solver-5.xlsx). n = 5 is not in that file.
CR_THRESHOLDS = {
    3: {3: 0.1667, 4: 0.1121, 5: 0.1354, 6: 0.1330, 7: 0.1294, 8: 0.1309, 9: 0.1359},
    4: {3: 0.1667, 4: 0.1529, 5: 0.1994, 6: 0.1990, 7: 0.2457, 8: 0.2521, 9: 0.2681},
    6: {3: 0.1667, 4: 0.2206, 5: 0.2546, 6: 0.3044, 7: 0.3029, 8: 0.3154, 9: 0.3337},
    7: {3: 0.1667, 4: 0.2527, 5: 0.2716, 6: 0.3144, 7: 0.3144, 8: 0.3408, 9: 0.3517},
    8: {3: 0.1667, 4: 0.2577, 5: 0.2844, 6: 0.3221, 7: 0.3251, 8: 0.3620, 9: 0.3620},
    9: {3: 0.1667, 4: 0.2683, 5: 0.2960, 6: 0.3262, 7: 0.3403, 8: 0.3657, 9: 0.3662},
}


# ---------------------------------------------------------------------- validation & consistency
def validate(best: int, worst: int, bo, ow) -> list[str]:
    """Structural checks that must pass before any weighting (the thesis's DM3 fails these)."""
    bo, ow = np.asarray(bo, float), np.asarray(ow, float)
    issues = []
    if best == worst:
        issues.append("best and worst criterion are the same")
    if not ((bo >= 1) & (bo <= 9)).all() or not ((ow >= 1) & (ow <= 9)).all():
        issues.append("values must lie in 1..9")
    if bo[best] != 1:
        issues.append(f"Best-to-Others: best vs itself must be 1 (got {bo[best]:g})")
    if ow[worst] != 1:
        issues.append(f"Others-to-Worst: worst vs itself must be 1 (got {ow[worst]:g})")
    if bo[worst] != ow[best]:
        issues.append(f"best-vs-worst differs between the two vectors ({bo[worst]:g} vs {ow[best]:g})")
    if bo[worst] < bo.max():
        issues.append("in Best-to-Others the worst criterion must have the largest value")
    if ow[best] < ow.max():
        issues.append("in Others-to-Worst the best criterion must have the largest value")
    return issues


def input_based_cr(best: int, worst: int, bo, ow) -> tuple[float, float | None, bool | None]:
    """Input-based consistency ratio CR^I and its threshold (Liang et al. 2020).

    CR_j = |a_Bj * a_jW - a_BW| / (a_BW^2 - a_BW) for a_BW > 1 (0 if a_BW = 1); CR^I = max_j CR_j.
    Returns (CR^I, threshold or None if not tabulated, acceptable or None).
    """
    bo, ow = np.asarray(bo, float), np.asarray(ow, float)
    a_bw = bo[worst]
    if a_bw <= 1:
        return 0.0, None, True
    cr = float(np.max(np.abs(bo * ow - a_bw) / (a_bw ** 2 - a_bw)))
    thr = CR_THRESHOLDS.get(len(bo), {}).get(int(round(a_bw)))
    return cr, thr, (None if thr is None else cr <= thr)


def linear_bwm(best: int, worst: int, bo, ow) -> tuple[np.ndarray, float]:
    """Linear BWM (Rezaei 2016) weights for one expert, solved as an LP."""
    from scipy.optimize import linprog
    bo, ow = np.asarray(bo, float), np.asarray(ow, float)
    n = len(bo)
    # variables: w_0..w_{n-1}, xi ; minimise xi
    c = np.r_[np.zeros(n), 1.0]
    A, b = [], []
    for j in range(n):
        for sgn in (1, -1):
            r = np.zeros(n + 1); r[best] += sgn; r[j] -= sgn * bo[j]; r[-1] = -1; A.append(r); b.append(0)
            r = np.zeros(n + 1); r[j] += sgn; r[worst] -= sgn * ow[j]; r[-1] = -1; A.append(r); b.append(0)
    res = linprog(c, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(n), 0.0]], b_eq=[1.0],
                  bounds=[(0, None)] * (n + 1), method="highs")
    return res.x[:n], float(res.x[-1])


# ---------------------------------------------------------------------- Bayesian BWM
@dataclass
class BayesBWMResult:
    w_star: np.ndarray        # (draws, n) posterior samples of the aggregated weights
    w_k: np.ndarray           # (draws, K, n) posterior samples of individual weights
    gamma: np.ndarray         # (draws,)
    accept: dict

    @property
    def mean(self) -> np.ndarray:
        return self.w_star.mean(axis=0)

    def interval(self, q=(0.025, 0.975)) -> np.ndarray:
        return np.quantile(self.w_star, q, axis=0).T

    def credal(self) -> np.ndarray:
        """P(w*_i > w*_j): the credal ranking matrix of Mohammadi & Rezaei (2020)."""
        w = self.w_star
        return (w[:, :, None] > w[:, None, :]).mean(axis=0)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def bayesian_bwm(BO, OW, draws: int = 4000, burn: int = 4000, thin: int = 2, seed: int = 0) -> BayesBWMResult:
    """Posterior sampling for the Bayesian group BWM. BO, OW: (K, n) arrays of 1..9 values."""
    BO, OW = np.asarray(BO, float), np.asarray(OW, float)
    K, n = BO.shape
    rng = np.random.default_rng(seed)

    def log_dir(w, alpha):  # Dirichlet log density on the simplex
        return gammaln(alpha.sum()) - gammaln(alpha).sum() + ((alpha - 1) * np.log(w)).sum()

    def log_lik_k(wk, k):
        inv = 1.0 / wk
        return (BO[k] * np.log(inv / inv.sum())).sum() + (OW[k] * np.log(wk)).sum()

    # logits with last component fixed at 0 (identifiable); Jacobian of softmax map = prod(w)
    zs = np.zeros(n)
    zk = np.zeros((K, n))
    lg = 0.0  # log gamma
    step_s, step_k, step_g = 0.3, np.full(K, 0.3), 0.3
    acc = {"star": 0, "k": 0, "gamma": 0, "n": 0}

    def lp_star(z, wks, g):
        ws = _softmax(z)
        return sum(log_dir(wks[k], g * ws) for k in range(K)) + np.log(ws).sum()  # Dir(1) prior + Jacobian

    def lp_k(z, k, ws, g):
        wk = _softmax(z)
        return log_lik_k(wk, k) + log_dir(wk, g * ws) + np.log(wk).sum()

    def lp_gamma(l, ws, wks):
        g = np.exp(l)
        return sum(log_dir(wks[k], g * ws) for k in range(K)) + (0.01 - 1) * l - 0.01 * g + l

    ws = _softmax(zs)
    wks = _softmax(zk)
    out_s, out_k, out_g = [], [], []
    total = burn + draws * thin
    for it in range(total):
        # individual weights
        for k in range(K):
            prop = zk[k].copy(); prop[:-1] += step_k[k] * rng.standard_normal(n - 1)
            cur = lp_k(zk[k], k, ws, np.exp(lg))
            new = lp_k(prop, k, ws, np.exp(lg))
            if np.log(rng.random()) < new - cur:
                zk[k] = prop; wks[k] = _softmax(prop); acc["k"] += 1
        # aggregated weights
        prop = zs.copy(); prop[:-1] += step_s * rng.standard_normal(n - 1)
        if np.log(rng.random()) < lp_star(prop, wks, np.exp(lg)) - lp_star(zs, wks, np.exp(lg)):
            zs = prop; ws = _softmax(prop); acc["star"] += 1
        # concentration
        propg = lg + step_g * rng.standard_normal()
        if np.log(rng.random()) < lp_gamma(propg, ws, wks) - lp_gamma(lg, ws, wks):
            lg = propg; acc["gamma"] += 1
        acc["n"] += 1
        # adapt step sizes during burn-in (target ~0.3 acceptance)
        if it < burn and (it + 1) % 100 == 0:
            ra = acc["star"] / acc["n"]; step_s *= np.exp(ra - 0.3)
            rk = acc["k"] / (acc["n"] * K); step_k *= np.exp(rk - 0.3)
            rg = acc["gamma"] / acc["n"]; step_g *= np.exp(rg - 0.3)
            acc = {"star": 0, "k": 0, "gamma": 0, "n": 0}
        if it >= burn and (it - burn) % thin == 0:
            out_s.append(ws.copy()); out_k.append(wks.copy()); out_g.append(np.exp(lg))
    rates = {"star": acc["star"] / max(acc["n"], 1), "k": acc["k"] / max(acc["n"] * K, 1),
             "gamma": acc["gamma"] / max(acc["n"], 1)}
    return BayesBWMResult(np.array(out_s), np.array(out_k), np.array(out_g), rates)


def simulate_expert(w_true: np.ndarray, rng: np.random.Generator, noise: float = 0.15):
    """Generate a BWM response consistent (up to noise) with weights w_true; used for testing."""
    w = w_true * rng.lognormal(0, noise, len(w_true))
    w = w / w.sum()
    b, wst = int(np.argmax(w)), int(np.argmin(w))
    bo = np.clip(np.rint(w[b] / w), 1, 9)
    ow = np.clip(np.rint(w / w[wst]), 1, 9)
    bo[b] = 1; ow[wst] = 1
    ow[b] = bo[wst]
    return b, wst, bo, ow
