"""Tests for the v4 changes made after the review of 7 Oct 2026."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swc.config import Params
from swc.evaluate import evaluate
from swc.forecast import block_bootstrap_ci, diebold_mariano, dm_panel
from swc.heuristic_pyvrp import swap_types
from swc.instance import generate_synthetic
from swc.risk import (expected_excess, mandatory_bins, overflow_risk, prob_exceeds, risk_tolerance, skip_penalty,
                      skip_serve_probs, unavoidable_violations)
from swc.simulate import (BlindRisk, BlindThreshold, FixedPeriodic, RiskMyopic, SimConfig, draw_increments,
                          mass_balance_residual, simulate, summarise)


def make(n=8, seed=0, cv=1.0, bpp=(1, 3), fleet=None, **risk):
    P = Params()
    P.risk = replace(P.risk, fill_rate_cv=cv, **risk)
    return generate_synthetic(n, seed=seed, params=P, bins_per_point=bpp,
                              fleet_counts=fleet or {"new_compactor": 1, "old_compactor": 1, "small_loader": 1})


# ----------------------------------------------------------------------------- risk model
def test_gamma_and_normal_tails_match_reviewer_example():
    mu = np.array([1 / 2.3])
    sd = mu * 1.0  # CV = 1
    assert prob_exceeds(np.array([0.5]), mu, sd, "gamma")[0] == pytest.approx(0.317, abs=1e-3)
    assert prob_exceeds(np.array([0.5]), mu, sd, "normal")[0] == pytest.approx(0.440, abs=1e-3)
    assert prob_exceeds(np.array([1.0]), mu, sd, "gamma")[0] == pytest.approx(0.100, abs=1e-3)
    assert prob_exceeds(np.array([-0.1]), mu, sd, "gamma")[0] == 1.0


def test_gamma_expected_excess_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sd = 0.4, 0.3
    k = (mu / sd) ** 2
    x = rng.gamma(k, mu / k, size=1_000_000)
    for r in (-0.2, 0.0, 0.3, 0.9):
        mc = np.maximum(x - r, 0).mean()
        assert expected_excess(np.array([r]), np.array([mu]), np.array([sd]))[0] == pytest.approx(mc, rel=1e-2, abs=1e-4)


def test_prize_prices_only_avoidable_risk():
    inst = make()
    inst.fill = np.zeros(inst.n)
    assert np.allclose(overflow_risk(inst, mode="count"), 0.0)       # emptying an empty point avoids nothing
    assert np.allclose(skip_penalty(inst), 0.0)
    inst.fill = np.full(inst.n, 0.6)
    ps, pv = skip_serve_probs(inst)
    assert np.allclose(overflow_risk(inst, mode="prob"), ps - pv)
    old = make(incremental=False)
    old.fill = inst.fill
    assert np.allclose(overflow_risk(old, mode="prob"), ps)
    # excess: avoidable expected volume is non-negative and below the skipped one
    e = overflow_risk(inst, mode="excess")
    assert (e >= 0).all() and (e <= overflow_risk(old, mode="excess") + 1e-12).all()


def test_decision_model_uses_the_simulated_law():
    inst = make()
    assert inst.params.risk.dist == "gamma"
    nrm = make(dist="normal")
    nrm.fill = inst.fill
    assert not np.allclose(skip_serve_probs(inst)[0], skip_serve_probs(nrm)[0])


def test_mandatory_and_unavoidable():
    inst = make(cv=1.4)
    inst.rate_mean_per_h = inst.rate_mean_per_h * 3          # fast filling: P_serve can exceed alpha
    inst.rate_sd_per_h = inst.rate_sd_per_h * 3
    ps, pv = skip_serve_probs(inst)
    a = risk_tolerance(inst)
    assert mandatory_bins(inst) == {i + 1 for i in range(inst.n) if ps[i] > a[i]}
    un = unavoidable_violations(inst)
    assert (un == (pv > a)).all() and un.any()
    assert all(i + 1 in mandatory_bins(inst) for i in np.where(un)[0])  # P_skip >= P_serve


# ----------------------------------------------------------------------------- spilled waste
def test_street_waste_adds_demand_and_manual_time():
    inst = make()
    q0, s0 = inst.demand_kg().copy(), inst.service_s().copy()
    inst.street_bins = np.full(inst.n, 0.5)
    ops, sp = inst.params.ops, inst.params.spill
    assert np.allclose(inst.demand_kg() - q0, 0.5 * ops.bin_full_kg, atol=1)
    assert np.allclose(inst.service_s() - s0, 0.5 * ops.service_s_per_bin * sp.manual_service_factor, atol=1)
    sub = inst.subset([1, 3])
    assert np.allclose(sub.street_bins, 0.5)


def _sim(inst, policy, days=8, seed=0):
    cfg = SimConfig(days=days, engine="pyvrp", pyvrp_iters=300, pyvrp_starts=1, pyvrp_s=30)
    inc = draw_increments(inst, days, seed)
    return simulate(inst, policy, inc, cfg)


@pytest.mark.parametrize("mode", ["recover", "lost"])
def test_mass_balance_closes(mode):
    inst = make(n=10, seed=4, cv=1.0)
    inst.params.spill = replace(inst.params.spill, mode=mode)
    recs = _sim(inst, FixedPeriodic(k=4))                     # infrequent service -> overflows
    assert sum(r.spill_bins for r in recs) > 0
    assert mass_balance_residual(inst, recs) == pytest.approx(0.0, abs=1e-9)
    if mode == "lost":
        assert sum(r.lost_bins for r in recs) == pytest.approx(sum(r.spill_bins for r in recs))
        assert all(r.coll_street_bins == 0 for r in recs)
    else:
        assert sum(r.lost_bins for r in recs) == 0
        assert sum(r.coll_street_bins for r in recs) > 0      # street waste is picked up later
    s = summarise(recs, burn_in=2)
    assert recs[0].t_per_bin == pytest.approx(inst.params.ops.bin_full_kg / 1000)
    assert s["cleanup_cost_per_day"] > 0 and s["cost_incl_cleanup_per_day"] > s["cost_per_day"]


def test_trucks_carry_recovered_street_waste():
    inst = make(n=10, seed=4)
    recs = _sim(inst, FixedPeriodic(k=4))
    kg = sum(r.kg_collected for r in recs)
    vol = sum(r.coll_bin_bins + r.coll_street_bins for r in recs) * inst.params.ops.bin_full_kg
    assert kg == pytest.approx(vol, rel=1e-3, abs=inst.n * len(recs))   # integer rounding per point


def test_street_pickup_never_makes_a_point_uncollectable():
    # large points with a big backlog: demand must stay within the largest payload
    inst = make(n=3, seed=2, bpp=(28, 30), fleet={"new_compactor": 3})
    inst.street_bins = np.full(inst.n, 200.0)
    inst.fill = np.full(inst.n, 0.9)
    recs = _sim(inst, FixedPeriodic(k=1), days=3)
    q_max = max(v.capacity_kg for v in inst.vehicles)
    assert all(r.mand_unserved == 0 for r in recs)
    assert sum(r.coll_street_bins for r in recs) > 0
    assert mass_balance_residual(inst, recs) == pytest.approx(0.0, abs=1e-9)
    assert inst.bins_per_point.max() * inst.params.ops.bin_full_kg <= q_max


# ----------------------------------------------------------------------------- no-sensor baselines
def test_fixed_periodic_serves_every_point_once_per_cycle():
    inst = make(n=13)
    inst.fill = np.full(inst.n, 0.5)
    for k in (1, 2, 3, 4):
        pol = FixedPeriodic(k=k)
        seen = []
        for d in range(k):
            seen += sorted(pol.decide(inst, d)[1])
        assert sorted(seen) == list(inst.bins)


def test_blind_policies_never_read_fill_levels():
    inst = make(n=10)
    for pol in (BlindThreshold(tau=0.7), BlindRisk(), BlindRisk(theta=5.0)):
        pol.decide(inst, 0)                                       # initial ages from the starting survey
        other = make(n=10)
        other.fill = np.random.default_rng(1).uniform(0.01, 1.0, inst.n)
        pa, ma = pol.decide(inst, 1)
        pb, mb = pol.decide(other, 1)
        assert ma == mb and np.allclose(pa, pb)
        assert not pol.uses_sensors


def test_blind_ages_follow_service():
    inst = make(n=6)
    pol = BlindThreshold(tau=0.7)
    pol.decide(inst, 0)
    a0 = pol._age.copy()
    served = np.array([True, False, True, False, False, False])
    pol.observe(served)
    assert (pol._age == np.where(served, 0, a0 + 1)).all()


def test_blind_risk_rises_with_elapsed_time():
    inst = make(n=6)
    pol = BlindRisk()
    pol.decide(inst, 0)
    pol._age = np.zeros(inst.n, dtype=int)
    p0, _ = pol.decide(inst, 0)
    pol._age = np.full(inst.n, 3)
    p3, _ = pol.decide(inst, 0)
    assert (p3 >= p0).all() and (p3 > p0).any()


def test_sensor_and_blind_risk_agree_when_fill_is_known_exactly():
    # with deterministic increments the blind belief equals the sensor reading
    inst = make(n=6, cv=1e-4)
    mu_day = inst.rate_mean_per_h * 24
    pol = BlindRisk()
    pol._age = np.full(inst.n, 2)
    inst.fill = 2 * mu_day
    pb, mb = pol.decide(inst, 0)
    ps, ms = RiskMyopic().decide(inst, 0)
    assert mb == ms and np.allclose(pb, ps, atol=1e-6)


# ----------------------------------------------------------------------------- carbon consistency
def test_swap_types_prices_carbon_like_the_search():
    inst = make(n=4, bpp=(1, 1), fleet={"new_compactor": 1, "old_compactor": 1})
    old = next(k for k, v in enumerate(inst.vehicles) if v.name == "old_compactor")
    D = inst.disposal
    routes = [[] for _ in inst.vehicles]
    routes[old] = [0, 1, 2, 3, 4, D, 0]
    same, changed = swap_types(inst, routes, 2, carbon_price=0.0)
    assert not changed                                     # the new truck's fixed cost does not pay
    moved, changed = swap_types(inst, routes, 2, carbon_price=1e5)
    assert changed and not moved[old]                      # at a high carbon price it does
    k0 = evaluate(inst, routes, np.zeros(inst.n))
    k1 = evaluate(inst, moved, np.zeros(inst.n))
    assert k1.priced(1e5) < k0.priced(1e5) and k1.cost_total > k0.cost_total


# ----------------------------------------------------------------------------- forecast tests
def test_dm_uses_the_horizon():
    rng = np.random.default_rng(0)
    e = rng.normal(size=600)
    d = e + np.r_[0, e[:-1]] + np.r_[0, 0, e[:-2]] + 0.05    # MA(2): overlapping 3-step errors
    s1, _ = diebold_mariano(d, np.zeros_like(d), h=1)
    s3, _ = diebold_mariano(d, np.zeros_like(d), h=3)
    assert abs(s3) < abs(s1)                                  # positive autocorrelation inflates h=1


def test_panel_tests_treat_a_date_as_one_observation():
    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange("2020-01-01", "2020-07-01", dtype="datetime64[D]"), 30)
    common = np.repeat(rng.normal(size=len(dates) // 30), 30)  # all bins of a date move together
    d = common + 0.1 * rng.normal(size=len(dates)) + 0.05
    s_pool, _ = diebold_mariano(d, np.zeros_like(d), h=1)
    s_pan, _, dbar, lag, T = dm_panel(d, np.zeros_like(d), dates, h=3)
    assert T == len(dates) // 30 and lag >= 2
    assert abs(s_pan) < abs(s_pool) / 3                       # pooled rows overstate the evidence
    est, lo, hi, p = block_bootstrap_ci(d, np.zeros_like(d), dates, B=500)
    assert est == pytest.approx(d.mean()) and lo < est < hi and 0 <= p <= 1
