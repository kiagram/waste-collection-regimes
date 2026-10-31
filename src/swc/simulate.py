"""Rolling-horizon, multi-day simulation of collection policies (audit C2/M4).

Timeline of day t:
    night t : the policy decides which points to collect and routes the fleet (DailyModel);
              collected points are emptied, and the crew also picks up spilled waste lying next
              to them (up to spill.recover_cap container volumes per container, and no more than
              fits with the containers in the largest truck).
    day t+1 : every point accumulates a random increment X ~ Gamma(mean mu_i*24h, CV);
              if f + X > 1 the point overflows (m_i bins overflow), the containers stay full and
              the excess is added to the point's street backlog (spill.mode = 'recover') or
              leaves the system ('lost', the accounting of the first draft).
All policies face the same increment realisations (common random numbers) so that
differences between policies are not sampling noise.

A mass ledger (in container volumes) closes every day:
    generated = collected from containers + collected from the street
                + change in container stock + change in street stock + lost.

Information: sensor policies read tonight's fill levels. No-sensor policies (FixedDaily,
FixedPeriodic, BlindThreshold, BlindRisk) never read them: they only know when they last served
each point and the mean fill rates. All policies are routed on the true demand, which favours the
no-sensor policies (their trucks never run out of room unexpectedly), so the estimated value of
sensing is conservative.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .evaluate import evaluate
from .instance import Instance
from .heuristic_pyvrp import solve_pyvrp_best
from .model_cpsat import DailyModel
from .risk import (increment_moments, mandatory_bins, overflow_prob_within, overflow_risk, prob_exceeds,
                   risk_tolerance, skip_penalty, skip_serve_probs, unavoidable_violations)


# ----------------------------------------------------------------------------- policies
@dataclass
class Policy:
    """Decides, from today's observable state, the penalty vector and mandatory set."""
    name: str
    uses_sensors: bool = True

    def decide(self, inst: Instance, day: int) -> tuple[np.ndarray, set]:
        raise NotImplementedError

    def observe(self, served: np.ndarray) -> None:
        """Called after each night with the points that were served."""


@dataclass
class FixedDaily(Policy):
    """Conventional practice: visit every point every night regardless of fill.

    min_fill > 0 needs fill readings; the default 0 only skips points that are physically empty
    (nothing to collect), which crews see on arrival.
    """
    name: str = "fixed_daily"
    uses_sensors: bool = False
    min_fill: float = 0.0

    def decide(self, inst, day):
        mand = {i + 1 for i in range(inst.n) if inst.fill[i] > self.min_fill}
        return np.zeros(inst.n), mand


@dataclass
class FixedPeriodic(Policy):
    """Fixed schedule without sensors: the district is cut into `k` angular sectors around the depot
    with equal numbers of points, and sector d mod k is served on night d (every point every k days)."""
    name: str = "fixed_periodic"
    uses_sensors: bool = False
    k: int = 2
    _group: np.ndarray | None = None

    def decide(self, inst, day):
        if self._group is None:
            v = inst.xy_m[1:-1] - inst.xy_m[0]
            ang = np.arctan2(v[:, 1], v[:, 0])
            rank = np.argsort(np.argsort(ang, kind="stable"), kind="stable")
            self._group = (rank * self.k) // inst.n
        mand = {i + 1 for i in range(inst.n) if self._group[i] == day % self.k and inst.fill[i] > 0}
        return np.zeros(inst.n), mand


@dataclass
class FixedAlternate(FixedPeriodic):
    """Alternate-day fixed schedule: two geographic halves served on alternate nights."""
    name: str = "fixed_alternate"
    k: int = 2


@dataclass
class _Elapsed(Policy):
    """Tracks days since each point was last served; the only state a no-sensor policy has.

    The initial ages are those implied by the starting fill levels (one inventory survey)."""
    uses_sensors: bool = False
    _age: np.ndarray | None = None

    def _init(self, inst):
        if self._age is None:
            mu_day = np.maximum(inst.rate_mean_per_h * 24.0, 1e-9)
            self._age = np.rint(inst.fill / mu_day).astype(int)

    def observe(self, served):
        self._age = np.where(served, 0, self._age + 1)


@dataclass
class BlindThreshold(_Elapsed):
    """Elapsed-time rule without sensors: serve a point when its expected fill, mean daily increment
    x days since service, reaches tau (a per-point periodic schedule)."""
    name: str = "blind_threshold70"
    tau: float = 0.70

    def decide(self, inst, day):
        self._init(inst)
        exp_fill = inst.rate_mean_per_h * 24.0 * self._age
        mand = {i + 1 for i in range(inst.n) if exp_fill[i] >= self.tau}
        return np.zeros(inst.n), mand


@dataclass
class BlindRisk(_Elapsed):
    """The risk-priced selective routing model run without sensor readings.

    Tonight's fill is unknown; given a = days since service, a point skipped tonight overflows by
    tomorrow if the sum of a + 1 daily increments exceeds one container (fill is monotone between
    services), and an emptied point if one day's increment does. Penalty, chance constraint and the
    optional look-ahead term (theta, as in RiskCFA) use these probabilities. Comparing it with the
    sensor policies isolates the value of information from that of selection and routing.
    """
    name: str = "blind_risk"
    theta: float = 0.0

    def decide(self, inst, day):
        self._init(inst)
        rp = inst.params.risk
        h = rp.horizon_h
        mu, sd = increment_moments(inst, h * (self._age + 1))
        p_skip = prob_exceeds(np.ones(inst.n), mu, sd, rp.dist)
        p_serve = skip_serve_probs(inst, fill=np.zeros(inst.n))[0]
        w = 1.0 if rp.incremental else 0.0
        risk = inst.bins_per_point * np.maximum(p_skip - w * p_serve, 0.0)
        pen = skip_penalty(inst, risk)
        if self.theta:
            mu2, sd2 = increment_moments(inst, h * (self._age + 2))
            pen = pen + self.theta * prob_exceeds(np.ones(inst.n), mu2, sd2, rp.dist)
        return pen, {i + 1 for i in np.where(p_skip > risk_tolerance(inst))[0]}


@dataclass
class Threshold(Policy):
    """Sensor threshold rule used in practice (thesis 'red bins'): collect if fill >= tau."""
    name: str = "threshold70"
    tau: float = 0.70

    def decide(self, inst, day):
        return np.zeros(inst.n), {i + 1 for i in range(inst.n) if inst.fill[i] >= self.tau}


@dataclass
class RiskMyopic(Policy):
    """Zone-weighted overflow risk for tonight only (chance constraint + penalty)."""
    name: str = "risk_myopic"

    def decide(self, inst, day):
        pen = skip_penalty(inst, overflow_risk(inst, mode="count"))
        return pen, mandatory_bins(inst)


@dataclass
class RiskCFA(Policy):
    """Cost function approximation: myopic risk + theta * P(point will need a visit tomorrow).

    theta (cost units) approximates the cost of an extra visit tomorrow; it is tuned by
    simulation on separate seeds (Powell's CFA; cf. Lagos 2026, Cuellar-Usaquen et al. 2025).
    """
    name: str = "risk_cfa"
    theta: float = 10.0

    def decide(self, inst, day):
        pen = skip_penalty(inst, overflow_risk(inst, mode="count"))
        p48 = overflow_prob_within(inst, 48.0)
        return pen + self.theta * p48, mandatory_bins(inst)


# ----------------------------------------------------------------------------- simulator
@dataclass
class SimConfig:
    days: int = 14
    time_limit: float = 10.0
    workers: int = 16
    solver_seed: int = 0
    objective: str = "cost"
    engine: str = "pyvrp"         # 'pyvrp' (validated heuristic) | 'cpsat' (exact, small instances)
    pyvrp_s: float = 120.0        # safety cap only
    pyvrp_iters: int | None = 5000  # deterministic stopping rule (None -> time-based)
    pyvrp_starts: int = 2           # independent warm-started runs per night; the cheapest is kept


@dataclass
class DayRecord:
    day: int
    status: str
    gap: float | None
    wall_s: float
    points_served: int
    trucks: int
    km: float
    cost_fixed: float
    cost_variable: float
    co2_kg: float
    kg_collected: float
    overflow_bins: int = 0
    overflow_weighted: float = 0.0
    overflow_volume_bins: float = 0.0
    relaxed: bool = False
    # mandatory demand
    mand_requested: int = 0          # points the policy declared mandatory
    mand_unserved: int = 0           # of which not served (only on relaxed nights)
    unavoidable: int = 0             # points with P_serve > alpha_i (constraint cannot hold if served)
    # mass ledger, container volumes
    gen_bins: float = 0.0            # generated during the day
    coll_bin_bins: float = 0.0       # emptied from containers tonight
    coll_street_bins: float = 0.0    # picked up from the street tonight
    spill_bins: float = 0.0          # spilled during the day
    lost_bins: float = 0.0           # spilled and removed from the system ('lost' mode only)
    stock_bins: float = 0.0          # in containers at the end of the day
    street_bins: float = 0.0         # on the street at the end of the day
    cleanup_cost: float = 0.0        # cleanup_cost_per_t x tonnes spilled
    t_per_bin: float = 0.22          # tonnes in one full container (converts the ledger)


def draw_increments(inst: Instance, days: int, seed: int) -> np.ndarray:
    """(days, n) daily fill increments, Gamma with the instance's mean and CV."""
    rng = np.random.default_rng(seed)
    mean = inst.rate_mean_per_h * 24.0
    cv = np.where(inst.rate_mean_per_h > 0, inst.rate_sd_per_h / np.maximum(inst.rate_mean_per_h, 1e-12), 0.3)
    cv_day = cv / np.sqrt(24.0)       # CV of a 24 h sum of i.i.d. hourly increments
    shape = 1.0 / np.maximum(cv_day, 1e-3) ** 2
    return rng.gamma(shape, mean / shape, size=(days, inst.n))


def simulate(base: Instance, policy: Policy, increments: np.ndarray, cfg: SimConfig) -> list[DayRecord]:
    inst = copy.copy(base)
    sp = base.params.spill
    m = base.bins_per_point.astype(float)
    fill = np.minimum(base.fill, 1.0).copy()
    street = np.zeros(base.n) if base.street_bins is None else base.street_bins.astype(float).copy()
    sens = base.sensitivity()
    t_per_bin = base.params.ops.bin_full_kg / 1000.0
    q_max = max(v.capacity_kg for v in base.vehicles) / base.params.ops.bin_full_kg  # largest payload, containers
    policy = copy.deepcopy(policy)
    recs = []
    for day in range(increments.shape[0]):
        inst.fill = fill.copy()
        # street waste the crew can take tonight: capped per visit, and never more than fits, with the
        # containers, in the largest truck (otherwise the point could never be served again)
        pickup = np.minimum(np.minimum(street, sp.recover_cap * m), np.maximum(q_max - m * fill, 0.0))
        inst.street_bins = pickup
        pen, mand = policy.decide(inst, day)
        relaxed = False

        def run(pen_, mand_):
            if cfg.engine == "pyvrp":
                return solve_pyvrp_best(inst, pen_, mand_, starts=cfg.pyvrp_starts, time_s=cfg.pyvrp_s,
                                        seed=cfg.solver_seed + 1000 * day, max_iterations=cfg.pyvrp_iters)
            model = DailyModel(inst, penalty=pen_, mandatory=mand_)
            return model.solve(cfg.objective, time_limit=cfg.time_limit, workers=cfg.workers,
                               seed=cfg.solver_seed)

        sol = run(pen, mand)
        if sol.status not in ("OPTIMAL", "FEASIBLE"):
            # fleet/shift cannot serve every mandatory point: soften them with a large penalty so that
            # the engine serves as many as it can. A heuristic failure is not proof of physical
            # infeasibility; such nights are counted (relaxed) with the mandatory points left unserved.
            relaxed = True
            big = pen.copy()
            for i in mand:
                big[i - 1] += 10 * inst.params.risk.overflow_cost * max(1, inst.bins_per_point[i - 1])
            sol = run(big, set())
        k = evaluate(inst, sol.routes, np.zeros(inst.n))
        served = np.zeros(inst.n, dtype=bool)
        for r in sol.routes:
            for node in r:
                if node in inst.bins:
                    served[node - 1] = True
        policy.observe(served)
        coll_bin = float((m * fill)[served].sum())
        coll_street = float(pickup[served].sum())
        street = np.where(served, street - pickup, street)
        fill = np.where(served, 0.0, fill)
        # next day's accumulation and overflows
        inc = increments[day]
        new = fill + inc
        over = new > 1.0
        spill = np.where(over, m * (new - 1.0), 0.0)
        fill = np.minimum(new, 1.0)
        lost = 0.0
        if sp.mode == "recover":
            street = street + spill
        elif sp.mode == "lost":
            lost = float(spill.sum())
        else:
            raise ValueError(sp.mode)
        rec = DayRecord(day=day, status=sol.status, gap=sol.gap, wall_s=sol.wall_s,
                        points_served=int(served.sum()), trucks=k.trucks_used, km=k.km,
                        cost_fixed=k.cost_fixed, cost_variable=k.cost_variable, co2_kg=k.co2_kg,
                        kg_collected=k.kg_collected, relaxed=relaxed, t_per_bin=t_per_bin)
        rec.overflow_bins = int(base.bins_per_point[over].sum())
        rec.overflow_weighted = float((m * sens)[over].sum())
        rec.overflow_volume_bins = float(spill.sum())
        rec.mand_requested = len(mand)
        rec.mand_unserved = int(sum(not served[i - 1] for i in mand))
        rec.unavoidable = int(unavoidable_violations(inst).sum())
        rec.gen_bins = float((m * inc).sum())
        rec.coll_bin_bins, rec.coll_street_bins = coll_bin, coll_street
        rec.spill_bins, rec.lost_bins = float(spill.sum()), lost
        rec.stock_bins, rec.street_bins = float((m * fill).sum()), float(street.sum())
        rec.cleanup_cost = sp.cleanup_cost_per_t * t_per_bin * rec.spill_bins
        recs.append(rec)
    return recs


def initial_stocks(base: Instance) -> tuple[float, float]:
    """(container stock, street stock) before day 0, in container volumes."""
    m = base.bins_per_point.astype(float)
    street = 0.0 if base.street_bins is None else float(base.street_bins.sum())
    return float((m * np.minimum(base.fill, 1.0)).sum()), street


def mass_balance_residual(base: Instance, recs: list[DayRecord]) -> float:
    """Initial stocks + generated - collected - lost - final stocks (container volumes); zero up to rounding."""
    s0, st0 = initial_stocks(base)
    gen = sum(r.gen_bins for r in recs)
    out = sum(r.coll_bin_bins + r.coll_street_bins + r.lost_bins for r in recs)
    return s0 + st0 + gen - out - recs[-1].stock_bins - recs[-1].street_bins


def summarise(recs: list[DayRecord], burn_in: int = 2) -> dict:
    r = recs[burn_in:]
    tot = lambda a: float(sum(getattr(x, a) for x in r))
    days = len(r)
    t = r[0].t_per_bin
    op = (tot("cost_fixed") + tot("cost_variable")) / days
    return {
        "days": days,
        "cost_per_day": op,
        "cleanup_cost_per_day": tot("cleanup_cost") / days,
        "cost_incl_cleanup_per_day": op + tot("cleanup_cost") / days,
        "co2_kg_per_day": tot("co2_kg") / days,
        "km_per_day": tot("km") / days,
        "truck_shifts_per_day": tot("trucks") / days,
        "points_served_per_day": tot("points_served") / days,
        "t_per_day": tot("kg_collected") / 1000 / days,
        "overflow_bins_per_day": tot("overflow_bins") / days,
        "overflow_weighted_per_day": tot("overflow_weighted") / days,
        "overflow_volume_bins_per_day": tot("overflow_volume_bins") / days,
        "kg_per_km": tot("kg_collected") / max(tot("km"), 1e-9),
        # mass balance over the measured window (tonnes per day)
        "gen_t_per_day": t * tot("gen_bins") / days,
        "coll_bin_t_per_day": t * tot("coll_bin_bins") / days,
        "coll_street_t_per_day": t * tot("coll_street_bins") / days,
        "spill_t_per_day": t * tot("spill_bins") / days,
        "lost_t_per_day": t * tot("lost_bins") / days,
        "street_stock_end_t": t * r[-1].street_bins,
        "street_stock_growth_t_per_day": t * (r[-1].street_bins - recs[burn_in - 1].street_bins) / days
        if burn_in > 0 else t * r[-1].street_bins / days,
        # mandatory demand and feasibility handling
        "relaxed_days": int(sum(x.relaxed for x in r)),
        "mand_requested_per_day": tot("mand_requested") / days,
        "mand_unserved_per_day": tot("mand_unserved") / days,
        "unavoidable_per_day": tot("unavoidable") / days,
        "non_optimal_days": int(sum(x.status not in ("OPTIMAL", "FEASIBLE") for x in r)),
        "mean_solve_s": tot("wall_s") / days,
    }
