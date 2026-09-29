"""Rolling-horizon, multi-day simulation of collection policies.

Timeline of day t:
    night t : sensors report fill f_t; the policy decides which points to collect and routes
              the fleet (DailyModel); collected points are emptied.
    day t+1 : every point accumulates a random increment X ~ Gamma(mean mu_i*24h, CV);
              if f + X > 1 the point overflows (m_i bins overflow, spill is recorded) and
              stays at 1 until it is collected.
All policies face the same increment realisations (common random numbers) so that
differences between policies are not sampling noise.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .evaluate import evaluate
from .instance import Instance
from .heuristic_pyvrp import solve_pyvrp_best
from .model_cpsat import DailyModel
from .risk import mandatory_bins, overflow_prob_within, overflow_risk, skip_penalty


# ----------------------------------------------------------------------------- policies
@dataclass
class Policy:
    """Decides, from today's observed state, the penalty vector and mandatory set."""
    name: str

    def decide(self, inst: Instance, day: int) -> tuple[np.ndarray, set]:
        raise NotImplementedError


@dataclass
class FixedDaily(Policy):
    """Conventional practice: visit every point every night regardless of fill."""
    name: str = "fixed_daily"
    min_fill: float = 0.0

    def decide(self, inst, day):
        mand = {i + 1 for i in range(inst.n) if inst.fill[i] > self.min_fill}
        return np.zeros(inst.n), mand


@dataclass
class FixedAlternate(Policy):
    """Alternate-day fixed schedule: two geographic halves served on alternate nights."""
    name: str = "fixed_alternate"
    _group: np.ndarray | None = None

    def decide(self, inst, day):
        if self._group is None:
            v = inst.xy_m[1:-1] - inst.xy_m[0]
            ang = np.arctan2(v[:, 1], v[:, 0])
            self._group = (ang > np.median(ang)).astype(int)
        mand = {i + 1 for i in range(inst.n) if self._group[i] == day % 2 and inst.fill[i] > 0}
        return np.zeros(inst.n), mand


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
    fill = base.fill.copy()
    sens = base.sensitivity()
    policy = copy.deepcopy(policy)
    recs = []
    for day in range(increments.shape[0]):
        inst.fill = fill.copy()
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
            # fleet/shift cannot serve every mandatory point: soften them with a large penalty
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
        fill = np.where(served, 0.0, fill)
        # next day's accumulation and overflows
        new = fill + increments[day]
        over = new > 1.0
        rec = DayRecord(day=day, status=sol.status, gap=sol.gap, wall_s=sol.wall_s,
                        points_served=int(served.sum()), trucks=k.trucks_used, km=k.km,
                        cost_fixed=k.cost_fixed, cost_variable=k.cost_variable, co2_kg=k.co2_kg,
                        kg_collected=k.kg_collected, relaxed=relaxed)
        m = base.bins_per_point
        rec.overflow_bins = int(m[over].sum())
        rec.overflow_weighted = float((m * sens)[over].sum())
        rec.overflow_volume_bins = float((m * (new - 1.0))[over].sum())
        fill = np.minimum(new, 1.0)
        recs.append(rec)
    return recs


def summarise(recs: list[DayRecord], burn_in: int = 2) -> dict:
    r = recs[burn_in:]
    tot = lambda a: float(sum(getattr(x, a) for x in r))
    days = len(r)
    return {
        "days": days,
        "cost_per_day": (tot("cost_fixed") + tot("cost_variable")) / days,
        "co2_kg_per_day": tot("co2_kg") / days,
        "km_per_day": tot("km") / days,
        "truck_shifts_per_day": tot("trucks") / days,
        "points_served_per_day": tot("points_served") / days,
        "t_per_day": tot("kg_collected") / 1000 / days,
        "overflow_bins_per_day": tot("overflow_bins") / days,
        "overflow_weighted_per_day": tot("overflow_weighted") / days,
        "overflow_volume_bins_per_day": tot("overflow_volume_bins") / days,
        "kg_per_km": tot("kg_collected") / max(tot("km"), 1e-9),
        "relaxed_days": int(sum(x.relaxed for x in r)),
        "non_optimal_days": int(sum(x.status not in ("OPTIMAL", "FEASIBLE") for x in r)),
        "mean_solve_s": tot("wall_s") / days,
    }
