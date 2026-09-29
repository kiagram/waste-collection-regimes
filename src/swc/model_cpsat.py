"""Daily selective heterogeneous-fleet, multi-trip collection model (exact, OR-Tools CP-SAT).

Merges thesis stages 1 and 2: instead of ranking grey bins with BWM and
cutting the list by solver time, every collection point is a candidate. Leaving point i
uncollected costs its zone-weighted overflow penalty pi_i (risk.py), and points whose
overflow probability exceeds their zone tolerance are mandatory (chance constraint).

Each physical truck p performs up to T trips. Trip t is a circuit
    start -> points -> transfer station D
where start is the depot for t = 0 and D for t >= 1 (the truck has just unloaded);
only the truck's last trip is charged the return D -> depot.

    f1 (cost) = sum_p F_p used_p + sum_{k} c_k sum_ij d_ij x_ijk + sum_i pi_i (1 - served_i)
    f2 (CO2)  = sum_{k} sum_ij d_ij (a_k x_ijk + b_k l_ijk)     (load-dependent)

l_ijk is the load (kg) carried on arc (i, j) during trip k.

Fixes relative to the thesis code: truck-specific variable cost, shift-length
limit, no zone multipliers on CO2, no mirror-symmetry constraint, single-point routes
allowed, arc filtering only as an explicit optional k-nearest-neighbour rule, and solver
status and bound always reported.

Route representation (per physical truck): one node sequence
    [0, i1, i2, D, j1, j2, D, 0]   (D separates trips; [] if the truck is unused).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .instance import Instance
from .risk import skip_penalty

STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
          cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
          cp_model.UNKNOWN: "UNKNOWN"}


@dataclass
class Solution:
    routes: list                  # per physical truck: node sequence, [] if unused
    status: str
    objective: str
    obj_value: float | None = None
    obj_bound: float | None = None
    wall_s: float = 0.0
    cost_int: int | None = None   # scaled integer objectives as seen by the solver
    co2_int: int | None = None
    info: dict = field(default_factory=dict)

    @property
    def gap(self) -> float | None:
        if self.obj_value is None or self.obj_bound is None or self.obj_value == 0:
            return None
        return abs(self.obj_value - self.obj_bound) / abs(self.obj_value)


class DailyModel:
    """Build once, then solve repeatedly with different objectives / epsilon bounds."""

    def __init__(self, inst: Instance, penalty: np.ndarray | None = None, knn: int | None = None,
                 mandatory: set[int] | None = None, max_trips: int | None = None,
                 drop_worthless: bool = True):
        self.inst = inst
        p = inst.params
        n, D = inst.n, inst.disposal
        self.D = D
        self.nodes = list(range(n + 2))
        self.bins = list(inst.bins)
        T = max_trips or p.ops.max_trips
        self.trips = [(truck, t) for truck in range(len(inst.vehicles)) for t in range(T)]
        self.K = list(range(len(self.trips)))
        dist = inst.dist_m
        tt = inst.travel_s()
        q = inst.demand_kg()
        pen = skip_penalty(inst) if penalty is None else np.asarray(penalty, float)
        self.penalty = pen
        self.mandatory = set(mandatory or ())
        cs, es = p.scale.cost_scale, p.scale.co2_scale
        allowed = self._allowed_bin_arcs(knn)

        m = cp_model.CpModel()
        x, l, skip, used = {}, {}, {}, {}
        cost_terms, co2_terms = [], []
        trip_time = {}
        for k, (truck, t) in enumerate(self.trips):
            veh = inst.vehicles[truck]
            Q = veh.capacity_kg
            start = 0 if t == 0 else D                     # physical start of this trip
            arcs = []
            for i in self.nodes:
                skip[i, k] = m.NewBoolVar(f"skip_{i}_{k}")
                arcs.append((i, i, skip[i, k]))
            used[k] = skip[0, k].Not()
            m.Add(skip[D, k] == skip[0, k])
            for i in self.nodes:
                for j in self.nodes:
                    if i == j or not self._arc_ok(i, j, D, allowed):
                        continue
                    v = m.NewBoolVar(f"x_{i}_{j}_{k}")
                    x[i, j, k] = v
                    arcs.append((i, j, v))
                    if i in inst.bins:  # load carried out of a collection point
                        ub = max(Q - (int(q[j - 1]) if j in inst.bins else 0), 0)
                        l[i, j, k] = m.NewIntVar(0, ub, f"l_{i}_{j}_{k}")
                        m.Add(l[i, j, k] <= ub * v)
                        m.Add(l[i, j, k] >= int(q[i - 1]) * v)
            m.AddCircuit(arcs)
            for i in self.bins:
                m.AddImplication(skip[0, k], skip[i, k])
                out_l = [l[i, j, k] for j in self.nodes if (i, j, k) in l]
                in_l = [l[h, i, k] for h in self.nodes if (h, i, k) in l]
                m.Add(sum(out_l) - sum(in_l) == int(q[i - 1]) * skip[i, k].Not())

            # physical distance / time of each modelled arc for this trip
            def phys(i, j):
                if (i, j) == (D, 0):
                    return None                            # return leg handled via `last`
                return (start if i == 0 else i), j

            tsum = []
            for (i, j, kk), v in x.items():
                if kk != k:
                    continue
                pij = phys(i, j)
                if pij is None:
                    continue
                d = dist[pij]
                tsum.append(int(tt[pij]) * v)
                cost_terms.append(int(round(veh.var_cost_per_km * d / 1000.0 * cs)) * v)
                co2_terms.append(int(round(veh.a_g_per_m * d * es)) * v)
            for (i, j, kk), v in l.items():
                if kk == k:
                    co2_terms.append(int(round(veh.b_g_per_m_per_kg * dist[i, j] * es)) * v)
            svc = inst.service_s()
            service = sum(int(svc[i - 1]) * skip[i, k].Not() for i in self.bins)
            trip_time[k] = sum(tsum) + service + p.ops.unload_s * used[k]

        # trip ordering, last-trip return leg, truck use, shift length
        self.truck_used = {}
        for truck in range(len(inst.vehicles)):
            veh = inst.vehicles[truck]
            ks = [k for k, (tr, _) in enumerate(self.trips) if tr == truck]
            for a, b in zip(ks[:-1], ks[1:]):
                m.AddImplication(used[b], used[a])
            ret_d = dist[D, 0]
            ret_terms_t = []
            for idx, k in enumerate(ks):
                nxt = ks[idx + 1] if idx + 1 < len(ks) else None
                last = m.NewBoolVar(f"last_{k}")
                if nxt is None:
                    m.Add(last == used[k])
                else:
                    m.AddBoolAnd([used[k], used[nxt].Not()]).OnlyEnforceIf(last)
                    m.AddBoolOr([used[k].Not(), used[nxt]]).OnlyEnforceIf(last.Not())
                cost_terms.append(int(round(veh.var_cost_per_km * ret_d / 1000.0 * cs)) * last)
                co2_terms.append(int(round(veh.a_g_per_m * ret_d * es)) * last)
                ret_terms_t.append(int(inst.travel_s()[D, 0]) * last)
            self.truck_used[truck] = used[ks[0]]
            cost_terms.append(int(round(veh.fixed_cost * cs)) * used[ks[0]])
            m.Add(sum(trip_time[k] for k in ks) + sum(ret_terms_t) <= p.ops.shift_s)

        served = {}
        for i in self.bins:
            served[i] = m.NewBoolVar(f"served_{i}")
            m.Add(sum(skip[i, k].Not() for k in self.K) == served[i])
            if i in self.mandatory:
                m.Add(served[i] == 1)
            elif drop_worthless and int(round(pen[i - 1] * cs)) <= 0:
                # exact reduction: serving i can only add cost and CO2
                m.Add(served[i] == 0)

        # symmetry breaking between identical physical trucks
        for truck in range(1, len(inst.vehicles)):
            if inst.vehicles[truck] is inst.vehicles[truck - 1]:
                m.AddImplication(self.truck_used[truck], self.truck_used[truck - 1])

        pen_int = {i: int(round(pen[i - 1] * cs)) for i in self.bins}
        self.pen_const = sum(pen_int.values())
        cost_expr = self.pen_const + sum(-pen_int[i] * served[i] for i in self.bins) + sum(cost_terms)
        co2_expr = sum(co2_terms)

        self.m, self.x, self.l, self.skip, self.used, self.served = m, x, l, skip, used, served
        self.cost_var = m.NewIntVar(0, 2**50, "cost")
        self.co2_var = m.NewIntVar(0, 2**50, "co2")
        m.Add(self.cost_var == cost_expr)
        m.Add(self.co2_var == co2_expr)

    # ------------------------------------------------------------------
    def _allowed_bin_arcs(self, knn):
        if knn is None:
            return None
        d = self.inst.dist_m
        allowed = set()
        for i in self.bins:
            order = [j for j in np.argsort(d[i]) if j in self.inst.bins and j != i][:knn]
            for j in order:
                allowed.add((i, int(j)))
                allowed.add((int(j), i))
        return allowed

    @staticmethod
    def _arc_ok(i, j, D, allowed):
        if j == 0:
            return i == D           # the circuit closes only via the transfer station
        if i == D:
            return False
        if i == 0 and j == D:
            return False            # a used trip must collect something
        if i != 0 and j != D and allowed is not None:
            return (i, j) in allowed
        return True

    # ------------------------------------------------------------------
    def solve(self, objective: str = "cost", co2_ub: int | None = None, cost_ub: int | None = None,
              time_limit: float = 60.0, workers: int = 8, seed: int = 0, hint: Solution | None = None,
              log: bool = False) -> Solution:
        # Work on a clone so epsilon bounds / hints never leak between solves.
        m = self.m.Clone()
        self._cur = m
        cost_v = m.GetIntVarFromProtoIndex(self.cost_var.Index())
        co2_v = m.GetIntVarFromProtoIndex(self.co2_var.Index())
        if co2_ub is not None:
            m.Add(co2_v <= int(co2_ub))
        if cost_ub is not None:
            m.Add(cost_v <= int(cost_ub))
        if hint is not None and hint.routes:
            self._add_hint(hint)
        m.Minimize(cost_v if objective == "cost" else co2_v)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        solver.parameters.random_seed = seed
        solver.parameters.log_search_progress = log
        t0 = time.perf_counter()
        st = solver.Solve(m)
        wall = time.perf_counter() - t0
        sol = Solution(routes=[], status=STATUS[st], objective=objective, wall_s=wall)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol.obj_value = solver.ObjectiveValue()
            sol.obj_bound = solver.BestObjectiveBound()
            sol.cost_int = solver.Value(self.cost_var)
            sol.co2_int = solver.Value(self.co2_var)
            sol.routes = self._extract(solver)
        sol.info = {"deterministic_time": solver.deterministic_time, "branches": solver.NumBranches(),
                    "wall_time": solver.WallTime()}
        return sol

    def _trip_sequence(self, solver, k) -> list:
        succ = {i: j for (i, j, kk), v in self.x.items() if kk == k and solver.Value(v)}
        seq, cur = [], 0
        while True:
            cur = succ[cur]
            if cur == self.D:
                return seq
            seq.append(cur)
            if len(seq) > len(self.nodes):
                raise RuntimeError("route extraction loop")

    def _extract(self, solver) -> list:
        routes = []
        for truck in range(len(self.inst.vehicles)):
            ks = [k for k, (tr, _) in enumerate(self.trips) if tr == truck]
            seq = [0]
            for k in ks:
                if not solver.Value(self.used[k]):
                    break
                seq += self._trip_sequence(solver, k) + [self.D]
            routes.append(seq + [0] if len(seq) > 1 else [])
        return routes

    @staticmethod
    def split_trips(route: list, D: int) -> list:
        """[0, a, b, D, c, D, 0] -> [[a, b], [c]]"""
        trips, cur = [], []
        for node in route[1:-1]:
            if node == D:
                trips.append(cur)
                cur = []
            else:
                cur.append(node)
        return trips

    def _add_hint(self, sol: Solution):
        m = self._cur
        on, visited, used_k = set(), set(), set()
        for truck, r in enumerate(sol.routes):
            if not r:
                continue
            ks = [k for k, (tr, _) in enumerate(self.trips) if tr == truck]
            for k, trip in zip(ks, self.split_trips(r, self.D)):
                used_k.add(k)
                path = [0] + trip + [self.D, 0]
                on.update((a, b, k) for a, b in zip(path[:-1], path[1:]))
                visited.update((i, k) for i in trip)
        for (i, j, k), v in self.x.items():
            m.AddHint(m.GetBoolVarFromProtoIndex(v.Index()), (i, j, k) in on)
        for (i, k), v in self.skip.items():
            vv = m.GetBoolVarFromProtoIndex(v.Index())
            if i in (0, self.D):
                m.AddHint(vv, k not in used_k)
            else:
                m.AddHint(vv, (i, k) not in visited)
