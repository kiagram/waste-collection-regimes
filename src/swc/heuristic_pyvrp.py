"""Heuristic engine for the daily model: PyVRP hybrid genetic search (Wouda et al., 2024).

Same problem as model_cpsat.DailyModel with the cost objective:
selective (prize = skip penalty), required points = mandatory, heterogeneous trucks,
multiple trips (reload at the transfer station D), shift duration.

Mapping details:
* start depot = depot, end depot = D, reload depots = [D], max_reloads = max_trips - 1;
* the constant return leg D -> depot of every used truck is added to its fixed cost, and its
  travel time plus the final unload are removed from the shift budget (PyVRP charges the
  depot service time at reloads, not at the end depot);
* CO2 is not optimised here (it is load-dependent); it is recomputed by the checker. An optional
  carbon price adds lambda * a_k (empty-truck emissions per metre) to the distance cost.
Used for the rolling-horizon simulation; validated against the exact CP-SAT model on small
instances (scripts/04_validate_heuristic.py).
"""
from __future__ import annotations

import time

import numpy as np
from pyvrp import Model, PenaltyParams, SolveParams
from pyvrp import Solution as PyVRPSolution
from pyvrp._pyvrp import Activity, ActivityType as AT, Route
from pyvrp.stop import MaxIterations, MaxRuntime, NoImprovement, MultipleCriteria

from .evaluate import evaluate
from .instance import Instance
from .model_cpsat import Solution

SCALE = 100_000  # cost units -> integer


def solve_pyvrp(inst: Instance, penalty: np.ndarray, mandatory: set[int], time_s: float = 5.0,
                seed: int = 0, max_trips: int | None = None, carbon_price: float = 0.0,
                max_no_improve: int = 20_000, max_iterations: int | None = None,
                warm_start: bool = True, fleet_descent: bool = False, reduce_iters: int = 2000) -> Solution:
    p = inst.params
    D = inst.disposal
    T = max_trips or p.ops.max_trips
    q = inst.demand_kg()
    dist, tt, svc = inst.dist_m, inst.travel_s(), inst.service_s()

    cand = [i for i in inst.bins if i in mandatory or int(round(penalty[i - 1] * SCALE)) > 0]
    t0 = time.perf_counter()
    if not cand:
        return Solution(routes=[[] for _ in inst.vehicles], status="OPTIMAL", objective="cost",
                        wall_s=0.0, info={"engine": "pyvrp", "trivial": True})

    m = Model()
    node_loc = {}
    for node in [0, D] + cand:
        node_loc[node] = m.add_location(float(inst.xy_m[node, 0]), float(inst.xy_m[node, 1]))
    depot = m.add_depot(node_loc[0], name="depot")
    tstation = m.add_depot(node_loc[D], service_duration=p.ops.unload_s, name="transfer")
    for i in cand:
        m.add_client(node_loc[i], pickup=[int(q[i - 1])], service_duration=int(svc[i - 1]),
                     prize=int(round(penalty[i - 1] * SCALE)), required=i in mandatory, name=str(i))
    ret_d, ret_t = int(dist[D, 0]), int(tt[D, 0])
    # identical trucks (same VehicleType object) share one PyVRP vehicle type with a count;
    # type_trucks maps PyVRP vehicle-type index -> list of physical truck indices
    type_trucks = []
    for k, veh in enumerate(inst.vehicles):
        for grp in type_trucks:
            if inst.vehicles[grp[0]] is veh:
                grp.append(k)
                break
        else:
            type_trucks.append([k])
    for grp in type_trucks:
        veh = inst.vehicles[grp[0]]
        per_m = veh.var_cost_per_km / 1000.0 + carbon_price * veh.a_g_per_m / 1e6  # carbon price per tonne
        m.add_vehicle_type(
            num_available=len(grp), capacity=[veh.capacity_kg], start_depot=depot, end_depot=tstation,
            reload_depots=[tstation], max_reloads=T - 1,
            fixed_cost=int(round((veh.fixed_cost + per_m * ret_d) * SCALE)),
            unit_distance_cost=int(round(per_m * SCALE)),
            shift_duration=int(p.ops.shift_s - p.ops.unload_s - ret_t), name=veh.name)
    nodes = [0, D] + cand
    for a in nodes:
        for b in nodes:
            if a != b:
                m.add_edge(node_loc[a], node_loc[b], distance=int(dist[a, b]), duration=int(tt[a, b]))

    # Iteration-based stopping makes results independent of machine load (reproducible);
    # time_s is then only a safety cap.
    crit = [MaxRuntime(time_s), NoImprovement(max_no_improve)]
    if max_iterations is not None:
        crit.append(MaxIterations(max_iterations))
    stop = MultipleCriteria(crit)
    # Prizes are large relative to the default maximum penalty on load/duration excess, which lets
    # overloaded routes look attractive; raise the cap so infeasibility never pays.
    params = SolveParams(penalty=PenaltyParams(max_penalty=1e12))
    init = None
    if warm_start:
        data = m.data()
        plan = greedy_plan(inst, cand, mandatory, penalty, type_trucks, T)
        rts = [Route(data, [Activity(AT.CLIENT, cand.index(i)) if i != D else Activity(AT.DEPOT, 1) for i in seq], vt)
               for vt, seq in plan]
        init = PyVRPSolution(data, rts)
    res = m.solve(stop=stop, seed=seed, display=False, params=params, initial_solution=init)
    best = res.best
    if warm_start and fleet_descent and best.is_feasible():
        best = reduce_fleet(m.data(), best, res.cost(), seed, params, time_s, reduce_iters)

    def to_routes(sol):
        out = [[] for _ in inst.vehicles]
        free = [list(g) for g in type_trucks]
        for rt in sol.routes():   # read routes even if infeasible; repair() restores feasibility when possible
            k = free[rt.vehicle_type()].pop(0)
            seq = [0]
            for act in rt.schedule():
                if act.is_client():
                    seq.append(cand[act.idx])
                elif act.is_depot() and act.idx == 1 and len(seq) > 1 and seq[-1] != D:
                    seq.append(D)   # reload or end at the transfer station
            if seq[-1] != D:
                seq.append(D)
            out[k] = seq + [0]
        return out

    routes = to_routes(best)
    # truck-type swaps, then a short re-optimisation from the swapped plan (repeat while it helps)
    vt_of = {k: vt for vt, grp in enumerate(type_trucks) for k in grp}
    for _ in range(3 if (warm_start and best.is_feasible()) else 0):
        routes, changed = swap_types(inst, routes, T)
        if not changed:
            break
        data = m.data()
        rts = [Route(data, [Activity(AT.CLIENT, cand.index(i)) if i != D else Activity(AT.DEPOT, 1) for i in r[1:-2]],
                     vt_of[k]) for k, r in enumerate(routes) if r]
        # the swapped plan is feasible (same trips, loads within the new payload, same travel times);
        # the search keeps it as incumbent, so its result is never worse
        res = m.solve(stop=stop, seed=seed + 1, display=False, params=params, initial_solution=PyVRPSolution(data, rts))
        if res.best.is_feasible():
            best = res.best
            routes = to_routes(best)
    wall = time.perf_counter() - t0
    repaired = False
    if not best.is_feasible():
        routes, repaired = repair(inst, routes, penalty, mandatory, T)
    ok = evaluate(inst, routes, penalty, mandatory=mandatory).feasible
    status = "FEASIBLE" if ok else "INFEASIBLE"
    return Solution(routes=routes, status=status, objective="cost", wall_s=wall,
                    obj_value=res.cost() / SCALE if best.is_feasible() else None,
                    info={"engine": "pyvrp", "iterations": res.num_iterations, "repaired": repaired})


def _as_route(data, rt):
    """Rebuild a PyVRP route for (possibly modified) problem data: clients and reload depots in order."""
    acts = [Activity(AT.CLIENT, a.idx) if a.is_client() else Activity(AT.DEPOT, a.idx) for a in rt.schedule()[1:-1]]
    return Route(data, acts, rt.vehicle_type())


def reduce_fleet(data, best, best_cost, seed, params, time_s, iters, max_rounds=8):
    """Fleet-use descent: for each truck type in use, re-solve with one truck of that type fewer
    (its lightest route removed, its points left for the search to re-insert) and keep the best
    improvement; repeat until no reduction pays.

    Local search moves one or two visits at a time and so rarely empties a route or trades a truck
    for a cheaper type; with fixed costs per truck-shift that dominate the objective, this outer
    loop over fleet use closes most of the remaining gap (see scripts/17_heuristic_quality.py).
    """
    import pyvrp
    stop = MultipleCriteria([MaxRuntime(time_s), MaxIterations(iters)])
    for rnd in range(max_rounds):
        routes = list(best.routes())
        usage = {}
        for r in routes:
            usage[r.vehicle_type()] = usage.get(r.vehicle_type(), 0) + 1
        cand = None
        for vt, u in usage.items():
            vts = [data.vehicle_type(k) for k in range(data.num_vehicle_types)]
            if u > 1:
                vts[vt] = vts[vt].replace(num_available=u - 1)
            else:   # PyVRP needs >= 1 vehicle per type: price the last one out instead
                vts[vt] = vts[vt].replace(fixed_cost=vts[vt].fixed_cost + 10 ** 13)
            d2 = data.replace(vehicle_types=vts)
            drop = min((r for r in routes if r.vehicle_type() == vt), key=lambda r: sum(r.pickup()))
            init = PyVRPSolution(d2, [_as_route(d2, r) for r in routes if r is not drop])
            res = pyvrp.solve(d2, stop=stop, seed=seed + 7 * rnd + vt, collect_stats=False, display=False,
                              params=params, initial_solution=init)
            if res.best.is_feasible() and res.cost() < best_cost - 1e-6 and (cand is None or res.cost() < cand[0]):
                cand = (res.cost(), res.best)
        if cand is None:
            return best
        best_cost, best = cand
    return best


def solve_pyvrp_best(inst: Instance, penalty: np.ndarray, mandatory: set[int], starts: int = 2, seed: int = 0,
                     **kw) -> Solution:
    """Best of `starts` independent warm-started runs (seeds seed, seed+1, ...), by the checker's cost."""
    best, best_cost, wall = None, np.inf, 0.0
    for s in range(starts):
        sol = solve_pyvrp(inst, penalty, mandatory, seed=seed + s, **kw)
        wall += sol.wall_s
        k = evaluate(inst, sol.routes, penalty, mandatory=mandatory)
        c = k.cost_total if k.feasible else np.inf
        if best is None or c < best_cost:
            best, best_cost = sol, c
    best.wall_s = wall
    return best


def swap_types(inst: Instance, routes: list, max_trips: int) -> tuple[list, bool]:
    """Exact truck-type swap: move a used truck's whole plan onto an idle truck of another type when
    that truck can carry every trip within its shift and the cost (fixed + distance) falls.

    Local search moves visits one or two at a time, so it rarely changes the type of a whole route;
    with fixed costs per shift that differ by type this leaves easy savings. Best improvement first.
    """
    from .model_cpsat import DailyModel
    D, p = inst.disposal, inst.params
    q, tt, svc, dist = inst.demand_kg(), inst.travel_s(), inst.service_s(), inst.dist_m
    routes = [list(r) for r in routes]
    changed = False
    while True:
        best = (0.0, None, None)
        idle = {inst.vehicles[j].name: j for j, r in enumerate(routes) if not r}
        for k, r in enumerate(routes):
            if not r:
                continue
            vk = inst.vehicles[k]
            trips = DailyModel.split_trips(r, D)
            km = sum(int(dist[a, b]) for a, b in zip(r[:-1], r[1:])) / 1000.0
            loads = [sum(q[i - 1] for i in t) for t in trips]
            for name, j in idle.items():
                vj = inst.vehicles[j]
                if vj is vk or max(loads) > vj.capacity_kg:
                    continue
                delta = (vj.fixed_cost - vk.fixed_cost) + (vj.var_cost_per_km - vk.var_cost_per_km) * km
                if delta < best[0] - 1e-9:
                    best = (delta, k, j)
        if best[1] is None:
            return routes, changed
        _, k, j = best
        routes[j], routes[k] = routes[k], []
        changed = True


def greedy_plan(inst: Instance, cand: list, mandatory: set, penalty: np.ndarray, type_trucks: list, T: int) -> list:
    """Warm start for the local search: few, well-filled trucks.

    Fixed truck-shift costs dominate a collection night, and local search from a random start rarely
    empties a route, so it converges to plans with too many trucks. This constructor packs trips
    close to payload: points are swept by angle around the depot; vehicle types are used in order of
    estimated cost per tonne; each truck takes up to T trips while its shift allows.
    Optional points are included only if their penalty exceeds an estimate of their collection cost.
    Returns [(pyvrp vehicle-type index, [point, ..., D, point, ...]), ...] with D separating trips.
    """
    p = inst.params
    D = inst.disposal
    q = inst.demand_kg()
    tt, svc = inst.travel_s(), inst.service_s()
    km = 2 * float(np.mean(inst.dist_m[D, cand])) / 1000.0            # rough trip length
    est = []
    for vt, grp in enumerate(type_trucks):
        v = inst.vehicles[grp[0]]
        est.append(((v.fixed_cost + v.var_cost_per_km * km * T) / (T * v.capacity_kg), vt))
    per_kg = min(e for e, _ in est)
    pts = [i for i in cand if i in mandatory or penalty[i - 1] >= per_kg * q[i - 1]]
    ang = {i: np.arctan2(*(inst.xy_m[i] - inst.xy_m[0])[::-1]) for i in pts}
    todo = sorted(pts, key=lambda i: ang[i])
    todo.sort(key=lambda i: i not in mandatory)                      # mandatory first, then optional
    plan = []
    for _, vt in sorted(est):
        v = inst.vehicles[type_trucks[vt][0]]
        for _ in type_trucks[vt]:
            if not todo:
                return plan
            seq, clock, prev = [], 0, 0
            for t in range(T):
                trip, load, start = [], 0, prev
                for i in list(todo):
                    if load + q[i - 1] > v.capacity_kg:
                        continue
                    dt = tt[prev, i] + svc[i - 1]
                    back = tt[i, D] + p.ops.unload_s + tt[D, 0]
                    if clock + dt + back > p.ops.shift_s:
                        continue
                    trip.append(i)
                    todo.remove(i)
                    load += q[i - 1]
                    clock += dt
                    prev = i
                    if load > 0.97 * v.capacity_kg:
                        break
                if not trip:
                    break
                clock += tt[prev, D] + p.ops.unload_s
                prev = D
                seq += trip + [D]
            if seq:
                plan.append((vt, seq[:-1]))                          # the final D is the end depot
    return plan


def repair(inst: Instance, routes: list, penalty: np.ndarray, mandatory: set, max_trips: int):
    """Greedy repair: drop optional points with the lowest penalty per kg from overloaded trips,
    then the lowest-penalty optional points from trucks exceeding the shift, until feasible."""
    from .model_cpsat import DailyModel
    D, p = inst.disposal, inst.params
    q, tt, svc = inst.demand_kg(), inst.travel_s(), inst.service_s()
    out = []
    for k, r in enumerate(routes):
        if not r:
            out.append([])
            continue
        cap = inst.vehicles[k].capacity_kg
        trips = [t for t in DailyModel.split_trips(r, D) if t][:max_trips]
        for t in trips:
            while sum(q[i - 1] for i in t) > cap:
                opt = [i for i in t if i not in mandatory]
                if not opt:
                    break
                t.remove(min(opt, key=lambda i: penalty[i - 1] / max(q[i - 1], 1)))

        def duration(trs):
            seq = [0] + [n for t in trs if t for n in t + [D]] + [0]
            ntr = sum(1 for t in trs if t)
            service = sum(int(svc[i - 1]) for t in trs for i in t)
            return sum(int(tt[a, b]) for a, b in zip(seq[:-1], seq[1:])) + service + ntr * p.ops.unload_s

        while any(trips) and duration(trips) > p.ops.shift_s:
            opt = [i for t in trips for i in t if i not in mandatory]
            if not opt:
                break
            worst = min(opt, key=lambda i: penalty[i - 1])
            for t in trips:
                if worst in t:
                    t.remove(worst)
        trips = [t for t in trips if t]
        out.append([0] + [n for t in trips for n in t + [D]] + [0] if trips else [])
    return out, True
