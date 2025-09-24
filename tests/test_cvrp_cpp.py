import math

import numpy as np
import pytest

import cvrp_cpp as cc


# --- Fixtures for different CVRP cases ---


@pytest.fixture(
    params=[
        "all_fit_one_route",
        "must_split_capacity",
        "non_metric_forced_split",
        "infeasible_node",
        "single_node",
        "fail_max_interesting",
    ]
)
def cvrp_case(request):
    """Yields (cvrp, p, description)."""
    name = request.param

    if name == "all_fit_one_route":
        # Everything fits in one route
        n = 4
        dist = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
        depot = [2, 2, 2, 2]
        q = [1, 1, 1, 1]
        cap = 10.0
        p = [0, 1, 2, 3]
        cvrp = cc.CVRP(depot, dist, q, cap)
        return cvrp, p, 7.0, [[0, 1, 2, 3]], name

    if name == "must_split_capacity":
        # Capacity forces split: two routes of load 2 each
        n = 4
        dist = [
            [0, 1, 1, 5],
            [1, 0, 1, 5],
            [1, 1, 0, 1],
            [5, 5, 1, 0],
        ]
        depot = [2, 2, 2, 2]
        q = [1, 1, 1, 1]
        cap = 2.0
        p = [0, 1, 2, 3]
        cvrp = cc.CVRP(depot, dist, q, cap)
        return cvrp, p, 10.0, [[0, 1], [2, 3]], name

    if name == "non_metric_forced_split":
        # Non-metric costs; adjacent edges cheap, cross edges expensive
        n = 4
        dist = [
            [0, 1, 100, 100],
            [1, 0, 1, 100],
            [100, 1, 0, 1],
            [100, 100, 1, 0],
        ]
        depot = [2, 2, 2, 2]
        q = [1, 1, 1, 1]
        cap = 2.0  # forces split into [0,1] and [2,3]
        p = [0, 1, 2, 3]
        cvrp = cc.CVRP(depot, dist, q, cap)
        return cvrp, p, 10.0, [[0, 1], [2, 3]], name

    if name == "infeasible_node":
        # One node exceeds capacity -> DP returns -1
        n = 3
        dist = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ]
        depot = [1, 1, 1]
        q = [5, 1, 1]  # node 0 infeasible if capacity is 3
        cap = 3.0
        p = [0, 1, 2]
        cvrp = cc.CVRP(depot, dist, q, cap)
        return cvrp, p, np.inf, [], name

    if name == "single_node":
        # Instance with a single node
        n = 1
        dist = [[0]]
        depot = [1]
        q = [3.0]
        cap = 3.0
        p = [0]
        cvrp = cc.CVRP(depot, dist, q, cap)
        return cvrp, p, 2.0, [[0]], name

    if name == "fail_max_interesting":
        # The cost in internal DP computation does not always increase,
        # so we can not break unless we make a full pass
        n = 3
        dist = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ]
        depot = [1, 100, 1]
        q = [1, 1, 1]  # node 0 infeasible if capacity is 3
        cap = 3.0
        p = [0, 1, 2]
        cvrp = cc.CVRP(depot, dist, q, cap)
        return cvrp, p, 4.0, [[0, 1, 2]], name

    raise RuntimeError("Unknown case")


# --- Tests ---


def test_dp_matches_manual_ans(cvrp_case):
    cvrp, p, ans, routes, name = cvrp_case
    cpp_val = cvrp.dynamic_programming_fitness(p)
    # For infeasible case both should be inf; else they should match approximately
    if math.isinf(ans):
        assert math.isinf(cpp_val)
    else:
        assert cpp_val == pytest.approx(ans, rel=1e-12, abs=1e-12)

def test_dp_routes_match_manual_ans(cvrp_case):
    cvrp, p, ans, routes, name = cvrp_case
    cpp_val, correct_routes = cvrp.dynamic_programming_fitness_and_ans(p)
    # For infeasible case both should be inf; else they should match approximately
    if math.isinf(ans):
        assert math.isinf(cpp_val)
    else:
        assert cpp_val == pytest.approx(ans, rel=1e-12, abs=1e-12)
    assert np.array_equal(routes, correct_routes)


def test_dp_single_route_formula_when_applicable():
    # A specific case where everything clearly fits one route
    n = 5
    dist = [
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
        [4, 3, 2, 1, 0],
    ]
    depot = [2, 2, 2, 2, 2]
    q = [1, 1, 1, 1, 1]
    cap = 10.0
    p = [0, 1, 2, 3, 4]
    cvrp = cc.CVRP(depot, dist, q, cap)

    # closed-form for one continuous route
    path_cost = (
        depot[p[0]] + sum(dist[p[i]][p[i + 1]] for i in range(n - 1)) + depot[p[-1]]
    )
    cpp_val = cvrp.dynamic_programming_fitness(p)
    assert cpp_val == pytest.approx(path_cost, rel=1e-12, abs=1e-12)


def test_dp_bruteforce_small_instance():
    # Small brute-force over all valid split points under capacity
    n = 5
    dist = [
        [0, 1, 3, 4, 2],
        [1, 0, 1, 5, 2],
        [3, 1, 0, 1, 4],
        [4, 5, 1, 0, 1],
        [2, 2, 4, 1, 0],
    ]
    depot = [2, 2, 2, 2, 2]
    q = [1, 1, 2, 1, 1]
    cap = 3.0
    p = [0, 1, 2, 3, 4]
    cvrp = cc.CVRP(depot, dist, q, cap)

    def route_cost(seg):
        # seg is contiguous indices in p, like [l..r]
        l, r = seg
        cost = (
            depot[p[l]] + sum(dist[p[i]][p[i + 1]] for i in range(l, r)) + depot[p[r]]
        )
        return cost

    # brute-force all segmentations by boundaries that respect capacity
    best = math.inf

    # boundaries represented by indices [b0=0 < b1 < ... < bk=n]
    def feasible(l, r):
        return sum(q[p[i]] for i in range(l, r)) <= cap

    from itertools import combinations

    for k in range(1, n + 1):  # number of segments
        # choose k-1 internal cut positions from 1..n-1
        for cuts in combinations(range(1, n), k - 1):
            bounds = (0,) + cuts + (n,)
            ok = True
            total = 0.0
            for a, b in zip(bounds[:-1], bounds[1:]):
                if not feasible(a, b):
                    ok = False
                    break
                total += route_cost((a, b - 1))
            if ok:
                best = min(best, total)

    cpp_val = cvrp.dynamic_programming_fitness(p)
    assert cpp_val == pytest.approx(best, rel=1e-12, abs=1e-12)


def test_dp_max_interesting_keeps_optimum(cvrp_case):
    cvrp, p, ans, routes, name = cvrp_case
    # If baseline is feasible, setting max_interesting to the optimum must not change the answer
    base = cvrp.dynamic_programming_fitness(p, -1.0)
    if base >= 0.0:
        with_cut = cvrp.dynamic_programming_fitness(
            p, base
        )  # note: prune triggers only on ">"
        assert with_cut == pytest.approx(base, rel=1e-12, abs=1e-12)
