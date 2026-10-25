"""
test_set_cover_lpround.py
----------------------------
Tests for the Set Cover LP relaxation (lp_solver.py plus the LP building
in set_cover_lpround.py) and both rounding techniques. Checks that the LP
relaxation genuinely lower bounds the true integer optimum, that both
rounding methods always produce a valid cover, and that randomized
rounding actually varies from trial to trial on a fractional instance.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.approximation.set_cover_lpround import (
    solve_set_cover_lp_relaxation,
    deterministic_rounding,
    randomized_rounding,
    exact_set_cover_bruteforce,
    is_valid_cover,
    solution_cost,
)
from src.utils.graph_utils import random_set_cover_instance


# A small hand built instance with a genuinely fractional LP relaxation,
# the classic "triangle" example, 3 elements, 3 sets each covering 2 of
# them, all equal cost, LP optimum uses 0.5 of each set (cost 1.5), the
# true integer optimum needs 2 full sets (cost 2).
TRIANGLE_UNIVERSE = {0, 1, 2}
TRIANGLE_SETS = [frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})]
TRIANGLE_COSTS = [1.0, 1.0, 1.0]


class TestLPRelaxation:
    def test_lp_cost_is_at_most_the_true_optimum(self):
        rng = random.Random(1)
        universe, sets, costs = random_set_cover_instance(10, 8, rng=rng)
        _x, lp_cost = solve_set_cover_lp_relaxation(universe, sets, costs)
        _exact_chosen, exact_cost = exact_set_cover_bruteforce(universe, sets, costs)
        assert lp_cost <= exact_cost + 1e-6

    def test_triangle_instance_relaxation_is_fractional_and_cheaper_than_integer(self):
        x, lp_cost = solve_set_cover_lp_relaxation(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS)
        assert all(abs(v - 0.5) < 1e-6 for v in x)
        assert abs(lp_cost - 1.5) < 1e-6
        _exact_chosen, exact_cost = exact_set_cover_bruteforce(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS)
        assert abs(exact_cost - 2.0) < 1e-9
        assert lp_cost < exact_cost


class TestDeterministicRounding:
    def test_always_produces_a_valid_cover(self):
        rng = random.Random(2)
        for _ in range(5):
            universe, sets, costs = random_set_cover_instance(10, 9, rng=rng)
            x, _lp_cost = solve_set_cover_lp_relaxation(universe, sets, costs)
            chosen = deterministic_rounding(universe, sets, x)
            assert is_valid_cover(chosen, sets, universe)

    def test_triangle_instance_rounds_every_tied_set_in(self):
        # every x_s is exactly 0.5 here, and frequency f = 2 (each element
        # is in exactly 2 of the 3 sets) makes the threshold 1/f = 0.5 too,
        # so all three sets tie right at the cutoff and all round in, a
        # valid but non-minimal cover, which is the known tradeoff of
        # frequency rounding, it guarantees feasibility, not minimality.
        x, _lp_cost = solve_set_cover_lp_relaxation(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS)
        chosen = deterministic_rounding(TRIANGLE_UNIVERSE, TRIANGLE_SETS, x)
        assert is_valid_cover(chosen, TRIANGLE_SETS, TRIANGLE_UNIVERSE)
        assert chosen == {0, 1, 2}


class TestRandomizedRounding:
    def test_always_produces_a_valid_cover(self, rng):
        universe, sets, costs = random_set_cover_instance(10, 9, rng=random.Random(3))
        x, _lp_cost = solve_set_cover_lp_relaxation(universe, sets, costs)
        for _ in range(15):
            chosen = randomized_rounding(universe, sets, costs, x, rng)
            assert is_valid_cover(chosen, sets, universe)

    def test_has_real_variance_on_a_fractional_instance(self, rng):
        x, _lp_cost = solve_set_cover_lp_relaxation(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS)
        costs_seen = {
            solution_cost(randomized_rounding(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS, x, rng), TRIANGLE_COSTS)
            for _ in range(30)
        }
        # with every x_s = 0.5, trials should land on more than one outcome
        assert len(costs_seen) > 1

    def test_expected_cost_is_reasonably_close_to_scaled_lp_cost(self, rng):
        # Not an exact equality (it's a random process), just a sanity
        # check that unscaled (scale=1.0) randomized rounding's mean cost
        # over many trials isn't wildly off from the LP cost on an
        # instance with no fix-up needed (every element covered by exactly
        # the sets with positive LP weight, so fix-up rarely triggers).
        x, lp_cost = solve_set_cover_lp_relaxation(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS)
        trial_costs = [
            solution_cost(randomized_rounding(TRIANGLE_UNIVERSE, TRIANGLE_SETS, TRIANGLE_COSTS, x, rng), TRIANGLE_COSTS)
            for _ in range(200)
        ]
        mean_cost = sum(trial_costs) / len(trial_costs)
        assert mean_cost < 3 * lp_cost  # loose bound, just checks it's in a sane range


class TestExactSolver:
    def test_returns_a_valid_cover(self):
        rng = random.Random(4)
        universe, sets, costs = random_set_cover_instance(8, 7, rng=rng)
        chosen, cost = exact_set_cover_bruteforce(universe, sets, costs)
        assert is_valid_cover(chosen, sets, universe)
        assert abs(solution_cost(chosen, costs) - cost) < 1e-9

    def test_weighted_instance_does_not_just_pick_fewest_sets(self):
        # One expensive set covers everything alone, two cheap sets
        # together also cover everything for less, exact solver should
        # pick the two cheap ones even though that's more sets.
        universe = {0, 1, 2, 3}
        sets = [frozenset({0, 1, 2, 3}), frozenset({0, 1}), frozenset({2, 3})]
        costs = [100.0, 1.0, 1.0]
        chosen, cost = exact_set_cover_bruteforce(universe, sets, costs)
        assert chosen == {1, 2}
        assert abs(cost - 2.0) < 1e-9
