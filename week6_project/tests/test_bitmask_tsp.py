"""
test_bitmask_tsp.py
----------------------
pytest tests for src/dp_advanced/bitmask_traveling_salesman.py.

tsp_brute_force is only exercised with a handful of cities (it's O(n!)), mirroring how
Week 5's fibonacci_recursive/knapsack_recursive are only tested at small n -- the real
correctness check for larger instances is tsp_bitmask_dp agreeing with brute force
everywhere brute force can still reasonably finish.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp_advanced.bitmask_traveling_salesman import tsp_brute_force, tsp_bitmask_dp


def build_known_graph():
    """A classic 4-city symmetric TSP instance with a known optimal tour cost of 80
    (verified independently, not hand-derived from this codebase)."""
    return [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]


def tour_cost(dist, tour):
    return sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def is_valid_tour(tour, n):
    """A valid tour starts and ends at city 0, visits every other city exactly once,
    and has the right length (n + 1, counting the return to 0)."""
    return (
        len(tour) == n + 1
        and tour[0] == 0
        and tour[-1] == 0
        and sorted(tour[:-1]) == list(range(n))
    )


class TestTSPBruteForce:
    def test_matches_known_optimum(self):
        dist = build_known_graph()
        cost, tour = tsp_brute_force(dist)
        assert cost == 80
        assert is_valid_tour(tour, 4)
        assert tour_cost(dist, tour) == cost

    def test_single_city_trip_costs_nothing(self):
        assert tsp_brute_force([[0]]) == (0, [0])

    def test_two_cities_round_trip(self):
        dist = [[0, 5], [5, 0]]
        cost, tour = tsp_brute_force(dist)
        assert cost == 10
        assert is_valid_tour(tour, 2)


class TestTSPBitmaskDP:
    def test_matches_known_optimum(self):
        dist = build_known_graph()
        cost, tour = tsp_bitmask_dp(dist)
        assert cost == 80
        assert is_valid_tour(tour, 4)
        assert tour_cost(dist, tour) == cost

    def test_single_city_trip_costs_nothing(self):
        assert tsp_bitmask_dp([[0]]) == (0, [0])

    def test_two_cities_round_trip(self):
        dist = [[0, 5], [5, 0]]
        cost, tour = tsp_bitmask_dp(dist)
        assert cost == 10
        assert is_valid_tour(tour, 2)

    def test_handles_more_cities_than_brute_force_would_survive(self):
        # 14 cities is 13! = 6,227,020,800 permutations -- not something to ask
        # tsp_brute_force to attempt. Just confirm bitmask DP returns a valid,
        # internally-consistent tour; the agreement tests below are what verify
        # correctness against brute force at sizes where both can run.
        rng = random.Random(5)
        n = 14
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = rng.randint(1, 100)
        cost, tour = tsp_bitmask_dp(dist)
        assert is_valid_tour(tour, n)
        assert tour_cost(dist, tour) == cost


class TestBothVariantsAgree:
    def test_agree_on_the_fixture_graph(self):
        dist = build_known_graph()
        bf_cost, _ = tsp_brute_force(dist)
        dp_cost, _ = tsp_bitmask_dp(dist)
        assert bf_cost == dp_cost

    def test_agree_across_random_instances(self):
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(2, 8)  # keep small enough for brute force to finish quickly
            dist = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist[i][j] = rng.randint(1, 50)

            bf_cost, bf_tour = tsp_brute_force(dist)
            dp_cost, dp_tour = tsp_bitmask_dp(dist)

            # Costs must agree exactly; tours themselves may legitimately differ when
            # more than one optimal tour exists (ties), so each tour is checked for
            # validity and its own cost against ITS OWN reported cost instead of
            # requiring the two tours to be identical.
            assert bf_cost == dp_cost
            assert is_valid_tour(bf_tour, n)
            assert is_valid_tour(dp_tour, n)
            assert tour_cost(dist, bf_tour) == bf_cost
            assert tour_cost(dist, dp_tour) == dp_cost
