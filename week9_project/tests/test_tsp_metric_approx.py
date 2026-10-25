"""
test_tsp_metric_approx.py
----------------------------
Tests for the MST doubling metric TSP approximation and the exact brute
force solver it gets compared against. Checks that the MST is actually a
valid spanning tree, that the final tour visits every city exactly once,
and that the approximation holds to both the 2 * MST bound the proof
relies on and the weaker 2 * OPT bound it's actually supposed to
guarantee.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.approximation.tsp_metric_approx import (
    build_mst_edges,
    tsp_mst_doubling_approx,
    tsp_exact_bruteforce,
    approximation_ratio,
)
from src.utils.graph_utils import random_points, tour_length, euclidean_distance


def _is_valid_tour(tour, num_points):
    return sorted(tour) == list(range(num_points))


def _mst_weight(mst_edges, points):
    return sum(euclidean_distance(points[u], points[v]) for u, v in mst_edges)


class TestMST:
    def test_has_n_minus_one_edges_and_connects_everything(self):
        points = random_points(8, rng=random.Random(1))
        mst_edges = build_mst_edges(points)
        assert len(mst_edges) == len(points) - 1

        # union-find style reachability check, every vertex connects back to 0
        adjacency = {i: set() for i in range(len(points))}
        for u, v in mst_edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        seen = {0}
        frontier = [0]
        while frontier:
            v = frontier.pop()
            for u in adjacency[v]:
                if u not in seen:
                    seen.add(u)
                    frontier.append(u)
        assert seen == set(range(len(points)))


class TestMSTDoublingApprox:
    def test_visits_every_city_exactly_once(self):
        points = random_points(9, rng=random.Random(2))
        tour = tsp_mst_doubling_approx(points)
        assert _is_valid_tour(tour, len(points))

    def test_two_cities_is_trivially_handled(self):
        points = [(0.0, 0.0), (10.0, 0.0)]
        tour = tsp_mst_doubling_approx(points)
        assert _is_valid_tour(tour, 2)

    def test_tour_length_never_exceeds_twice_the_mst_weight(self):
        rng = random.Random(3)
        for _ in range(6):
            points = random_points(rng.randint(5, 10), rng=rng)
            mst_edges = build_mst_edges(points)
            mst_weight = _mst_weight(mst_edges, points)
            tour = tsp_mst_doubling_approx(points)
            tour_len = tour_length(tour, points)
            assert tour_len <= 2 * mst_weight + 1e-6


class TestExactBruteforce:
    def test_visits_every_city_exactly_once(self):
        points = random_points(6, rng=random.Random(4))
        tour, length = tsp_exact_bruteforce(points)
        assert _is_valid_tour(tour, len(points))
        assert abs(length - tour_length(tour, points)) < 1e-9

    def test_matches_a_known_square_layout(self):
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        _tour, length = tsp_exact_bruteforce(points)
        assert abs(length - 40.0) < 1e-6


class TestApproximationRatio:
    def test_never_beats_optimal_and_stays_within_2x(self):
        rng = random.Random(5)
        for _ in range(6):
            points = random_points(rng.randint(5, 9), rng=rng)
            tour = tsp_mst_doubling_approx(points)
            approx_length = tour_length(tour, points)
            _opt_tour, opt_length = tsp_exact_bruteforce(points)
            ratio = approximation_ratio(approx_length, opt_length)
            assert ratio >= 1.0 - 1e-9
            assert ratio <= 2.0 + 1e-9

    def test_ratio_of_one_when_optimal_is_zero_length(self):
        assert approximation_ratio(0.0, 0.0) == 1.0
