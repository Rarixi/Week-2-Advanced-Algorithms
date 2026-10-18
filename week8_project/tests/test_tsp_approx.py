"""
test_tsp_approx.py
---------------------
Tests for the nearest neighbor TSP heuristic and the exact brute force
solver it gets compared against. Checks that both produce actual valid
tours (every city visited exactly once) and that the heuristic's tour is
never shorter than the true optimal, since that would mean the "optimal"
solver itself is broken.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.complexity.traveling_salesman_approx import (
    tsp_nearest_neighbor,
    tsp_exact_bruteforce,
    approximation_ratio,
)
from src.utils.graph_utils import random_points, tour_length


def _is_valid_tour(tour, num_points):
    return sorted(tour) == list(range(num_points))


class TestNearestNeighbor:
    def test_visits_every_city_exactly_once(self):
        points = random_points(9, rng=random.Random(2))
        tour = tsp_nearest_neighbor(points)
        assert _is_valid_tour(tour, len(points))

    def test_two_cities_is_trivially_handled(self):
        points = [(0.0, 0.0), (10.0, 0.0)]
        tour = tsp_nearest_neighbor(points)
        assert _is_valid_tour(tour, 2)


class TestExactBruteforce:
    def test_visits_every_city_exactly_once(self):
        points = random_points(6, rng=random.Random(4))
        tour, length = tsp_exact_bruteforce(points)
        assert _is_valid_tour(tour, len(points))
        assert length == tour_length(tour, points)

    def test_matches_a_known_square_layout(self):
        # Four corners of a square, side length 10, optimal tour is just
        # walking the perimeter, length 40, not cutting across a diagonal.
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        _tour, length = tsp_exact_bruteforce(points)
        assert abs(length - 40.0) < 1e-6


class TestApproximationRatio:
    def test_heuristic_never_beats_optimal(self):
        rng = random.Random(6)
        for _ in range(5):
            points = random_points(rng.randint(5, 8), rng=rng)
            nn_tour = tsp_nearest_neighbor(points)
            nn_length = tour_length(nn_tour, points)
            _opt_tour, opt_length = tsp_exact_bruteforce(points)
            ratio = approximation_ratio(nn_length, opt_length)
            assert ratio >= 1.0 - 1e-9

    def test_ratio_of_one_when_optimal_is_zero_length(self):
        # A degenerate single point tour, zero length either way.
        points = [(5.0, 5.0)]
        assert approximation_ratio(0.0, 0.0) == 1.0
