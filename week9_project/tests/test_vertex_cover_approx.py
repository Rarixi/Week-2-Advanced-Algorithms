"""
test_vertex_cover_approx.py
------------------------------
Tests for the greedy 2-approximation vertex cover algorithm and the exact
brute force solver it gets compared against. Checks that both produce
actual valid covers, and that the approximation never exceeds the 2x
bound it's supposed to guarantee.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.approximation.vertex_cover_2approx import vertex_cover_greedy_approx, exact_minimum_vertex_cover
from src.approximation.facility_location_greedy import facility_location_greedy, exact_facility_location_bruteforce
from src.utils.graph_utils import connected_random_graph, is_vertex_cover, random_facility_location_instance


class TestGreedyApprox:
    def test_returns_a_valid_cover(self):
        vertices, edges = connected_random_graph(10, 0.3, rng=random.Random(5))
        cover = vertex_cover_greedy_approx(vertices, edges)
        assert is_vertex_cover(cover, edges)

    def test_empty_graph_gives_empty_cover(self):
        cover = vertex_cover_greedy_approx([0, 1, 2], [])
        assert cover == set()

    def test_single_edge_covers_with_both_endpoints(self):
        cover = vertex_cover_greedy_approx([0, 1], [(0, 1)])
        assert cover == {0, 1}


class TestExactSolver:
    def test_returns_a_valid_cover(self):
        vertices, edges = connected_random_graph(8, 0.3, rng=random.Random(9))
        size, cover = exact_minimum_vertex_cover(vertices, edges)
        assert is_vertex_cover(cover, edges)
        assert len(cover) == size

    def test_triangle_needs_exactly_two_vertices(self):
        vertices = [0, 1, 2]
        edges = [(0, 1), (1, 2), (0, 2)]
        size, cover = exact_minimum_vertex_cover(vertices, edges)
        assert size == 2
        assert is_vertex_cover(cover, edges)


class TestApproximationRatioBound:
    def test_greedy_never_exceeds_twice_the_optimal_across_several_graphs(self):
        rng = random.Random(11)
        for _ in range(6):
            n = rng.randint(5, 11)
            vertices, edges = connected_random_graph(n, rng.uniform(0.2, 0.5), rng=rng)
            if not edges:
                continue
            approx_cover = vertex_cover_greedy_approx(vertices, edges)
            exact_size, _ = exact_minimum_vertex_cover(vertices, edges)
            assert len(approx_cover) <= 2 * exact_size


class TestFacilityLocationGreedy:
    # facility_location_greedy.py has no test file of its own in this
    # project's structure, it shares Part 1 with vertex cover, so its
    # tests live here instead.

    def test_greedy_never_beats_the_true_optimal(self):
        rng = random.Random(6)
        for _ in range(5):
            costs, distances = random_facility_location_instance(
                rng.randint(4, 7), rng.randint(6, 10), rng=rng
            )
            _open_f, greedy_cost = facility_location_greedy(costs, distances)
            _exact_open, exact_cost = exact_facility_location_bruteforce(costs, distances)
            assert greedy_cost >= exact_cost - 1e-9

    def test_opens_at_least_one_facility(self):
        costs, distances = random_facility_location_instance(5, 8, rng=random.Random(2))
        open_facilities, cost = facility_location_greedy(costs, distances)
        assert len(open_facilities) >= 1
        assert cost < float("inf")

    def test_exact_solver_matches_brute_force_on_two_facilities(self):
        # Two facilities, cheap vs expensive, clients closer to the cheap
        # one, exact solver should just open the cheap facility alone if
        # that beats opening both.
        costs = [10.0, 1000.0]
        distances = [[1.0, 500.0], [2.0, 500.0], [1.5, 500.0]]
        open_facilities, cost = exact_facility_location_bruteforce(costs, distances)
        assert open_facilities == {0}
        assert abs(cost - (10.0 + 1.0 + 2.0 + 1.5)) < 1e-9
