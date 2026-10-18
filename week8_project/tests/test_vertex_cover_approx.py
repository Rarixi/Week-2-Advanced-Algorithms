"""
test_vertex_cover_approx.py
------------------------------
Tests for the greedy 2-approximation vertex cover algorithm and the exact
brute force solver it gets compared against. Checks that both produce
actual valid covers, and that the approximation never exceeds the 2x bound
the algorithm is supposed to guarantee.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.complexity.vertex_cover_approx import vertex_cover_greedy_approx, exact_minimum_vertex_cover
from src.utils.graph_utils import connected_random_graph, is_vertex_cover


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

    def test_finds_smaller_cover_than_greedy_in_general(self):
        # The exact solver should never do worse than greedy, since greedy
        # is only guaranteed to be within 2x, not optimal.
        vertices, edges = connected_random_graph(10, 0.35, rng=random.Random(3))
        approx_cover = vertex_cover_greedy_approx(vertices, edges)
        exact_size, _exact_cover = exact_minimum_vertex_cover(vertices, edges)
        assert exact_size <= len(approx_cover)


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
