"""
test_maxcut_randomized.py
----------------------------
Tests for the randomized Max-Cut approximation, the local search
improvement heuristic, and the exact brute force solver they get compared
against. Checks that random_cut's measured mean lands close to the
theoretical |E| / 2 expectation over many trials, that local search never
makes a cut worse, and that both stay within the bounds their proofs
promise.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.approximation.maxcut_randomized import (
    random_cut,
    expected_cut_value,
    local_search_maxcut,
    exact_maxcut_bruteforce,
)
from src.utils.graph_utils import connected_random_graph, build_adjacency


class TestRandomCut:
    def test_every_vertex_gets_assigned_a_side(self):
        vertices, edges = connected_random_graph(8, 0.3, rng=random.Random(1))
        assignment, _cut = random_cut(vertices, edges, random.Random(2))
        assert set(assignment.keys()) == set(vertices)
        assert all(side in (0, 1) for side in assignment.values())

    def test_cut_value_never_exceeds_edge_count(self):
        vertices, edges = connected_random_graph(10, 0.4, rng=random.Random(3))
        rng = random.Random(4)
        for _ in range(20):
            _assignment, cut = random_cut(vertices, edges, rng)
            assert 0 <= cut <= len(edges)

    def test_mean_cut_over_many_trials_is_close_to_expected_value(self):
        vertices, edges = connected_random_graph(14, 0.3, rng=random.Random(5))
        rng = random.Random(6)
        cuts = [random_cut(vertices, edges, rng)[1] for _ in range(500)]
        mean_cut = sum(cuts) / len(cuts)
        expected = expected_cut_value(len(edges))
        # generous tolerance, this is a statistical check, not an exact one
        assert abs(mean_cut - expected) < 0.1 * len(edges)


class TestLocalSearch:
    def test_never_produces_a_worse_cut_than_the_start(self):
        vertices, edges = connected_random_graph(10, 0.3, rng=random.Random(7))
        adjacency = build_adjacency(vertices, edges)
        rng = random.Random(8)
        for _ in range(10):
            assignment, cut = random_cut(vertices, edges, rng)
            _improved_assignment, improved_cut = local_search_maxcut(vertices, edges, assignment, adjacency)
            assert improved_cut >= cut

    def test_reaches_a_local_optimum_no_single_flip_helps(self):
        vertices, edges = connected_random_graph(8, 0.35, rng=random.Random(9))
        adjacency = build_adjacency(vertices, edges)
        assignment, _cut = random_cut(vertices, edges, random.Random(10))
        improved_assignment, _improved_cut = local_search_maxcut(vertices, edges, assignment, adjacency)

        for v in vertices:
            same_side = sum(1 for u in adjacency[v] if improved_assignment[u] == improved_assignment[v])
            other_side = sum(1 for u in adjacency[v] if improved_assignment[u] != improved_assignment[v])
            assert same_side <= other_side  # flipping v would not help


class TestExactSolver:
    def test_matches_a_known_square_graph(self):
        # A 4-cycle, 0-1-2-3-0, max cut is all 4 edges (bipartite, 2 vs 2)
        vertices = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        assert exact_maxcut_bruteforce(vertices, edges) == 4

    def test_triangle_max_cut_is_two(self):
        vertices = [0, 1, 2]
        edges = [(0, 1), (1, 2), (0, 2)]
        assert exact_maxcut_bruteforce(vertices, edges) == 2

    def test_local_search_reaches_optimal_on_small_graphs_most_of_the_time(self):
        rng = random.Random(11)
        hits = 0
        trials = 10
        for _ in range(trials):
            vertices, edges = connected_random_graph(7, 0.4, rng=rng)
            adjacency = build_adjacency(vertices, edges)
            assignment, _cut = random_cut(vertices, edges, rng)
            _improved_assignment, improved_cut = local_search_maxcut(vertices, edges, assignment, adjacency)
            optimal = exact_maxcut_bruteforce(vertices, edges)
            if improved_cut == optimal:
                hits += 1
        assert hits >= trials // 2  # not every local optimum is global, but most should be on small graphs
