"""
test_floyd_warshall.py
-------------------------
pytest tests for src/dp_advanced/floyd_warshall.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp_advanced.floyd_warshall import (
    floyd_warshall,
    reconstruct_path,
    has_negative_cycle,
    dijkstra_single_source,
    compare_with_dijkstra,
)

INF = float("inf")


def build_known_graph():
    """A small 4-node directed graph with a "trap" identical in spirit to Week 4's
    Dijkstra test fixture: the direct edge 0 -> 3 (weight 10) looks cheap, but routing
    0 -> 1 -> 2 -> 3 (5 + 3 + 1 = 9) is actually shorter.

    Hand-computed shortest distances from node 0:
        0 = 0
        1 = 5           (direct edge)
        2 = 8           (0 -> 1 -> 2)
        3 = 9           (0 -> 1 -> 2 -> 3, NOT the direct 0 -> 3 edge)
    """
    return [
        [0, 5, INF, 10],
        [INF, 0, 3, INF],
        [INF, INF, 0, 1],
        [INF, INF, INF, 0],
    ]


class TestFloydWarshallBasics:
    def test_distance_to_self_is_zero(self):
        matrix = build_known_graph()
        distances, _ = floyd_warshall(matrix)
        for i in range(len(matrix)):
            assert distances[i][i] == 0

    def test_shortest_distances_match_hand_computed_values(self):
        matrix = build_known_graph()
        distances, _ = floyd_warshall(matrix)
        assert distances[0] == [0, 5, 8, 9]

    def test_prefers_the_genuinely_shorter_routed_path(self):
        # The direct 0 -> 3 edge costs 10; routing through 1 and 2 costs 9. If this
        # fails, the DP isn't actually considering intermediate nodes.
        matrix = build_known_graph()
        distances, _ = floyd_warshall(matrix)
        assert distances[0][3] == 9

    def test_unreachable_pairs_stay_infinite(self):
        matrix = [
            [0, 1, INF],
            [INF, 0, INF],
            [INF, INF, 0],
        ]
        distances, _ = floyd_warshall(matrix)
        assert distances[0][2] == INF
        assert distances[2][0] == INF

    def test_single_node_graph(self):
        distances, _ = floyd_warshall([[0]])
        assert distances == [[0]]


class TestPathReconstruction:
    def test_reconstructs_the_routed_path(self):
        matrix = build_known_graph()
        _, predecessor = floyd_warshall(matrix)
        path = reconstruct_path(predecessor, 0, 3)
        assert path == [0, 1, 2, 3]

    def test_path_cost_matches_reported_distance(self):
        matrix = build_known_graph()
        distances, predecessor = floyd_warshall(matrix)
        path = reconstruct_path(predecessor, 0, 3)
        total = sum(matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
        assert total == distances[0][3]

    def test_unreachable_path_is_empty(self):
        matrix = [
            [0, 1, INF],
            [INF, 0, INF],
            [INF, INF, 0],
        ]
        _, predecessor = floyd_warshall(matrix)
        assert reconstruct_path(predecessor, 0, 2) == []

    def test_path_to_self_is_single_node(self):
        matrix = build_known_graph()
        _, predecessor = floyd_warshall(matrix)
        assert reconstruct_path(predecessor, 2, 2) == [2]


class TestNegativeCycles:
    def test_no_false_positive_on_a_normal_graph(self):
        matrix = build_known_graph()
        distances, _ = floyd_warshall(matrix)
        assert has_negative_cycle(distances) is False

    def test_detects_a_real_negative_cycle(self):
        # 0 -> 1 -> 2 -> 0 has total weight 1 + (-3) + (-1) = -3 -- a negative cycle.
        matrix = [
            [0, 1, INF],
            [INF, 0, -3],
            [-1, INF, 0],
        ]
        distances, _ = floyd_warshall(matrix)
        assert has_negative_cycle(distances) is True

    def test_negative_edges_without_a_cycle_are_fine(self):
        # A negative edge is fine on its own; it's only a CYCLE of negative total
        # weight that breaks shortest-path well-definedness.
        matrix = [
            [0, -5, INF],
            [INF, 0, 2],
            [INF, INF, 0],
        ]
        distances, _ = floyd_warshall(matrix)
        assert has_negative_cycle(distances) is False
        assert distances[0][2] == -3


class TestDijkstraComparison:
    """Dijkstra assumes no negative edges, so these tests only use non-negative
    graphs -- that restriction is exactly why Floyd-Warshall is more general
    (it tolerates negative edges, as long as there's no negative cycle)."""

    def test_dijkstra_matches_floyd_warshall_single_source(self):
        matrix = build_known_graph()
        distances, _ = floyd_warshall(matrix)
        dijkstra_distances = dijkstra_single_source(matrix, 0)
        assert dijkstra_distances == distances[0]

    def test_dijkstra_matches_from_every_source(self):
        matrix = build_known_graph()
        distances, _ = floyd_warshall(matrix)
        for source in range(len(matrix)):
            assert dijkstra_single_source(matrix, source) == distances[source]

    def test_compare_with_dijkstra_agrees_on_a_known_graph(self):
        matrix = build_known_graph()
        result = compare_with_dijkstra(matrix)
        assert result["distances_agree"] is True
        assert result["floyd_warshall_time"] >= 0
        assert result["dijkstra_total_time"] >= 0

    def test_compare_with_dijkstra_agrees_on_random_small_graphs(self):
        import random
        rng = random.Random(11)
        for _ in range(10):
            n = rng.randint(2, 8)
            matrix = [[0 if i == j else INF for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j and rng.random() < 0.4:
                        matrix[i][j] = rng.randint(1, 20)
            result = compare_with_dijkstra(matrix)
            assert result["distances_agree"] is True
