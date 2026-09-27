"""
test_dijkstra.py
-----------------
pytest tests for dijkstra() and dijkstra_list_based() (src/graphs/dijkstra.py).

Like the other test files in this project, most tests run once per Graph representation
("list" and "matrix") -- Dijkstra's correctness shouldn't depend on which storage strategy
the graph uses underneath. On top of that, several tests also run once per *implementation*
(heap-based vs. list-based): the whole reason both exist is that they're supposed to
compute IDENTICAL results, just at different speeds -- that's the premise the benchmark in
Part 3 depends on, so it's worth testing directly, not just assuming.

"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.graphs.graph import Graph
from src.graphs.dijkstra import dijkstra, dijkstra_list_based

REPRESENTATIONS = ["list", "matrix"]
IMPLEMENTATIONS = [dijkstra, dijkstra_list_based]


def build_weighted_graph(representation):
    """A small directed, weighted graph with a "trap": the direct edge A -> B (weight 4)
    looks cheaper than it is -- going A -> C -> B (1 + 2 = 3) is actually shorter. Any
    implementation that just takes the first edge it sees instead of genuinely finding the
    shortest path will get B (and everything downstream of it) wrong.

        A -> B (4)
        A -> C (1)
        C -> B (2)
        B -> D (5)
        C -> D (8)
        D -> E (3)
        B -> E (10)

    Hand-computed shortest distances from A:
        A = 0
        C = 1                  (A -> C)
        B = 3                  (A -> C -> B, NOT the direct A -> B edge)
        D = 8                  (A -> C -> B -> D)
        E = 11                 (A -> C -> B -> D -> E, NOT the direct B -> E edge)
    """
    g = Graph(directed=True, representation=representation)
    for u, v, w in [
        ("A", "B", 4),
        ("A", "C", 1),
        ("C", "B", 2),
        ("B", "D", 5),
        ("C", "D", 8),
        ("D", "E", 3),
        ("B", "E", 10),
    ]:
        g.add_edge(u, v, w)
    return g


def build_disconnected_graph(representation):
    """Two components: A -> B -> C (reachable from A) and X -> Y (completely separate)."""
    g = Graph(directed=True, representation=representation)
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", 1)
    g.add_edge("X", "Y", 1)
    return g


def reconstruct_path(predecessors, destination):
    """Walk predecessors backward from `destination` to whatever node has no predecessor
    (the source, or an unreachable node stuck at its own starting point)."""
    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = predecessors[node]
    path.reverse()
    return path


class TestDijkstraBasics:
    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_distance_to_source_is_zero(self, representation, dijkstra_fn):
        g = build_weighted_graph(representation)
        distances, predecessors = dijkstra_fn(g, "A")
        assert distances["A"] == 0
        assert predecessors["A"] is None

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_shortest_distances_match_hand_computed_values(self, representation, dijkstra_fn):
        g = build_weighted_graph(representation)
        distances, _ = dijkstra_fn(g, "A")
        # See build_weighted_graph's docstring for how these were computed by hand.
        assert distances == {"A": 0, "B": 3, "C": 1, "D": 8, "E": 11}

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_prefers_the_genuinely_shorter_indirect_path(self, representation, dijkstra_fn):
        # The direct A -> B edge has weight 4; A -> C -> B costs only 3. If this test
        # fails, the implementation is taking the first edge it finds rather than
        # actually comparing path costs.
        g = build_weighted_graph(representation)
        distances, predecessors = dijkstra_fn(g, "A")
        assert distances["B"] == 3
        assert predecessors["B"] == "C"
        assert predecessors["C"] == "A"

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_predecessor_chain_reconstructs_a_correct_shortest_path(self, representation, dijkstra_fn):
        g = build_weighted_graph(representation)
        distances, predecessors = dijkstra_fn(g, "A")

        path = reconstruct_path(predecessors, "E")
        assert path == ["A", "C", "B", "D", "E"]

        # Cross-check: manually summing the edge weights along the reconstructed path
        # should equal the reported distance -- this is what actually matters for path
        # reconstruction being *useful*, not just "predecessors happens to look right."
        total = 0
        for u, v in zip(path, path[1:]):
            total += g.get_neighbors(u)[v]
        assert total == distances["E"]

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_single_node_graph(self, representation, dijkstra_fn):
        g = Graph(directed=True, representation=representation)
        g.add_node("A")
        distances, predecessors = dijkstra_fn(g, "A")
        assert distances == {"A": 0}
        assert predecessors == {"A": None}

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_unknown_source_leaves_every_distance_infinite(self, representation, dijkstra_fn):
        g = build_weighted_graph(representation)
        distances, predecessors = dijkstra_fn(g, "does-not-exist")
        assert all(d == float("inf") for d in distances.values())
        assert all(p is None for p in predecessors.values())


class TestDijkstraDisconnectedGraphs:
    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_unreachable_nodes_get_infinite_distance_and_no_predecessor(self, representation, dijkstra_fn):
        g = build_disconnected_graph(representation)
        distances, predecessors = dijkstra_fn(g, "A")

        # Reachable component: real, finite distances.
        assert distances["A"] == 0
        assert distances["B"] == 1
        assert distances["C"] == 2

        # X and Y live in a completely separate component -- unreachable from A. This is
        # what "handle disconnected graphs gracefully" means here: no crash, no missing
        # keys, just an honest float('inf') and no predecessor.
        assert distances["X"] == float("inf")
        assert distances["Y"] == float("inf")
        assert predecessors["X"] is None
        assert predecessors["Y"] is None

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_every_node_appears_in_the_result_even_if_unreachable(self, representation, dijkstra_fn):
        g = build_disconnected_graph(representation)
        distances, predecessors = dijkstra_fn(g, "A")
        # "Distances from source to ALL nodes" -- X and Y must still show up as keys,
        # not be silently dropped just because they're unreachable.
        assert set(distances.keys()) == g.nodes
        assert set(predecessors.keys()) == g.nodes


class TestDijkstraEdgeWeights:
    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_negative_weight_raises_value_error(self, representation, dijkstra_fn):
        g = Graph(directed=True, representation=representation)
        g.add_edge("A", "B", weight=-5)
        with pytest.raises(ValueError):
            dijkstra_fn(g, "A")

    @pytest.mark.parametrize("dijkstra_fn", IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_zero_weight_edges_are_allowed(self, representation, dijkstra_fn):
        g = Graph(directed=True, representation=representation)
        g.add_edge("A", "B", weight=0)
        g.add_edge("B", "C", weight=0)
        distances, _ = dijkstra_fn(g, "A")
        assert distances == {"A": 0, "B": 0, "C": 0}


class TestHeapAndListImplementationsAgree:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_agree_on_the_fixture_graph(self, representation):
        g = build_weighted_graph(representation)
        d1, p1 = dijkstra(g, "A")
        d2, p2 = dijkstra_list_based(g, "A")
        assert d1 == d2
        assert p1 == p2

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_agree_on_random_graphs(self, representation):
        # The heap-based and list-based versions are supposed to be interchangeable --
        # same graph, same source, same distances and predecessors, just different
        # internal bookkeeping. This is really the core test of this class: if it ever
        # fails, one of the two implementations has a real bug, and the benchmark
        # comparing their speed would be meaningless without this guarantee.
        random.seed(1234)
        for _ in range(10):
            n = random.randint(2, 25)
            g = Graph(directed=True, representation=representation)
            for i in range(n):
                g.add_node(i)
            for _ in range(n * 3):
                u, v = random.randrange(n), random.randrange(n)
                if u == v:
                    continue
                g.add_edge(u, v, weight=random.randint(0, 20))

            source = random.randrange(n)
            d1, p1 = dijkstra(g, source)
            d2, p2 = dijkstra_list_based(g, source)
            assert d1 == d2
            assert p1 == p2
