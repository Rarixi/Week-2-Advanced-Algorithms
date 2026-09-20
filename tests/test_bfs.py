"""
test_bfs.py
-----------
pytest tests for bfs() and bfs_full() (src/graphs/bfs.py).

Like test_graph_representation.py, most tests here run once per Graph representation
("list" and "matrix") via parametrize -- BFS's correctness shouldn't depend on which
storage strategy the graph underneath happens to use.

Run with:
    pytest tests/test_bfs.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.graphs.graph import Graph
from src.graphs.bfs import bfs, bfs_full

REPRESENTATIONS = ["list", "matrix"]


def build_tree_graph(representation):
    """A small directed graph shaped like a tree:
        A -> B, A -> C
        B -> D, B -> E
        C -> F
    Nodes are introduced in the same order edges are added (A, B, C, D, E, F), which
    keeps neighbor iteration order identical between the list and matrix representations
    -- that's what lets us assert one exact, specific traversal order below instead of
    just checking "the right nodes got visited, in some order or other."
    """
    g = Graph(directed=True, representation=representation)
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]:
        g.add_edge(u, v)
    return g


def build_disconnected_graph(representation):
    """Two separate components: A-B-C (a little chain) and X-Y (a separate pair),
    with no edges between the two groups at all."""
    g = Graph(directed=False, representation=representation)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("X", "Y")
    return g


class TestBFSBasics:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_from_unknown_start_returns_empty(self, representation):
        g = Graph(representation=representation)
        g.add_node("A")
        assert bfs(g, "does-not-exist") == []
        # ^ Starting from a node that was never added shouldn't crash -- just nothing to see.

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_on_single_node_graph(self, representation):
        g = Graph(representation=representation)
        g.add_node("A")
        assert bfs(g, "A") == ["A"]

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_visits_each_node_exactly_once(self, representation):
        g = build_tree_graph(representation)
        order = bfs(g, "A")
        assert len(order) == len(set(order))
        # ^ No duplicates -- every node should be enqueued (and therefore visited) once,
        #   even though multiple nodes might point at the same neighbor.
        assert set(order) == g.nodes

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_visits_nearest_nodes_first(self, representation):
        g = build_tree_graph(representation)
        order = bfs(g, "A")
        # This is the actual defining property of BFS: A's direct neighbors (B, C) must
        # both appear before EITHER of B's or C's own neighbors (D, E, F) -- breadth
        # before depth. We check this structurally instead of just hardcoding the exact
        # list, so the test still means something if the fixture graph changes shape.
        pos = {node: i for i, node in enumerate(order)}
        assert pos["A"] < pos["B"] < pos["D"]
        assert pos["A"] < pos["B"] < pos["E"]
        assert pos["A"] < pos["C"] < pos["F"]
        assert pos["B"] < pos["D"] and pos["B"] < pos["E"]
        # And, specific to this fixture: both of A's direct neighbors (distance 1) must
        # come before any of the distance-2 nodes.
        assert max(pos["B"], pos["C"]) < min(pos["D"], pos["E"], pos["F"])

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_exact_order_on_known_graph(self, representation):
        # This fixture graph was specifically built so list/matrix neighbor order lines
        # up (see build_tree_graph's docstring), so we can pin down one exact order.
        g = build_tree_graph(representation)
        assert bfs(g, "A") == ["A", "B", "C", "D", "E", "F"]


class TestBFSDisconnectedGraphs:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_from_one_node_misses_other_components(self, representation):
        g = build_disconnected_graph(representation)
        order = bfs(g, "A")
        # ^ A plain bfs() call only ever sees what's reachable from its start node -- X and
        #   Y live in a totally separate component, so they should NOT show up here. This
        #   is the behavior bfs_full() exists specifically to work around.
        assert set(order) == {"A", "B", "C"}
        assert "X" not in order
        assert "Y" not in order

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_full_covers_every_component(self, representation):
        g = build_disconnected_graph(representation)
        order = bfs_full(g)
        # ^ bfs_full() is what "handle disconnected graphs gracefully" actually means in
        #   practice: nothing gets silently left out just because it wasn't reachable from
        #   wherever traversal happened to start.
        assert set(order) == {"A", "B", "C", "X", "Y"}
        assert len(order) == 5

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_full_keeps_each_component_contiguous(self, representation):
        g = build_disconnected_graph(representation)
        order = bfs_full(g)
        pos = {node: i for i, node in enumerate(order)}
        # Within a single call to bfs_full, one component should be fully explored
        # before the next one starts -- so no node from {A,B,C} should be interleaved
        # with a node from {X,Y}.
        abc_positions = sorted(pos[n] for n in ("A", "B", "C"))
        xy_positions = sorted(pos[n] for n in ("X", "Y"))
        assert abc_positions == list(range(abc_positions[0], abc_positions[0] + 3))
        assert xy_positions == list(range(xy_positions[0], xy_positions[0] + 2))


class TestBFSCycles:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_bfs_terminates_and_visits_once_on_a_cycle(self, representation):
        # A -> B -> C -> A is a cycle. Without the `visited` check, this would loop
        # forever; BFS should just visit each of the three nodes exactly once.
        g = Graph(directed=True, representation=representation)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        order = bfs(g, "A")
        assert sorted(order) == ["A", "B", "C"]
        assert len(order) == 3
