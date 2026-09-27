"""
test_dfs.py
-----------
pytest tests for dfs_iterative(), dfs_recursive(), and dfs_full() (src/graphs/dfs.py).

Same shape as test_bfs.py: most tests run once per Graph representation ("list" and
"matrix") via parametrize. On top of that, several tests here also run once per DFS
*implementation* (iterative vs. recursive), since the whole point of having two versions
is that they're supposed to agree on every graph -- if they ever disagree, that's a real
bug in one of them, not just a style difference.

"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.graphs.graph import Graph
from src.graphs.dfs import dfs_iterative, dfs_recursive, dfs_full

REPRESENTATIONS = ["list", "matrix"]
DFS_IMPLEMENTATIONS = [dfs_iterative, dfs_recursive]


def build_tree_graph(representation):
    """Same fixture as test_bfs.py's build_tree_graph -- see that docstring for why
    nodes are introduced in this exact order (it keeps list/matrix neighbor order
    identical, so we can assert one specific traversal order below):
        A -> B, A -> C
        B -> D, B -> E
        C -> F
    """
    g = Graph(directed=True, representation=representation)
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]:
        g.add_edge(u, v)
    return g


def build_disconnected_graph(representation):
    """Two separate components: A-B-C (a little chain) and X-Y (a separate pair)."""
    g = Graph(directed=False, representation=representation)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("X", "Y")
    return g


class TestDFSBasics:
    @pytest.mark.parametrize("dfs_fn", DFS_IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_from_unknown_start_returns_empty(self, representation, dfs_fn):
        g = Graph(representation=representation)
        g.add_node("A")
        assert dfs_fn(g, "does-not-exist") == []

    @pytest.mark.parametrize("dfs_fn", DFS_IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_on_single_node_graph(self, representation, dfs_fn):
        g = Graph(representation=representation)
        g.add_node("A")
        assert dfs_fn(g, "A") == ["A"]

    @pytest.mark.parametrize("dfs_fn", DFS_IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_visits_each_node_exactly_once(self, representation, dfs_fn):
        g = build_tree_graph(representation)
        order = dfs_fn(g, "A")
        assert len(order) == len(set(order))
        assert set(order) == g.nodes

    @pytest.mark.parametrize("dfs_fn", DFS_IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_goes_deep_before_wide(self, representation, dfs_fn):
        g = build_tree_graph(representation)
        order = dfs_fn(g, "A")
        pos = {node: i for i, node in enumerate(order)}
        # The defining property of DFS, as opposed to BFS: once we head down the A -> B
        # branch, we should reach one of B's children (D or E) before we ever back up to
        # visit A's OTHER branch (C). A breadth-first order would visit C right after B;
        # a depth-first order must not.
        assert pos["B"] < pos["C"]
        assert min(pos["D"], pos["E"]) < pos["C"]

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_exact_order_on_known_graph(self, representation):
        # Fixture built so list/matrix neighbor order matches (see build_tree_graph),
        # so we can pin down one exact expected order for both DFS implementations.
        g = build_tree_graph(representation)
        expected = ["A", "B", "D", "E", "C", "F"]
        assert dfs_iterative(g, "A") == expected
        assert dfs_recursive(g, "A") == expected

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_iterative_and_recursive_agree(self, representation):
        # The two implementations are supposed to be interchangeable -- same graph, same
        # start node, same traversal order. This is really the core test of this file:
        # if this ever fails, one of the two DFS implementations has a real bug.
        g = build_tree_graph(representation)
        assert dfs_iterative(g, "A") == dfs_recursive(g, "A")


class TestDFSDisconnectedGraphs:
    @pytest.mark.parametrize("dfs_fn", DFS_IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_from_one_node_misses_other_components(self, representation, dfs_fn):
        g = build_disconnected_graph(representation)
        order = dfs_fn(g, "A")
        # A plain single-start DFS only sees its own component -- X and Y live in a
        # completely separate one. dfs_full() (tested below) is what fixes this.
        assert set(order) == {"A", "B", "C"}
        assert "X" not in order
        assert "Y" not in order

    @pytest.mark.parametrize("algorithm", ["iterative", "recursive"])
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_full_covers_every_component(self, representation, algorithm):
        g = build_disconnected_graph(representation)
        order = dfs_full(g, algorithm=algorithm)
        # This is what "handle disconnected graphs gracefully" means for DFS: every node
        # gets visited, even the ones no single start node could have reached.
        assert set(order) == {"A", "B", "C", "X", "Y"}
        assert len(order) == 5

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_full_keeps_each_component_contiguous(self, representation):
        g = build_disconnected_graph(representation)
        order = dfs_full(g)
        pos = {node: i for i, node in enumerate(order)}
        abc_positions = sorted(pos[n] for n in ("A", "B", "C"))
        xy_positions = sorted(pos[n] for n in ("X", "Y"))
        assert abc_positions == list(range(abc_positions[0], abc_positions[0] + 3))
        assert xy_positions == list(range(xy_positions[0], xy_positions[0] + 2))


class TestDFSCycles:
    @pytest.mark.parametrize("dfs_fn", DFS_IMPLEMENTATIONS)
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_dfs_terminates_and_visits_once_on_a_cycle(self, representation, dfs_fn):
        # A -> B -> C -> A is a cycle. Without the `visited` check, dfs_recursive would
        # recurse forever and dfs_iterative would loop forever; both should instead just
        # visit each of the three nodes exactly once.
        g = Graph(directed=True, representation=representation)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        order = dfs_fn(g, "A")
        assert sorted(order) == ["A", "B", "C"]
        assert len(order) == 3
