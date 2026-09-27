"""
test_graph_representation.py
-----------------------------
pytest tests for Graph (src/graphs/graph.py).

The whole point of this file is that Graph supports two different storage
strategies under the hood (adjacency list vs. adjacency matrix), and from
the OUTSIDE they're supposed to behave identical same nodes, same
edges, same neighbors, same weights. So most tests here run TWICE, once
per representation, using pytest's parametrize feature. If a test only
passes for one representation, that's a real bug: the two are supposed
to be interchangeable.

"""

import sys
import os

# Make sure "src" is importable when running pytest from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.graphs.graph import Graph


# Every test that takes a `representation` argument gets run once with
# "list" and once with "matrix" -- pytest calls the test function twice,
# substituting this value in each time.
REPRESENTATIONS = ["list", "matrix"]


# ---------------------------------------------------------------------------
# Empty graph / add_node
# ---------------------------------------------------------------------------

class TestAddNode:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_new_graph_starts_empty(self, representation):
        g = Graph(representation=representation)
        assert g.nodes == set()
        # ^ Nothing has been added yet, so the master node set should be empty
        #   regardless of which storage style is underneath.

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_add_node_registers_it(self, representation):
        g = Graph(representation=representation)
        g.add_node("A")
        assert "A" in g.nodes
        assert g.get_neighbors("A") == {}
        # ^ A freshly added node exists but has no edges yet.

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_adding_same_node_twice_is_a_no_op(self, representation):
        g = Graph(representation=representation)
        g.add_edge("A", "B", weight=7)
        g.add_node("A")
        # ^ Re-adding a node that already has edges must NOT wipe them out.
        #   add_node() should just see "A" is already in self.nodes and return early.
        assert g.get_neighbors("A") == {"B": 7}

    def test_add_node_grows_matrix_correctly(self):
        # Matrix-specific: check the actual grid shape, not just the public API.
        # None (not 0) means "no edge" -- see the Graph class docstring: 0 is a
        # legal edge weight, so it can't double as the empty-cell sentinel.
        g = Graph(representation="matrix")
        g.add_node("A")
        assert g.adj_matrix == [[None]]
        g.add_node("B")
        # ^ Adding a second node should widen every existing row by one column
        #   AND add a brand new row -- so the matrix stays square.
        assert g.adj_matrix == [[None, None], [None, None]]
        g.add_node("C")
        assert g.adj_matrix == [[None, None, None], [None, None, None], [None, None, None]]


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------

class TestAddEdge:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_undirected_edge_goes_both_ways(self, representation):
        g = Graph(directed=False, representation=representation)
        g.add_edge("A", "B", weight=3)
        assert g.get_neighbors("A") == {"B": 3}
        assert g.get_neighbors("B") == {"A": 3}
        # ^ Undirected means "A-B" is really the same connection seen from
        #   either end, so both sides must know about it.

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_directed_edge_goes_one_way_only(self, representation):
        g = Graph(directed=True, representation=representation)
        g.add_edge("A", "B", weight=3)
        assert g.get_neighbors("A") == {"B": 3}
        assert g.get_neighbors("B") == {}
        # ^ Directed means A -> B does NOT imply B -> A (one-way street).

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_add_edge_auto_creates_missing_nodes(self, representation):
        g = Graph(representation=representation)
        g.add_edge("A", "B")
        # ^ Neither "A" nor "B" existed before this call -- add_edge should
        #   create them automatically instead of raising an error.
        assert g.nodes == {"A", "B"}

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_default_weight_is_one(self, representation):
        g = Graph(representation=representation)
        g.add_edge("A", "B")
        # ^ No weight argument given -> should fall back to 1 (an unweighted edge).
        assert g.get_neighbors("A") == {"B": 1}

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_re_adding_edge_overwrites_weight(self, representation):
        g = Graph(representation=representation)
        g.add_edge("A", "B", weight=5)
        g.add_edge("A", "B", weight=9)
        # ^ Adding the "same" edge again with a new weight should update it,
        #   not create a duplicate or leave the old weight in place.
        assert g.get_neighbors("A") == {"B": 9}


# ---------------------------------------------------------------------------
# remove_edge
# ---------------------------------------------------------------------------

class TestRemoveEdge:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_remove_undirected_edge_clears_both_sides(self, representation):
        g = Graph(directed=False, representation=representation)
        g.add_edge("A", "B", weight=2)
        g.remove_edge("A", "B")
        assert g.get_neighbors("A") == {}
        assert g.get_neighbors("B") == {}
        # ^ Both nodes should forget about each other, since undirected
        #   edges are really one connection, not two independent ones.

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_remove_directed_edge_only_clears_the_one_direction(self, representation):
        g = Graph(directed=True, representation=representation)
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.remove_edge("A", "B")
        # ^ Removing A -> B should leave B -> A completely untouched,
        #   since in a directed graph they're two separate edges.
        assert g.get_neighbors("A") == {}
        assert g.get_neighbors("B") == {"A": 1}

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_nodes_still_exist_after_removing_their_edge(self, representation):
        g = Graph(representation=representation)
        g.add_edge("A", "B")
        g.remove_edge("A", "B")
        # ^ remove_edge should only touch the CONNECTION, not delete the nodes
        #   themselves -- they should still be around, just with no edges.
        assert g.nodes == {"A", "B"}


# ---------------------------------------------------------------------------
# remove_node
# ---------------------------------------------------------------------------

class TestRemoveNode:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_remove_node_drops_it_from_nodes(self, representation):
        g = Graph(representation=representation)
        g.add_edge("A", "B")
        g.remove_node("A")
        assert "A" not in g.nodes
        assert g.nodes == {"B"}

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_remove_node_cleans_up_dangling_references(self, representation):
        g = Graph(directed=False, representation=representation)
        g.add_edge("A", "B", weight=1)
        g.add_edge("A", "C", weight=1)
        g.add_edge("B", "C", weight=1)
        g.remove_node("A")
        # ^ B and C were both connected to A. After A is gone, neither of
        #   them should still list "A" as a neighbor -- that would be a
        #   dangling reference to a node that no longer exists.
        assert "A" not in g.get_neighbors("B")
        assert "A" not in g.get_neighbors("C")
        # The B-C edge, unrelated to A, should be completely unaffected.
        assert g.get_neighbors("B") == {"C": 1}
        assert g.get_neighbors("C") == {"B": 1}

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_remove_unknown_node_does_nothing(self, representation):
        g = Graph(representation=representation)
        g.add_node("A")
        g.remove_node("does-not-exist")
        # ^ Removing something that was never there should be a harmless
        #   no-op, not an exception.
        assert g.nodes == {"A"}

    def test_remove_node_reindexes_matrix_correctly(self):
        # Matrix-specific: this is the trickiest method in the whole class.
        # Deleting a row/column shifts every later index down by one, and
        # node_index has to be updated to match or future lookups break.
        g = Graph(directed=True, representation="matrix")
        g.add_edge("A", "B", weight=1)
        g.add_edge("B", "C", weight=1)
        g.add_edge("A", "C", weight=4)
        g.add_edge("C", "A", weight=9)
        # index before removal: A=0, B=1, C=2

        g.remove_node("B")
        # "B" was index 1, so "C" (previously index 2) must now shift to index 1.
        # None (not 0) means "no edge" here too -- see Graph's docstring.
        assert g.node_index == {"A": 0, "C": 1}
        assert g.adj_matrix == [[None, 4], [9, None]]

        # And the public API should still work correctly after the reindex --
        # this is the real test: a bug here wouldn't show up unless you
        # actually try to use the graph again after a removal.
        assert g.get_neighbors("A") == {"C": 4}
        assert g.get_neighbors("C") == {"A": 9}


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_neighbors_of_isolated_node_is_empty(self, representation):
        g = Graph(representation=representation)
        g.add_node("A")
        assert g.get_neighbors("A") == {}

    def test_get_neighbors_list_repr_returns_empty_dict_for_unknown_node(self):
        # NOTE: the two representations behave differently here on purpose
        # (or, arguably, by accident -- worth discussing with your instructor).
        # adj_list.get(node, {}) quietly returns {} for a node that was
        # never added at all.
        g = Graph(representation="list")
        assert g.get_neighbors("ghost") == {}

    def test_get_neighbors_matrix_repr_raises_for_unknown_node(self):
        # The matrix version looks up self.node_index[node], which raises
        # a KeyError if "node" was never added -- there's no equivalent
        # ".get(..., default)" safety net for the matrix path.
        # This test documents that inconsistency: it's not obviously a bug,
        # but it IS a case where the two representations diverge, which is
        # exactly the sort of thing this test file is meant to catch.
        g = Graph(representation="matrix")
        with pytest.raises(KeyError):
            g.get_neighbors("ghost")


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestStr:
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_str_includes_every_node_and_its_neighbors(self, representation):
        g = Graph(directed=False, representation=representation)
        g.add_edge("A", "B", weight=1)
        text = str(g)
        # We don't pin down the exact line order (self.nodes is a set, so
        # iteration order isn't guaranteed), but every node's line and its
        # correct neighbor dict must show up somewhere in the output.
        assert "A -> {'B': 1}" in text
        assert "B -> {'A': 1}" in text
