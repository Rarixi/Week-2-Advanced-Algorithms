"""
test_network.py
------------------
Tests for core/network.py. Checks the residual edge bookkeeping itself,
add_edge's forward/backward pairing, push_flow's residual capacity
updates, reset_flow, get_residual_graph, and the flow validation helper,
independent of any algorithm that uses them.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.network import Network, is_valid_flow


class TestAddEdge:
    def test_creates_a_paired_backward_edge(self):
        network = Network()
        fwd, bwd = network.add_edge("a", "b", 10.0)
        assert network.edges[fwd].u == "a"
        assert network.edges[fwd].v == "b"
        assert network.edges[fwd].capacity == 10.0
        assert network.edges[bwd].u == "b"
        assert network.edges[bwd].v == "a"
        assert network.edges[bwd].capacity == 0.0

    def test_reverse_of_is_its_own_inverse(self):
        network = Network()
        fwd, bwd = network.add_edge("a", "b", 5.0)
        assert network.reverse_of(fwd) == bwd
        assert network.reverse_of(bwd) == fwd

    def test_parallel_edges_get_independent_residual_pairs(self):
        network = Network()
        fwd1, _ = network.add_edge("a", "b", 3.0)
        fwd2, _ = network.add_edge("a", "b", 7.0)
        assert fwd1 != fwd2
        assert network.residual_capacity(fwd1) == 3.0
        assert network.residual_capacity(fwd2) == 7.0

    def test_negates_cost_on_the_backward_edge(self):
        network = Network()
        fwd, bwd = network.add_edge("a", "b", 5.0, cost=2.5)
        assert network.edges[fwd].cost == 2.5
        assert network.edges[bwd].cost == -2.5


class TestPushFlow:
    def test_reduces_forward_residual_and_credits_backward(self):
        network = Network()
        fwd, bwd = network.add_edge("a", "b", 10.0)
        network.push_flow(fwd, 4.0)
        assert network.residual_capacity(fwd) == 6.0
        assert network.residual_capacity(bwd) == 4.0

    def test_rejects_pushing_more_than_residual_capacity(self):
        network = Network()
        fwd, _ = network.add_edge("a", "b", 5.0)
        try:
            network.push_flow(fwd, 5.1)
            assert False, "expected a ValueError"
        except ValueError:
            pass

    def test_pushing_along_backward_edge_undoes_forward_flow(self):
        network = Network()
        fwd, bwd = network.add_edge("a", "b", 10.0)
        network.push_flow(fwd, 6.0)
        # pushing 2.0 back along bwd undoes 2.0 of the forward flow,
        # leaving 4.0 of forward flow in place and forward residual back
        # up to 6.0
        network.push_flow(bwd, 2.0)
        assert network.residual_capacity(fwd) == 6.0
        assert network.residual_capacity(bwd) == 4.0


class TestResetFlow:
    def test_zeroes_every_edge(self):
        network = Network()
        fwd, _ = network.add_edge("a", "b", 10.0)
        network.push_flow(fwd, 7.0)
        network.reset_flow()
        assert all(f == 0.0 for f in network.flow)
        assert network.residual_capacity(fwd) == 10.0


class TestResidualGraph:
    def test_aggregates_parallel_and_backward_edges(self):
        network = Network()
        fwd1, bwd1 = network.add_edge("a", "b", 3.0)
        fwd2, _bwd2 = network.add_edge("a", "b", 2.0)
        network.push_flow(fwd1, 3.0)
        residual = network.get_residual_graph()
        # a -> b residual: fwd2 still has 2.0 left, fwd1 is drained
        assert residual["a"]["b"] == 2.0
        # b -> a residual: bwd1 now carries 3.0 of undo capacity
        assert residual["b"]["a"] == 3.0


class TestFlowMatrixAndCost:
    def test_flow_matrix_only_includes_positive_flow_edges(self):
        network = Network()
        fwd, _ = network.add_edge("a", "b", 10.0)
        network.add_edge("b", "c", 10.0)
        network.push_flow(fwd, 4.0)
        matrix = network.flow_matrix()
        assert matrix == {("a", "b"): 4.0}

    def test_total_cost_sums_only_forward_flow(self):
        network = Network()
        fwd, _ = network.add_edge("a", "b", 10.0, cost=3.0)
        network.push_flow(fwd, 4.0)
        assert network.total_cost() == 12.0


class TestIsValidFlow:
    def test_a_correctly_pushed_flow_is_valid(self):
        network = Network()
        e1, _ = network.add_edge("s", "a", 5.0)
        e2, _ = network.add_edge("a", "t", 5.0)
        network.push_flow(e1, 3.0)
        network.push_flow(e2, 3.0)
        assert is_valid_flow(network, "s", "t")

    def test_broken_conservation_is_caught(self):
        network = Network()
        e1, _ = network.add_edge("s", "a", 5.0)
        network.add_edge("a", "t", 5.0)
        # push flow into a but never out, violating conservation at a
        network.push_flow(e1, 3.0)
        assert not is_valid_flow(network, "s", "t")
