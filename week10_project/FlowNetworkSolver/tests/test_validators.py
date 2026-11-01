"""
test_validators.py
---------------------
Tests for utils/validators.py. Builds small networks, pushes flow by
hand or via a real algorithm, and checks each validator both accepts a
genuinely valid flow and catches a deliberately broken one.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.edmonds_karp import edmonds_karp_max_flow
from core.min_cost_flow import min_cost_max_flow
from core.network import Network
from utils.validators import (
    check_flow_conservation,
    check_capacity_constraints,
    check_residual_correctness,
    check_min_cost_feasibility,
)


class TestFlowConservation:
    def test_passes_on_a_correctly_pushed_flow(self):
        network = Network()
        e1, _ = network.add_edge("s", "a", 5.0)
        e2, _ = network.add_edge("a", "t", 5.0)
        network.push_flow(e1, 3.0)
        network.push_flow(e2, 3.0)
        assert check_flow_conservation(network, "s", "t")

    def test_catches_an_unbalanced_internal_node(self):
        network = Network()
        e1, _ = network.add_edge("s", "a", 5.0)
        network.add_edge("a", "t", 5.0)
        network.push_flow(e1, 3.0)
        assert not check_flow_conservation(network, "s", "t")


class TestCapacityConstraints:
    def test_passes_within_capacity(self):
        network = Network()
        e1, _ = network.add_edge("s", "t", 5.0)
        network.push_flow(e1, 5.0)
        assert check_capacity_constraints(network)

    def test_catches_flow_exceeding_capacity(self):
        network = Network()
        e1, _ = network.add_edge("s", "t", 5.0)
        network.flow[e1] = 7.0  # directly corrupt the flow, bypassing push_flow's own guard
        assert not check_capacity_constraints(network)


class TestResidualCorrectness:
    def test_passes_after_a_real_algorithm_run(self):
        network = Network()
        network.add_edge("s", "a", 4.0)
        network.add_edge("a", "t", 4.0)
        edmonds_karp_max_flow(network, "s", "t")
        assert check_residual_correctness(network)

    def test_catches_a_broken_residual_pair(self):
        network = Network()
        fwd, bwd = network.add_edge("s", "t", 5.0)
        network.flow[fwd] = 3.0
        network.flow[bwd] = -1.0  # should be -3.0 to match, deliberately wrong
        assert not check_residual_correctness(network)


class TestMinCostFeasibility:
    def test_passes_with_the_algorithms_own_reported_cost(self):
        network = Network()
        network.add_edge("s", "a", 3.0, cost=2.0)
        network.add_edge("a", "t", 3.0, cost=2.0)
        result = min_cost_max_flow(network, "s", "t")
        assert check_min_cost_feasibility(network, result["cost"])

    def test_fails_against_a_wrong_expected_cost(self):
        network = Network()
        network.add_edge("s", "a", 3.0, cost=2.0)
        network.add_edge("a", "t", 3.0, cost=2.0)
        result = min_cost_max_flow(network, "s", "t")
        assert not check_min_cost_feasibility(network, result["cost"] + 5.0)
