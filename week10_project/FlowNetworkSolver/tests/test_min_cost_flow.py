"""
test_min_cost_flow.py
------------------------
Tests for core/min_cost_flow.py. Checks a hand built instance with a
cheap and an expensive path where the cheap one should be exhausted
first, checks that min_cost_max_flow's reported value matches a plain
max-flow run on the same topology when every cost is zero (cost
shouldn't change how much flow gets through, only which paths carry
it), and checks the returned flow is valid and its cost matches what
utils/validators.py computes directly from the network.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.edmonds_karp import edmonds_karp_max_flow
from core.min_cost_flow import min_cost_max_flow
from core.network import Network, is_valid_flow
from utils.generators import random_flow_network
from utils.validators import check_min_cost_feasibility


def _copy_network(network):
    copy = Network()
    for node in network.nodes:
        copy.add_node(node)
    for edge in network.edges[::2]:
        copy.add_edge(edge.u, edge.v, edge.capacity, edge.cost)
    return copy


class TestCheapPathPreferredFirst:
    def test_cheap_and_expensive_parallel_paths(self):
        # s -> a -> t costs 1 per unit, capacity 3
        # s -> b -> t costs 5 per unit, capacity 3
        # total capacity 6, but a cost minimizing solver should fill the
        # cheap path completely (3 units at cost 1) before touching the
        # expensive one (3 units at cost 5) if it needs more flow at all,
        # min cost MAX flow still has to use both paths to reach max flow 6
        network = Network()
        network.add_edge("s", "a", 3.0, cost=1.0)
        network.add_edge("a", "t", 3.0, cost=1.0)
        network.add_edge("s", "b", 3.0, cost=5.0)
        network.add_edge("b", "t", 3.0, cost=5.0)

        result = min_cost_max_flow(network, "s", "t")

        assert result["value"] == 6.0
        # cheap path costs 2 per unit through it (1 + 1), expensive 10 per unit
        expected_cost = 3.0 * 2.0 + 3.0 * 10.0
        assert abs(result["cost"] - expected_cost) < 1e-6
        assert is_valid_flow(network, "s", "t")
        assert check_min_cost_feasibility(network, result["cost"])

    def test_cheaper_detour_is_preferred_over_direct_expensive_edge(self):
        # s -> t directly costs 10 per unit, capacity 5
        # s -> m -> t costs 1 + 1 = 2 per unit, capacity 2
        # optimal use of 2 units of flow should go entirely through the
        # detour (cost 4) rather than the direct edge (cost 20)
        network = Network()
        network.add_edge("s", "t", 5.0, cost=10.0)
        network.add_edge("s", "m", 2.0, cost=1.0)
        network.add_edge("m", "t", 2.0, cost=1.0)

        result = min_cost_max_flow(network, "s", "t")

        assert result["value"] == 7.0  # 5 direct + 2 through the detour
        expected_cost = 5.0 * 10.0 + 2.0 * 2.0
        assert abs(result["cost"] - expected_cost) < 1e-6


class TestZeroCostMatchesPlainMaxFlow:
    def test_zero_cost_value_matches_edmonds_karp(self):
        rng = random.Random(30)
        for _ in range(6):
            n = rng.randint(4, 9)
            ek_network, source, sink = random_flow_network(n, rng.uniform(0.3, 0.6), rng=rng)
            mcf_network = _copy_network(ek_network)

            ek_result = edmonds_karp_max_flow(ek_network, source, sink)
            mcf_result = min_cost_max_flow(mcf_network, source, sink)

            assert abs(ek_result["value"] - mcf_result["value"]) < 1e-6
            assert abs(mcf_result["cost"]) < 1e-6
            assert is_valid_flow(mcf_network, source, sink)


class TestWithRandomCosts:
    def test_valid_flow_and_feasible_cost_on_random_cost_networks(self):
        rng = random.Random(31)
        for _ in range(6):
            n = rng.randint(4, 8)
            network, source, sink = random_flow_network(
                n, rng.uniform(0.3, 0.6), cost_range=(1.0, 9.0), rng=rng
            )
            result = min_cost_max_flow(network, source, sink)
            assert is_valid_flow(network, source, sink)
            assert check_min_cost_feasibility(network, result["cost"])
