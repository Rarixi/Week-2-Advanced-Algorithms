"""
test_edmonds_karp.py
-----------------------
Tests for core/edmonds_karp.py. Checks a few hand built networks with
known max-flow values, cross checks against the independent DFS
reference solver in conftest.py on random instances, and checks the
returned flow is actually valid and the trace structure makes sense.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import reference_max_flow_dfs
from core.edmonds_karp import edmonds_karp_max_flow
from core.network import Network, is_valid_flow
from utils.generators import random_flow_network


def _diamond_network():
    # s -> a -> t and s -> b -> t, bottlenecks 3 and 2, max flow 5
    network = Network()
    network.add_edge("s", "a", 3.0)
    network.add_edge("a", "t", 10.0)
    network.add_edge("s", "b", 10.0)
    network.add_edge("b", "t", 2.0)
    return network


def _shared_bottleneck_network():
    # s -> a -> t and s -> b -> t, both routed through a shared middle
    # edge a -> b of capacity 1, so max flow is bounded by that single
    # link's capacity plus whatever each side's remaining direct
    # capacity to t can carry.
    network = Network()
    network.add_edge("s", "a", 4.0)
    network.add_edge("s", "b", 4.0)
    network.add_edge("a", "t", 2.0)
    network.add_edge("b", "t", 2.0)
    network.add_edge("a", "b", 10.0)
    return network


class TestKnownInstances:
    def test_diamond_network_max_flow_is_five(self):
        network = _diamond_network()
        result = edmonds_karp_max_flow(network, "s", "t")
        assert result["value"] == 5.0
        assert is_valid_flow(network, "s", "t")

    def test_shared_bottleneck_network_max_flow_is_four(self):
        # s->a (4), s->b (4), a->t (2), b->t (2), a->b (10)
        # a can push 2 directly to t and route its other 2 through b,
        # b can push its own 2 directly to t, giving a->t 2 + b->t 2 = 4
        network = _shared_bottleneck_network()
        result = edmonds_karp_max_flow(network, "s", "t")
        assert result["value"] == 4.0
        assert is_valid_flow(network, "s", "t")

    def test_disconnected_sink_gives_zero_flow(self):
        network = Network()
        network.add_edge("s", "a", 5.0)
        network.add_node("t")
        result = edmonds_karp_max_flow(network, "s", "t")
        assert result["value"] == 0.0
        assert result["iterations"] == 0


class TestAgainstReferenceSolver:
    def test_matches_dfs_reference_on_random_networks(self):
        rng = random.Random(10)
        for _ in range(8):
            n = rng.randint(4, 9)
            network, source, sink = random_flow_network(n, rng.uniform(0.3, 0.6), rng=rng)
            # rebuild an identical copy for the reference solver so the two
            # algorithms never share mutated state
            reference_net = Network()
            for node in network.nodes:
                reference_net.add_node(node)
            for edge in network.edges[::2]:
                reference_net.add_edge(edge.u, edge.v, edge.capacity, edge.cost)

            ek_result = edmonds_karp_max_flow(network, source, sink)
            reference_value = reference_max_flow_dfs(reference_net, source, sink)

            assert abs(ek_result["value"] - reference_value) < 1e-6
            assert is_valid_flow(network, source, sink)


class TestTrace:
    def test_trace_has_one_entry_per_iteration_with_matching_bottleneck_sum(self):
        network = _diamond_network()
        result = edmonds_karp_max_flow(network, "s", "t")
        assert len(result["trace"]) == result["iterations"]
        assert abs(sum(entry["bottleneck"] for entry in result["trace"]) - result["value"]) < 1e-9
        for entry in result["trace"]:
            assert entry["path"][0][0] == "s"
            assert entry["path"][-1][1] == "t"
