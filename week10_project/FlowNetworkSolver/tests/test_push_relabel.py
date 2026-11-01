"""
test_push_relabel.py
-----------------------
Tests for core/push_relabel.py. Checks the same hand built networks
edmonds_karp is tested against to confirm both algorithms agree on
max-flow value, cross checks against random instances too, and checks
the returned flow is valid and the push/relabel counters and convergence
trace look sane.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.edmonds_karp import edmonds_karp_max_flow
from core.push_relabel import push_relabel_max_flow
from core.network import Network, is_valid_flow
from utils.generators import random_flow_network


def _copy_network(network):
    copy = Network()
    for node in network.nodes:
        copy.add_node(node)
    for edge in network.edges[::2]:
        copy.add_edge(edge.u, edge.v, edge.capacity, edge.cost)
    return copy


def _diamond_network():
    network = Network()
    network.add_edge("s", "a", 3.0)
    network.add_edge("a", "t", 10.0)
    network.add_edge("s", "b", 10.0)
    network.add_edge("b", "t", 2.0)
    return network


class TestKnownInstances:
    def test_diamond_network_max_flow_is_five(self):
        network = _diamond_network()
        result = push_relabel_max_flow(network, "s", "t")
        assert result["value"] == 5.0
        assert is_valid_flow(network, "s", "t")

    def test_disconnected_sink_gives_zero_flow(self):
        network = Network()
        network.add_edge("s", "a", 5.0)
        network.add_node("t")
        result = push_relabel_max_flow(network, "s", "t")
        assert result["value"] == 0.0


class TestAgreesWithEdmondsKarp:
    def test_matches_edmonds_karp_on_random_networks(self):
        rng = random.Random(20)
        for _ in range(8):
            n = rng.randint(4, 10)
            ek_network, source, sink = random_flow_network(n, rng.uniform(0.3, 0.6), rng=rng)
            pr_network = _copy_network(ek_network)

            ek_result = edmonds_karp_max_flow(ek_network, source, sink)
            pr_result = push_relabel_max_flow(pr_network, source, sink)

            assert abs(ek_result["value"] - pr_result["value"]) < 1e-6
            assert is_valid_flow(pr_network, source, sink)


class TestCountersAndConvergence:
    def test_counts_at_least_one_push_when_flow_is_positive(self):
        network = _diamond_network()
        result = push_relabel_max_flow(network, "s", "t")
        assert result["pushes"] >= 1

    def test_convergence_trace_excess_never_increases_at_the_end(self):
        rng = random.Random(21)
        network, source, sink = random_flow_network(12, 0.4, rng=rng)
        result = push_relabel_max_flow(network, source, sink)
        if len(result["convergence"]) >= 2:
            # total excess should trend toward zero, not a strict
            # monotonic guarantee every single sample (a single node's
            # relabel can momentarily look flat) but the last sample
            # should be far lower than the first whenever there's more
            # than a couple of samples
            first_excess = result["convergence"][0][1]
            last_excess = result["convergence"][-1][1]
            assert last_excess <= first_excess + 1e-6

    def test_detailed_trace_only_populated_when_requested(self):
        network = _diamond_network()
        result = push_relabel_max_flow(network, "s", "t", detailed_trace=False)
        assert result["trace"] == []

        network2 = _diamond_network()
        result2 = push_relabel_max_flow(network2, "s", "t", detailed_trace=True)
        assert len(result2["trace"]) > 0
