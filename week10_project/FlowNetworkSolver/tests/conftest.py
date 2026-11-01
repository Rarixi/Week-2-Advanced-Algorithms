import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def rng():
    return random.Random(42)


def reference_max_flow_dfs(network, source, sink):
    """
    A deliberately simple, independent Ford-Fulkerson reference solver,
    DFS instead of Edmonds-Karp's BFS, used purely to cross check
    edmonds_karp_max_flow, push_relabel_max_flow, and
    min_cost_max_flow's reported max-flow values against something that
    doesn't share any code with them. Not meant to be fast or a project
    deliverable, just an independent second opinion for the test suite,
    the same role the brute force solvers played for last week's
    approximation algorithms.
    """

    def find_path():
        parent_edge = {source: None}
        stack = [source]
        while stack:
            u = stack.pop()
            if u == sink:
                break
            for edge_index in network.edges_from(u):
                v = network.edges[edge_index].v
                if v in parent_edge:
                    continue
                if network.residual_capacity(edge_index) <= 1e-12:
                    continue
                parent_edge[v] = edge_index
                stack.append(v)
        if sink not in parent_edge:
            return None
        path = []
        node = sink
        while node != source:
            edge_index = parent_edge[node]
            path.append(edge_index)
            node = network.edges[edge_index].u
        path.reverse()
        return path

    total = 0.0
    while True:
        path = find_path()
        if path is None:
            break
        bottleneck = min(network.residual_capacity(i) for i in path)
        for edge_index in path:
            network.push_flow(edge_index, bottleneck)
        total += bottleneck
    return total
