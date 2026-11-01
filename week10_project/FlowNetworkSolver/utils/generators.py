"""
generators.py
----------------
Part 4 of the flow network project. Instance generators for every
problem shape this project needs, random dense and sparse flow networks,
bipartite matching graphs, grid graphs for image segmentation and
scaling benchmarks, and configurable capacity (and optionally cost)
distributions.
"""

import random

from core.network import Network


def random_flow_network(num_nodes, density, capacity_range=(1.0, 20.0), cost_range=None, rng=None):
    """
    Random flow network on `num_nodes` nodes, labeled 0..num_nodes-1,
    node 0 as source and num_nodes-1 as sink by convention. Edges only
    go from a lower-numbered node to a higher-numbered one, so the
    network is acyclic and every node has a shot at sitting on some
    source to sink path, with `density` the probability any given (i, j)
    pair with i < j gets an edge. Capacities are drawn uniformly from
    capacity_range. If cost_range is given, each edge also gets a cost
    drawn uniformly from it, for Min-Cost Flow instances, otherwise
    every edge costs 0. A chain 0 -> 1 -> ... -> num_nodes - 1 is added
    first to guarantee the sink is always reachable, then extra random
    edges are layered on top up to the requested density.

    Returns (network, source, sink).
    """
    rng = rng or random.Random()
    network = Network()
    for n in range(num_nodes):
        network.add_node(n)

    def make_capacity():
        return rng.uniform(*capacity_range)

    def make_cost():
        return rng.uniform(*cost_range) if cost_range else 0.0

    existing = set()
    for i in range(num_nodes - 1):
        network.add_edge(i, i + 1, make_capacity(), make_cost())
        existing.add((i, i + 1))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in existing:
                continue
            if rng.random() < density:
                network.add_edge(i, j, make_capacity(), make_cost())
                existing.add((i, j))

    return network, 0, num_nodes - 1


def random_bipartite_graph(num_left, num_right, edge_probability, rng=None):
    """
    Random bipartite graph for the matching application. Returns
    (left_nodes, right_nodes, allowed_pairs), left_nodes and right_nodes
    lists of labels ("L0", "L1", ... and "R0", "R1", ...), allowed_pairs
    a list of (left, right) tuples, each possible pairing included
    independently with probability edge_probability.
    """
    rng = rng or random.Random()
    left_nodes = [f"L{i}" for i in range(num_left)]
    right_nodes = [f"R{i}" for i in range(num_right)]
    allowed_pairs = [
        (left, right)
        for left in left_nodes
        for right in right_nodes
        if rng.random() < edge_probability
    ]
    return left_nodes, right_nodes, allowed_pairs


def grid_graph(width, height, capacity_range=(1.0, 10.0), rng=None):
    """
    Grid graph of the given dimensions, nodes are (x, y) tuples, each
    connected to its right and down neighbor (an undirected grid modeled
    as two directed edges so flow can move either way) with a random
    capacity drawn from capacity_range. Used both as a generic sparse
    benchmark instance (very low density, degree at most 4) and as the
    structural basis image_seg.py builds its pixel graph the same way
    over.

    Returns the Network. Source and sink are left for the caller to
    pick, grid graphs don't have a natural single source/sink the way
    random_flow_network's layered construction does.
    """
    rng = rng or random.Random()
    network = Network()
    for y in range(height):
        for x in range(width):
            network.add_node((x, y))

    for y in range(height):
        for x in range(width):
            if x + 1 < width:
                cap = rng.uniform(*capacity_range)
                network.add_edge((x, y), (x + 1, y), cap)
                network.add_edge((x + 1, y), (x, y), cap)
            if y + 1 < height:
                cap = rng.uniform(*capacity_range)
                network.add_edge((x, y), (x, y + 1), cap)
                network.add_edge((x, y + 1), (x, y), cap)

    return network
