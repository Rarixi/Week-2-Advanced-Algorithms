"""
min_cost_flow.py
-------------------
Part 1 of the flow network project. Min-Cost Max-Flow via successive
shortest augmenting paths, find the cheapest source to sink path in the
residual graph, push as much flow as its bottleneck allows, repeat until
no augmenting path remains, at which point the flow is both maximum
(same termination condition as Edmonds-Karp) and minimum cost among all
max-flows (a classical result, as long as every augmenting path chosen
really is cheapest, the resulting flow never contains a negative cost
cycle in its residual graph, which is exactly the condition that
characterizes a min-cost flow).

The one complication is that the residual graph has negative cost edges
by construction, every backward edge's cost is the negation of its
forward twin's, so plain Dijkstra (which assumes non-negative weights)
can't be used directly, and re-running Bellman-Ford every round would
work but costs an extra factor of V per round for no reason. Johnson's
trick fixes this with node potentials, reduced_cost(u, v) = cost(u, v) +
potential[u] - potential[v]. The first round runs Bellman-Ford once
(there are no negative edges yet, every backward edge still has 0
capacity, so this is really just Dijkstra with extra bookkeeping) to
seed potential[v] with each node's true shortest distance from the
source. After that, updating potential[v] += dist[v] every round using
that round's Dijkstra distances keeps every residual edge's reduced cost
non-negative forever, a standard invariant of successive shortest paths
with potentials, so every later round can safely use plain Dijkstra
instead of Bellman-Ford.
"""

import heapq


def _bellman_ford_potentials(network, source):
    """Initial potentials, true shortest path cost from source to every
    node using current residual edges (round 1 only, before any
    negative-cost backward edges have positive residual capacity)."""
    potential = {node: float("inf") for node in network.nodes}
    potential[source] = 0.0

    for _ in range(len(network.nodes) - 1):
        updated = False
        for u in network.nodes:
            if potential[u] == float("inf"):
                continue
            for edge_index in network.edges_from(u):
                if network.residual_capacity(edge_index) <= 1e-12:
                    continue
                edge = network.edges[edge_index]
                candidate = potential[u] + edge.cost
                if candidate < potential[edge.v] - 1e-12:
                    potential[edge.v] = candidate
                    updated = True
        if not updated:
            break

    return potential


def _dijkstra_with_potentials(network, source, potential):
    """One round's shortest path search using reduced costs, which stay
    non-negative given correctly maintained potentials. Returns
    (dist, parent_edge), dist in reduced-cost terms (the caller updates
    potentials with it directly) and parent_edge the edge index used to
    reach each node, for path reconstruction."""
    dist = {node: float("inf") for node in network.nodes}
    dist[source] = 0.0
    parent_edge = {}
    visited = set()

    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for edge_index in network.edges_from(u):
            if network.residual_capacity(edge_index) <= 1e-12:
                continue
            edge = network.edges[edge_index]
            v = edge.v
            reduced_cost = edge.cost + potential[u] - potential[v]
            # Reduced cost should be >= 0 given a correctly maintained
            # potential, clamp away tiny float noise rather than let it
            # push a node's distance backward.
            reduced_cost = max(0.0, reduced_cost)
            candidate = d + reduced_cost
            if candidate < dist[v] - 1e-12:
                dist[v] = candidate
                parent_edge[v] = edge_index
                heapq.heappush(heap, (candidate, v))

    return dist, parent_edge


def min_cost_max_flow(network, source, sink):
    """
    Runs successive shortest augmenting paths with Johnson potentials to
    find the minimum cost way to send the maximum possible flow from
    source to sink. Returns a dict with:
      cost            total cost of the final flow
      value           the max-flow value (same quantity edmonds_karp and
                      push_relabel report)
      flow_matrix     {(u, v): flow} for every edge carrying flow
      iterations      how many augmenting paths were used
    Does not mutate any flow already on `network`, call reset_flow()
    first if it was used by another algorithm.
    """
    potential = _bellman_ford_potentials(network, source)
    total_cost = 0.0
    total_flow = 0.0
    iterations = 0

    while True:
        dist, parent_edge = _dijkstra_with_potentials(network, source, potential)
        if sink not in parent_edge and sink != source:
            break
        if dist[sink] == float("inf"):
            break

        for node in network.nodes:
            if dist[node] < float("inf"):
                potential[node] += dist[node]

        path = []
        node = sink
        while node != source:
            edge_index = parent_edge[node]
            path.append(edge_index)
            node = network.edges[edge_index].u
        path.reverse()

        bottleneck = min(network.residual_capacity(i) for i in path)
        path_cost = sum(network.edges[i].cost for i in path)

        for edge_index in path:
            network.push_flow(edge_index, bottleneck)

        total_cost += bottleneck * path_cost
        total_flow += bottleneck
        iterations += 1

    return {
        "cost": total_cost,
        "value": total_flow,
        "flow_matrix": network.flow_matrix(),
        "iterations": iterations,
    }
