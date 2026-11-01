"""
edmonds_karp.py
------------------
Part 1 of the flow network project. The Edmonds-Karp polynomial time
max-flow algorithm, Ford-Fulkerson with BFS picking the shortest (fewest
edges) augmenting path each round rather than an arbitrary one, which is
the one tweak that bounds the iteration count by O(V * E) regardless of
how large the capacities are, instead of Ford-Fulkerson's capacity
dependent worst case.

Each round, BFS walks the residual graph (any edge, forward or backward,
with positive residual capacity counts as traversable) from source to
sink, recording the edge used to reach each node. If sink is reached, the
path is recovered by walking parent pointers backward, its bottleneck is
the smallest residual capacity along it, and that much flow gets pushed
along every edge on the path. BFS repeats on the updated residual graph
until sink is no longer reachable, at which point the current flow is
provably maximum (the final BFS frontier is exactly the source side of a
minimum cut).
"""

from collections import deque


def _bfs_find_path(network, source, sink):
    """One BFS over the residual graph. Returns the list of edge indices
    forming the shortest source -> sink path, or None if sink is
    unreachable (max-flow reached)."""
    parent_edge = {source: None}
    queue = deque([source])

    while queue:
        u = queue.popleft()
        if u == sink:
            break
        for edge_index in network.edges_from(u):
            v = network.edges[edge_index].v
            if v in parent_edge:
                continue
            if network.residual_capacity(edge_index) <= 1e-12:
                continue
            parent_edge[v] = edge_index
            queue.append(v)

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


def edmonds_karp_max_flow(network, source, sink):
    """
    Runs Edmonds-Karp on `network` from source to sink. Returns a dict
    with:
      value           the max-flow value
      flow_matrix     {(u, v): flow} for every edge carrying flow
      iterations      how many augmenting paths were used
      trace           one entry per round, {path, bottleneck}, path a
                      list of (u, v) node pairs and bottleneck the flow
                      pushed that round, exactly what
                      visualization/animate_flow.py needs to replay the
                      run and what the benchmark uses to report
                      augmentation counts.
    Does not mutate any flow already on `network`, call reset_flow()
    first if it was used by another algorithm.
    """
    trace = []
    max_flow_value = 0.0

    while True:
        path = _bfs_find_path(network, source, sink)
        if path is None:
            break

        bottleneck = min(network.residual_capacity(i) for i in path)
        for edge_index in path:
            network.push_flow(edge_index, bottleneck)

        node_path = [network.edges[path[0]].u] + [network.edges[i].v for i in path]
        trace.append({
            "path": list(zip(node_path, node_path[1:])),
            "bottleneck": bottleneck,
        })
        max_flow_value += bottleneck

    return {
        "value": max_flow_value,
        "flow_matrix": network.flow_matrix(),
        "iterations": len(trace),
        "trace": trace,
    }
