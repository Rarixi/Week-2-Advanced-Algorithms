"""
floyd_warshall.py
-------------------
Floyd-Warshall: all-pairs shortest paths. Given a weighted directed graph (edges can be
negative, but there must be no negative-weight CYCLE), find the shortest distance
between EVERY pair of nodes at once -- as opposed to Week 4's Dijkstra, which only finds
shortest paths from a single source.

The graph is given as an n x n distance matrix: matrix[i][j] is the direct edge weight
from node i to node j, or float('inf') if no direct edge exists (matrix[i][i] should be
0 -- the distance from a node to itself). This is exactly the "adjacency matrix"
representation from Week 4's graph.py, which is why Floyd-Warshall is naturally
matrix-based rather than needing Week 4's Graph class at all: the DP state IS the
matrix.

Provides:
    floyd_warshall(matrix)              -- (distances, predecessor), the full
                                            shortest-distance matrix and a predecessor
                                            matrix for path reconstruction
    reconstruct_path(predecessor, u, v) -- the actual shortest path from u to v, as a
                                            list of nodes, using the predecessor matrix
                                            above
    has_negative_cycle(distances)       -- True if the graph contains a negative-weight
                                            cycle (detectable from the diagonal of the
                                            finished distance matrix)
    dijkstra_single_source(matrix, source)
                                         -- Week 4's Dijkstra, adapted to this file's
                                            matrix representation, for a single source
    compare_with_dijkstra(matrix)       -- runs Floyd-Warshall once and Dijkstra from
                                            every node on the same small graph, checks
                                            they agree, and times both approaches

Time: O(V^3) -- three nested loops over every (k, i, j) triple. Space: O(V^2) for the
distance matrix (and another O(V^2) for the predecessor matrix).
"""

import heapq
import time


def floyd_warshall(matrix):
    """Compute all-pairs shortest distances. `matrix` is an n x n list of lists;
    matrix[i][j] is the direct edge weight i -> j, or float('inf') if none exists
    (matrix[i][i] is normally 0). Returns (distances, predecessor):

        distances[i][j]    -- shortest distance from i to j (float('inf') if
                               unreachable)
        predecessor[i][j]  -- the node visited right BEFORE j on the shortest path
                               from i to j (None if i == j or j is unreachable from
                               i) -- used by reconstruct_path() below to recover the
                               actual path, not just its length. This is the classic
                               predecessor matrix convention: to walk the path from i
                               to j, start at j and repeatedly jump to
                               predecessor[i][current] until i is reached, then
                               reverse the collected nodes.

    The core idea: for every intermediate node k, check whether routing i -> k -> j is
    shorter than the best i -> j distance found using only intermediate nodes
    0..k-1. After k has swept through every node 0..n-1, distances[i][j] is guaranteed
    to be the true shortest path using ANY of the n nodes as intermediate stops --
    this is the "optimal substructure" that makes it a DP: the best path through nodes
    0..k is built directly from the best paths through nodes 0..k-1.
    """
    n = len(matrix)
    distances = [row.copy() for row in matrix]
    predecessor = [
        [i if (i != j and matrix[i][j] != float("inf")) else None for j in range(n)]
        for i in range(n)
    ]

    for k in range(n):
        for i in range(n):
            if distances[i][k] == float("inf"):
                continue  # no path i -> k at all -- routing through k can't help
            for j in range(n):
                through_k = distances[i][k] + distances[k][j]
                if through_k < distances[i][j]:
                    distances[i][j] = through_k
                    # j's new predecessor, on the i -> k -> j path, is whatever k's
                    # own predecessor was on the i -> k path's LAST hop into j, i.e.
                    # the node that was already sitting right before j on the
                    # shortest k -> j path.
                    predecessor[i][j] = predecessor[k][j]

    return distances, predecessor


def reconstruct_path(predecessor, u, v):
    """Walk the predecessor matrix built by floyd_warshall() to recover the actual
    shortest path from u to v as a list of nodes, e.g. [0, 2, 3, 1]. Returns [] if
    there is no path from u to v (or if u == v with no self-loop, trivially [u]).

    Unlike a "next node" routing table, a predecessor matrix is walked BACKWARD: start
    at v, repeatedly jump to predecessor[u][current] until u is reached, then reverse
    the collected nodes to get the path in forward order.
    """
    if u == v:
        return [u]
    if predecessor[u][v] is None:
        return []

    path = [v]
    current = v
    while current != u:
        current = predecessor[u][current]
        path.append(current)
    path.reverse()
    return path


def has_negative_cycle(distances):
    """After floyd_warshall() finishes, a negative-weight cycle shows up as a negative
    value somewhere on the diagonal: distances[i][i] represents the shortest "path"
    from i back to itself, which should always be exactly 0 (the trivial empty path) --
    unless some cycle through i has negative total weight, in which case the DP keeps
    finding a cheaper and cheaper way to loop through it."""
    return any(distances[i][i] < 0 for i in range(len(distances)))


def dijkstra_single_source(matrix, source):
    """Dijkstra's algorithm, adapted from Week 4 to this file's n x n matrix
    representation (Week 4's version works directly on a Graph object's adjacency
    list instead). Returns a list `dist` of length n where dist[j] is the shortest
    distance from `source` to j (float('inf') if unreachable).

    Included here, rather than imported from week4_project, so this file (and its
    tests) are self contained -- the algorithm is identical to Week 4's, just reading
    edges out of a plain matrix instead of a Graph object. Dijkstra assumes no
    negative edge weights, same as Week 4.
    """
    n = len(matrix)
    dist = [float("inf")] * n
    dist[source] = 0
    visited = [False] * n
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(n):
            if u == v or matrix[u][v] == float("inf"):
                continue
            new_dist = d + matrix[u][v]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist


def compare_with_dijkstra(matrix):
    """Compare Floyd-Warshall against running Dijkstra from every single node, on the
    SAME small graph -- the natural way to get all-pairs shortest paths out of an
    algorithm that only solves one source at a time. Meant for small graphs (the
    "for small graphs" comparison from the assignment), not the large-scale benchmark
    sizes used elsewhere in this project, since repeating Dijkstra n times is its own
    separate O(n) multiplier on top of each run's own cost.

    Returns a dict with:
        "distances_agree"   -- True if every pair's distance matches between the two
                                approaches (the correctness check)
        "floyd_warshall_time" -- seconds for the single Floyd-Warshall call
        "dijkstra_total_time" -- seconds for running Dijkstra from every node, summed
    """
    n = len(matrix)

    start = time.perf_counter()
    fw_distances, _ = floyd_warshall(matrix)
    fw_time = time.perf_counter() - start

    start = time.perf_counter()
    dijkstra_distances = [dijkstra_single_source(matrix, source) for source in range(n)]
    dijkstra_time = time.perf_counter() - start

    agree = all(
        fw_distances[i][j] == dijkstra_distances[i][j]
        for i in range(n)
        for j in range(n)
    )

    return {
        "distances_agree": agree,
        "floyd_warshall_time": fw_time,
        "dijkstra_total_time": dijkstra_time,
    }
