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
    floyd_warshall(matrix)              -- (distances, next_node), the full
                                            shortest-distance matrix and a routing table
                                            for path reconstruction
    reconstruct_path(next_node, u, v)   -- the actual shortest path from u to v, as a
                                            list of nodes, using the routing table above
    has_negative_cycle(distances)       -- True if the graph contains a negative-weight
                                            cycle (detectable from the diagonal of the
                                            finished distance matrix)

Time: O(V^3) -- three nested loops over every (k, i, j) triple. Space: O(V^2) for the
distance matrix (and another O(V^2) for the routing table).
"""


def floyd_warshall(matrix):
    """Compute all-pairs shortest distances. `matrix` is an n x n list of lists;
    matrix[i][j] is the direct edge weight i -> j, or float('inf') if none exists
    (matrix[i][i] is normally 0). Returns (distances, next_node):

        distances[i][j]  -- shortest distance from i to j (float('inf') if unreachable)
        next_node[i][j]  -- the next node to visit after i, on a shortest path to j
                            (None if i == j or j is unreachable from i) -- used by
                            reconstruct_path() below to recover the actual path, not
                            just its length.

    The core idea: for every intermediate node k, check whether routing i -> k -> j is
    shorter than the best i -> j distance found using only intermediate nodes
    0..k-1. After k has swept through every node 0..n-1, distances[i][j] is guaranteed
    to be the true shortest path using ANY of the n nodes as intermediate stops --
    this is the "optimal substructure" that makes it a DP: the best path through nodes
    0..k is built directly from the best paths through nodes 0..k-1.
    """
    n = len(matrix)
    distances = [row.copy() for row in matrix]
    next_node = [
        [j if (i != j and matrix[i][j] != float("inf")) else None for j in range(n)]
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
                    next_node[i][j] = next_node[i][k]

    return distances, next_node


def reconstruct_path(next_node, u, v):
    """Walk the routing table built by floyd_warshall() to recover the actual shortest
    path from u to v as a list of nodes, e.g. [0, 2, 3, 1]. Returns [] if there is no
    path from u to v (or if u == v with no self-loop, trivially [u])."""
    if next_node[u][v] is None:
        return [] if u != v else [u]

    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)
    return path


def has_negative_cycle(distances):
    """After floyd_warshall() finishes, a negative-weight cycle shows up as a negative
    value somewhere on the diagonal: distances[i][i] represents the shortest "path"
    from i back to itself, which should always be exactly 0 (the trivial empty path) --
    unless some cycle through i has negative total weight, in which case the DP keeps
    finding a cheaper and cheaper way to loop through it."""
    return any(distances[i][i] < 0 for i in range(len(distances)))
