"""
tsp_metric_approx.py
-----------------------
Part 3 of week 9. A polynomial time 2-approximation for Metric TSP via MST
doubling, the simpler student-level variant the assignment spec explicitly
allows in place of Christofides' algorithm. Christofides needs a minimum
weight perfect matching on the MST's odd-degree vertices to get its
tighter 3/2 bound, a substantially more involved piece of machinery, MST
doubling gets a clean, provably correct 2-approximation with nothing more
exotic than a minimum spanning tree and an Eulerian circuit.

The algorithm, build a minimum spanning tree over the points, duplicate
every one of its edges (so every vertex's degree doubles, meaning every
vertex now has even degree), find an Eulerian circuit on that doubled
multigraph (one always exists on a connected graph where every vertex has
even degree), then shortcut the circuit by skipping any vertex it's
already visited, which turns the Eulerian walk into a genuine Hamiltonian
tour.

Why this is a 2-approximation, the optimal tour, with any single edge
removed, is itself a spanning tree, so MST weight <= OPT. The doubled
multigraph's total weight is exactly 2 * MST weight, and an Eulerian
circuit visits every edge exactly once, so it has that same total length.
Shortcutting only skips vertices, it never detours, and because the
instance is metric (distances satisfy the triangle inequality), skipping
a vertex can only shorten or keep equal the distance between consecutive
stops, never lengthen it. So the final tour's length is at most the
Eulerian circuit's length, which is exactly 2 * MST weight <= 2 * OPT.
"""

import itertools

from src.utils.graph_utils import euclidean_distance, tour_length


def build_mst_edges(points):
    """
    Prim's algorithm over the implicit complete graph on `points` (metric
    distance as edge weight). Returns a list of (u, v) edges forming a
    minimum spanning tree.
    """
    n = len(points)
    if n <= 1:
        return []

    in_tree = [False] * n
    in_tree[0] = True
    tree_edges = []

    # best_distance[v] / best_source[v]: cheapest known edge connecting v
    # to the tree so far, updated as new vertices join.
    best_distance = [euclidean_distance(points[0], points[v]) for v in range(n)]
    best_source = [0] * n

    for _ in range(n - 1):
        next_vertex = None
        next_distance = float("inf")
        for v in range(n):
            if not in_tree[v] and best_distance[v] < next_distance:
                next_distance = best_distance[v]
                next_vertex = v

        in_tree[next_vertex] = True
        tree_edges.append((best_source[next_vertex], next_vertex))

        for v in range(n):
            if not in_tree[v]:
                d = euclidean_distance(points[next_vertex], points[v])
                if d < best_distance[v]:
                    best_distance[v] = d
                    best_source[v] = next_vertex

    return tree_edges


def _eulerian_circuit(doubled_adjacency, start=0):
    """
    Hierholzer's algorithm on a multigraph where every vertex has even
    degree (true here since every mst edge was duplicated), given as
    {vertex: [neighbor, neighbor, ...]} with each edge appearing once per
    direction per copy. Returns the circuit as a list of vertices.
    Consumes (empties) the adjacency lists it's given.
    """
    stack = [start]
    circuit = []

    while stack:
        v = stack[-1]
        if doubled_adjacency[v]:
            u = doubled_adjacency[v].pop()
            doubled_adjacency[u].remove(v)
            stack.append(u)
        else:
            circuit.append(stack.pop())

    return circuit[::-1]


def tsp_mst_doubling_approx(points):
    """
    MST doubling 2-approximation. Returns a tour, a list of point indices
    in visiting order, built as described in the module docstring.
    """
    n = len(points)
    if n <= 2:
        return list(range(n))

    mst_edges = build_mst_edges(points)

    adjacency = {v: [] for v in range(n)}
    for u, v in mst_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        adjacency[u].append(v)
        adjacency[v].append(u)

    circuit = _eulerian_circuit(adjacency, start=0)

    visited = set()
    tour = []
    for v in circuit:
        if v not in visited:
            visited.add(v)
            tour.append(v)

    return tour


def tsp_exact_bruteforce(points):
    """
    Exact brute force solver, tries every ordering of cities 1..n-1 with
    city 0 fixed as the start (a closed tour's length doesn't depend on
    where you start or which direction you go around it), returns the
    shortest tour and its length. Only practical for small n, the search
    space is (n-1)! orderings.
    """
    n = len(points)
    if n <= 1:
        return list(range(n)), 0.0

    best_tour = None
    best_length = float("inf")

    for perm in itertools.permutations(range(1, n)):
        candidate = [0] + list(perm)
        length = tour_length(candidate, points)
        if length < best_length:
            best_length = length
            best_tour = candidate

    return best_tour, best_length


def approximation_ratio(approx_length, optimal_length):
    if optimal_length == 0:
        return 1.0
    return approx_length / optimal_length
