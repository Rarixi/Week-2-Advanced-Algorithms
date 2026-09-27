"""
graph_generator.py
-------------------
Builds random test graphs (src/graphs/graph.py) for benchmarking, in four flavors:

    generate_sparse_graph(n)    -- about 2 outgoing edges per node on average (E ~ 2V),
                                    close to a forest of trees. Most of the O(V^2)
                                    possible edges are simply absent.
    generate_dense_graph(n)     -- a fixed FRACTION of all possible directed edges exist
                                    (E ~ density * V*(V-1), default density 0.5).
    generate_random_graph(n)    -- textbook Erdos-Renyi G(n, p): every one of the n*(n-1)
                                    possible directed edges is independently included with
                                    probability p, so (unlike the two above) the exact edge
                                    count isn't fixed in advance -- it's a genuine random
                                    variable, which is the actual definition of a "random
                                    graph" in the graph-theory sense.
    generate_weighted_graph(n)  -- moderately sparse (like generate_sparse_graph), but with
                                    a much WIDER weight range (1-1000 by default instead of
                                    1-20). The point isn't topology here, it's giving
                                    Dijkstra's weight comparisons something more interesting
                                    to chew on than a narrow, nearly-uniform range of costs.

All four return a Graph built with representation="list" by default (the O(1) neighbor
lookup keeps benchmark timing focused on the ALGORITHM being tested rather than on
get_neighbors()'s own cost -- see bfs.py/dfs.py/dijkstra.py's docstrings for why the matrix
representation would muddy that). All four also assign random positive integer weights by
default (weighted=True), since Dijkstra needs them; pass weighted=False for a plain
unweighted graph (every edge gets weight 1).

Every function takes an `rng` argument (a random.Random instance, or the `random` module
itself). Call random.seed(...) once in your own script for reproducible benchmarks, or pass
your own random.Random(seed) instance if you need independent, non-interfering random
streams for different graphs in the same run.
"""

import random

from src.graphs.graph import Graph


def _new_graph(n, directed, representation):
    """Shared setup: an empty Graph with nodes 0..n-1 already added."""
    g = Graph(directed=directed, representation=representation)
    for i in range(n):
        g.add_node(i)
    return g


def _random_weight(rng, weighted, min_weight, max_weight):
    return rng.randint(min_weight, max_weight) if weighted else 1


def _add_random_edges(g, n, edge_count, rng, weighted, min_weight, max_weight):
    """Add exactly `edge_count` random edges with distinct, non-self-loop endpoints (capped
    at n*(n-1), the maximum a directed graph on n nodes can hold).

    Tracks already-picked (u, v) pairs in `seen` and re-rolls on a repeat, rather than just
    re-calling add_edge() on a duplicate pick. That distinction matters more than it might
    look: at low density, collisions are rare and barely affect the count either way -- but
    at high density (say, half of all possible edges), the birthday paradox makes repeat
    picks common, and simply letting add_edge() silently overwrite duplicates was
    UNDER-DELIVERING the requested density substantially (e.g. asking for 50% density could
    actually produce closer to 39%). For an assignment whose whole point is comparing
    behavior ACROSS densities, a generator that quietly hands back the wrong density
    defeats the purpose -- so this rejects duplicates outright to guarantee the exact count
    requested.

    This does mean generating a very high density (density approaching 1.0) gets slower,
    since correct rejection sampling means most late picks collide with an edge that
    already exists -- acceptable for the density range this project actually benchmarks
    (up to 0.5), but worth knowing if you push density higher.
    """
    max_possible_edges = n * (n - 1)
    edge_count = min(edge_count, max_possible_edges)

    seen = set()
    while len(seen) < edge_count:
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v or (u, v) in seen:
            continue
        seen.add((u, v))
        g.add_edge(u, v, weight=_random_weight(rng, weighted, min_weight, max_weight))
    return g


def generate_sparse_graph(
    n, edges_per_node=2, directed=True, weighted=True,
    min_weight=1, max_weight=20, representation="list", rng=random,
):
    """A sparse graph: about `edges_per_node` outgoing edges per node on average
    (E ~ edges_per_node * V). Close to a forest of trees -- almost none of the O(V^2)
    possible edges exist. This is the regime where BFS/DFS/Dijkstra's O(V+E) / O(E log V)
    guarantees look closest to O(V): edges barely add to the cost at all."""
    g = _new_graph(n, directed, representation)
    return _add_random_edges(g, n, n * edges_per_node, rng, weighted, min_weight, max_weight)


def generate_dense_graph(
    n, density=0.5, directed=True, weighted=True,
    min_weight=1, max_weight=20, representation="list", rng=random,
):
    """A dense graph: roughly `density` (default 0.5, i.e. half) of all n*(n-1) possible
    directed edges are present. This is the regime where the E term in O(V+E) / O(E log V)
    actually dominates, and where a matrix representation's O(V) get_neighbors() stops
    looking so wasteful compared to a list's O(degree) (see the traversal/Dijkstra
    benchmarks from Parts 2-3 for that comparison directly)."""
    edge_count = int(density * n * (n - 1))
    g = _new_graph(n, directed, representation)
    return _add_random_edges(g, n, edge_count, rng, weighted, min_weight, max_weight)


def generate_random_graph(
    n, edge_probability=0.05, directed=True, weighted=True,
    min_weight=1, max_weight=20, representation="list", rng=random,
):
    """Erdos-Renyi G(n, p): every single one of the n*(n-1) possible directed edges is
    independently included with probability `edge_probability`. Unlike
    generate_sparse_graph/generate_dense_graph, which both target a specific edge COUNT,
    this is the textbook definition of "a random graph" -- the actual edge count is a
    random variable itself (its expected value is edge_probability * n * (n-1), but any
    individual run can land a bit above or below that)."""
    g = _new_graph(n, directed, representation)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() < edge_probability:
                g.add_edge(u, v, weight=_random_weight(rng, weighted, min_weight, max_weight))
    return g


def generate_weighted_graph(
    n, edges_per_node=3, directed=True,
    min_weight=1, max_weight=1000, representation="list", rng=random,
):
    """A moderately sparse graph (like generate_sparse_graph, just a little denser by
    default) whose real point is its WIDE weight range -- 1 to 1000 by default, versus the
    narrow 1-20 range the other generators use. Topology isn't the interesting variable
    here; giving Dijkstra a wide spread of edge costs is, since a narrow, nearly-uniform
    weight range can make "shortest by total weight" and "shortest by hop count" agree so
    often that a Dijkstra bug (e.g. one that accidentally behaves like plain BFS) could
    hide undetected."""
    g = _new_graph(n, directed, representation)
    return _add_random_edges(
        g, n, n * edges_per_node, rng, weighted=True, min_weight=min_weight, max_weight=max_weight
    )
