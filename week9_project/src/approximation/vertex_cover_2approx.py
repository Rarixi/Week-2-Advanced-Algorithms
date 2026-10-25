"""
vertex_cover_2approx.py
--------------------------
Part 1 of week 9. The standard greedy edge-picking 2-approximation for
Vertex Cover, pick any uncovered edge, add both its endpoints to the
cover, remove every edge that's now covered, repeat.

Proof that cost <= 2 * OPT, every time the greedy step picks an edge (u, v)
and adds both endpoints, that edge was completely uncovered by the cover
built so far, which means no vertex cover, including the true optimal one,
can cover it using only one of u or v's "credit", any valid cover has to
include at least one of u or v. So every pair the greedy algorithm adds
corresponds to an edge the optimal solution also had to pay for with at
least one vertex. Since the edges greedy picks are disjoint by
construction (each one gets removed, along with every other edge touching
u or v, the moment it's picked), the optimal cover needs at least one
vertex per picked edge, meaning OPT >= number of edges picked. Greedy's
cover size is exactly 2 times the number of edges picked. So
greedy's cost = 2 * (edges picked) <= 2 * OPT. exact_minimum_vertex_cover
below exists specifically to demonstrate this bound empirically, not just
state it.
"""

import itertools

from src.utils.graph_utils import is_vertex_cover


def vertex_cover_greedy_approx(vertices, edges):
    """
    Greedy edge picking 2-approximation. Returns a set of vertices that
    covers every edge, guaranteed to be at most twice the size of the
    true minimum cover.
    """
    remaining = set(edges)
    cover = set()

    while remaining:
        u, v = next(iter(remaining))
        cover.add(u)
        cover.add(v)
        remaining = {e for e in remaining if u not in e and v not in e}

    return cover


def exact_minimum_vertex_cover(vertices, edges, max_size=None):
    """
    Brute force exact solver, used only to demonstrate the greedy
    algorithm's 2x bound on small instances, not as a practical
    alternative. Tries every subset size from 0 upward and returns the
    first (size, cover) that actually covers every edge, so the result is
    a true minimum vertex cover. max_size caps how large a cover it will
    even try.
    """
    if not edges:
        return 0, set()

    search_limit = max_size if max_size is not None else len(vertices)

    for size in range(search_limit + 1):
        for candidate in itertools.combinations(vertices, size):
            if is_vertex_cover(candidate, edges):
                return size, set(candidate)

    return len(vertices), set(vertices)
