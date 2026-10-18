"""
vertex_cover_approx.py
------------------------
Part 3 of week 8. Vertex cover asks for the smallest set of vertices that
touches every edge in a graph, and it's np hard in general. This module has
two solvers.

vertex_cover_greedy_approx is the standard 2-approximation, pick any edge
that isn't covered yet, add both its endpoints to the cover, remove every
edge that's now covered, repeat. Each step throws away at least one
uncovered edge touched by a pair of vertices that any valid cover would
have needed at least one of anyway, which is the whole argument behind the
2x bound, expected complexity O(E).

exact_minimum_vertex_cover is brute force, checking subsets of increasing
size until one actually covers every edge. It's exponential and only meant
for small instances, its entire job this week is giving the approximation
algorithm something real to be compared against.
"""

import itertools

from src.utils.graph_utils import is_vertex_cover


def vertex_cover_greedy_approx(vertices, edges):
    """
    Greedy edge picking 2-approximation. Returns a set of vertices that
    covers every edge, guaranteed to be at most twice the size of the true
    minimum cover.
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
    Brute force exact solver. Tries every subset size from 0 upward and
    returns the first (vertices_used, cover) where cover actually covers
    every edge, so the result is a true minimum vertex cover. max_size caps
    how large a cover it will even try, mostly so a caller can bail out
    early if it already knows the approximate cover's size, since there's
    no point brute forcing past that.

    This is only practical for small graphs (worked fine up to the sizes
    used in this project's tests and benchmarks, a few dozen vertices at
    most), the search space is every subset of the vertex set.
    """
    if not edges:
        return 0, set()

    search_limit = max_size if max_size is not None else len(vertices)

    for size in range(search_limit + 1):
        for candidate in itertools.combinations(vertices, size):
            if is_vertex_cover(candidate, edges):
                return size, set(candidate)

    # Shouldn't be reachable, the full vertex set always covers every edge.
    return len(vertices), set(vertices)
