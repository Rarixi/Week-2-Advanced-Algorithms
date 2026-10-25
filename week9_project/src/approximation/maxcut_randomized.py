"""
maxcut_randomized.py
-----------------------
Part 4 of week 9. Max-Cut asks for a partition of a graph's vertices into
two sides that maximizes the number of edges crossing between them, np
hard in general. The randomized approximation here is about as simple as
an algorithm gets, every vertex independently flips a fair coin to decide
which side it's on.

Proof that the expected cut value is >= OPT / 2, for any fixed edge (u, v),
u and v land on different sides with probability exactly 1/2 (their coin
flips are independent, and there are 4 equally likely outcomes, 2 of which
put them on opposite sides). By linearity of expectation, the expected
total number of crossing edges is just |E| * 1/2, regardless of the
graph's structure or which cut is actually optimal. Since no cut can ever
cross more edges than the graph has, OPT <= |E|, which means
E[cut] = |E| / 2 >= OPT / 2. That bound holds in expectation over the
random choice, not for every single random trial, which is exactly why
the benchmark measures variance across repeated trials rather than
trusting one run.

local_search_maxcut is included as the natural non-randomized comparison,
repeatedly flip whichever single vertex's flip would increase the cut,
until no single flip helps anymore. A local optimum under this rule
already guarantees cut >= |E| / 2 too, each vertex ends up with at least
as many neighbors on the other side as on its own, by definition of
"flipping it wouldn't help", summing that over all vertices and dividing
by 2 gives the same bound, so the real empirical question is how much
local search improves on a single random start, not whether it clears the
same floor.
"""

import itertools


def random_cut(vertices, edges, rng):
    """
    One random trial. Each vertex independently assigned side 0 or 1.
    Returns (assignment, cut_value).
    """
    assignment = {v: rng.randint(0, 1) for v in vertices}
    cut_value = sum(1 for u, v in edges if assignment[u] != assignment[v])
    return assignment, cut_value


def expected_cut_value(num_edges):
    """The theoretical expectation, |E| / 2, for comparison against
    measured results, not itself a random quantity."""
    return num_edges / 2.0


def local_search_maxcut(vertices, edges, assignment, adjacency):
    """
    Starting from `assignment` (typically a random_cut result), repeatedly
    flips whichever vertex's flip increases the cut value most, until no
    single flip helps, a local optimum. Returns (assignment, cut_value) for
    the improved partition, never worse than what it started from.
    """
    assignment = dict(assignment)
    improved = True

    while improved:
        improved = False
        for v in vertices:
            same_side = sum(1 for u in adjacency[v] if assignment[u] == assignment[v])
            other_side = sum(1 for u in adjacency[v] if assignment[u] != assignment[v])
            if same_side > other_side:
                assignment[v] = 1 - assignment[v]
                improved = True

    cut_value = sum(1 for u, v in edges if assignment[u] != assignment[v])
    return assignment, cut_value


def exact_maxcut_bruteforce(vertices, edges):
    """
    Brute force exact solver, tries every way to assign vertices to two
    sides (fixing the first vertex's side to cut the search space in half,
    since swapping both sides' labels doesn't change the cut) and returns
    the best cut value found. Only practical for a small number of
    vertices, the search space is 2^(n-1) assignments.
    """
    n = len(vertices)
    if n <= 1:
        return 0

    best = 0
    free_vertices = vertices[1:]
    for bits in itertools.product([0, 1], repeat=len(free_vertices)):
        assignment = {vertices[0]: 0}
        assignment.update(zip(free_vertices, bits))
        cut_value = sum(1 for u, v in edges if assignment[u] != assignment[v])
        best = max(best, cut_value)

    return best
