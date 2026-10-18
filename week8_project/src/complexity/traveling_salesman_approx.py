"""
traveling_salesman_approx.py
-------------------------------
Part 3 of week 8. The traveling salesman problem asks for the shortest
closed tour visiting every city exactly once, and like vertex cover it's np
hard in general, exact solving means checking every possible ordering.

tsp_nearest_neighbor is the heuristic, start somewhere, always walk to the
closest city not yet visited, repeat until all cities are visited, then
return home. Two nested loops, outer one over the n steps of the tour and
inner one scanning up to n remaining cities for the closest one, gives the
expected O(n^2).

tsp_exact_bruteforce is the exact solver, try every possible ordering of
the cities (fixing the first city, since a tour and its reverse have the
same length and the starting point doesn't matter for a closed tour) and
keep the shortest. That's (n-1)! orderings, which is only practical for
small n, its job this week is giving the heuristic something real to be
measured against.
"""

import itertools

from src.utils.graph_utils import euclidean_distance, tour_length


def tsp_nearest_neighbor(points, start=0):
    """
    Nearest neighbor heuristic. Returns a tour, a list of point indices in
    visiting order, built by always walking to whichever unvisited city is
    closest to the current one.
    """
    n = len(points)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    current = start

    for _ in range(n - 1):
        nearest = None
        nearest_distance = float("inf")
        for candidate in range(n):
            if visited[candidate]:
                continue
            distance = euclidean_distance(points[current], points[candidate])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = candidate

        visited[nearest] = True
        tour.append(nearest)
        current = nearest

    return tour


def tsp_exact_bruteforce(points):
    """
    Exact brute force solver. Tries every ordering of cities 1..n-1 with
    city 0 fixed as the start (valid since a closed tour's length doesn't
    depend on where you start or which direction you go around it), and
    returns the shortest tour found along with its length. Only practical
    for small n, the search space is (n-1)! orderings.
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


def approximation_ratio(heuristic_length, optimal_length):
    """How many times longer the heuristic's tour is than the true optimal
    tour. 1.0 means the heuristic happened to find the optimal tour."""
    if optimal_length == 0:
        return 1.0
    return heuristic_length / optimal_length
