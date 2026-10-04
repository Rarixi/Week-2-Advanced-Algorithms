"""
bitmask_traveling_salesman.py
-------------------------------
The Traveling Salesman Problem (TSP): given a complete graph of n cities and the
distance between every pair, find the cheapest possible round trip that visits every
city exactly once and returns to the start.

`dist` is an n x n distance matrix (dist[i][j] = distance from city i to city j;
dist[i][i] is unused). The graph is assumed complete (a direct edge between every pair
of cities) and symmetric is NOT required -- dist[i][j] need not equal dist[j][i].

Provides:
    tsp_brute_force(dist)   -- tries every permutation of the non-start cities,
                                O(n!) time. Only practical up to roughly n=10.
    tsp_bitmask_dp(dist)    -- Held-Karp dynamic programming, O(n^2 * 2^n) time,
                                O(n * 2^n) space. Practical up to roughly n=20, far
                                beyond what brute force can reach, but still
                                exponential -- TSP has no known polynomial-time exact
                                algorithm.

Both return (min_cost, tour), where `tour` is a list of city indices starting and
ending at city 0, e.g. [0, 2, 1, 3, 0] for 4 cities.

Why a BITMASK: the DP state needs to track "which subset of cities have been visited
so far" -- with n cities, there are 2^n possible subsets. A Python int used as a
bitmask (bit i set means "city i has been visited") represents any one of those
subsets in O(1) space and lets set-membership, union, and "remove one city" all be
done with O(1) bitwise operations (see src/utils/bitmask_utils.py) instead of an
actual set object, which is both faster and is what lets the subset itself be used
directly as a dict/array key.
"""

from itertools import permutations

from src.utils.bitmask_utils import full_mask, is_bit_set, clear_bit


def tsp_brute_force(dist):
    """Try every permutation of cities 1..n-1 (city 0 is fixed as the start/end, since
    a round trip's cost doesn't depend on which city it's labeled as "first"), compute
    each full tour's total cost, and keep the cheapest. O(n!) time, O(n) space (one
    permutation held at a time).
    """
    n = len(dist)
    if n <= 1:
        return 0, [0] if n == 1 else []

    best_cost = float("inf")
    best_tour = None

    for perm in permutations(range(1, n)):
        tour = [0] + list(perm) + [0]
        cost = sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_cost, best_tour


def tsp_bitmask_dp(dist):
    """Held-Karp dynamic programming. dp[(mask, i)] = the minimum cost to start at
    city 0, visit exactly the cities in `mask` (which always includes city 0 and
    city i), and end at city i. Built up by extending shorter paths one city at a
    time: dp[(mask, i)] is computed from dp[(mask without i, j)] for every city j
    already in mask -- "the best way to reach i having visited this exact set of
    cities is the best way to reach some other city j in that same set, then step
    from j to i."

    Once every dp[(full_mask, i)] is known (every city visited, ending at i), the
    answer is the cheapest of those plus the cost of returning from i back to city 0.

    Time: O(n^2 * 2^n) -- O(n * 2^n) distinct (mask, i) states, each considering up
    to n predecessor cities j. Space: O(n * 2^n) for the dp table (and a matching
    table to reconstruct the actual tour, not just its cost).
    """
    n = len(dist)
    if n <= 1:
        return 0, [0] if n == 1 else []

    all_visited = full_mask(n)
    # dp[mask][i]: min cost of a path starting at city 0, visiting exactly the cities
    # in `mask`, ending at city i. Only states where city 0 and city i are both in
    # `mask` are ever meaningful; everything else stays at infinity and is never read.
    dp = [[float("inf")] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    start_mask = 1  # just city 0 visited
    dp[start_mask][0] = 0

    for mask in range(1 << n):
        if not is_bit_set(mask, 0):
            continue  # every real state includes the start city
        for i in range(n):
            if not is_bit_set(mask, i) or dp[mask][i] == float("inf"):
                continue
            # Try extending this path by visiting one more, not-yet-visited city j.
            for j in range(n):
                if is_bit_set(mask, j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + dist[i][j]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
                    parent[new_mask][j] = i

    # Close the tour: from each city i (having visited everyone), return to city 0.
    best_cost = float("inf")
    best_last = -1
    for i in range(1, n):
        cost = dp[all_visited][i] + dist[i][0]
        if cost < best_cost:
            best_cost = cost
            best_last = i

    # Reconstruct the tour by walking `parent` backward from (all_visited, best_last).
    # parent[mask][i] = the city visited right before i, in the path that reaches
    # (mask, i) -- so stepping backward means recording i, then dropping i out of
    # mask to get the mask the PREDECESSOR state was computed under, and moving to
    # that predecessor city. This unwinds all the way back to city 0, whose own
    # parent entry was never set (stays -1), which is what ends the loop.
    path = []
    mask, i = all_visited, best_last
    while i != -1:
        path.append(i)
        prev_i = parent[mask][i]
        mask = clear_bit(mask, i)
        i = prev_i
    path.reverse()
    tour = path + [0]

    return best_cost, tour
