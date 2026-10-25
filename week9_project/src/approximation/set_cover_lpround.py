"""
set_cover_lpround.py
-----------------------
Part 2 of week 9. Set Cover as an integer program, minimize sum(cost_s *
x_s) subject to, for every element e, sum(x_s for every set s containing
e) >= 1, and x_s in {0, 1}. Relaxing x_s to the interval [0, 1] gives a
linear program solvable in polynomial time (via lp_solver.py), whose
optimal value lower bounds the true integer optimum, since the LP is
strictly less constrained. The two rounding techniques below turn that
fractional solution back into an actual (integer) set cover.

Randomized rounding, independently include each set s with probability
min(1, x_s* * ln(n)), scaled up by ln(n) so that, summed across all n
elements, the chance any single element gets left uncovered stays small.
The textbook analysis (Vazirani, Approximation Algorithms, ch. 5) shows
this gives expected cost O(ln n) * LP_OPT <= O(ln n) * OPT, the O(log n)
guarantee this part targets, and that every element ends up covered with
reasonably high probability, but not probability 1 on a single pass, so a
deterministic fix up step (adding the cheapest set covering anything still
missed) guarantees a valid cover every time without changing the expected
cost analysis by more than that fix up's own small expected contribution.

Deterministic rounding here uses the classic frequency based technique
instead, let f be the largest number of sets any single element belongs
to (its frequency), and round x_s to 1 whenever x_s* >= 1/f. This always
produces a valid cover (provable from the LP's own constraints) and gives
an f-approximation, not O(log n) in general, f can be much larger than ln
n when some element is covered by many overlapping sets, which is exactly
the tradeoff the report measures empirically, deterministic rounding is
simple and always feasible in one shot, randomized rounding targets the
better O(log n) bound but needs the fix up safety net and has run to run
variance.
"""

from src.utils.lp_solver import solve_lp_minimize


def solve_set_cover_lp_relaxation(universe, sets, costs):
    """
    Builds and solves Set Cover's LP relaxation. Returns (x, lp_cost),
    x a list of fractional "how much of this set to use" values in [0, 1]
    parallel to `sets`, and lp_cost the LP's optimal objective value, a
    valid lower bound on the true integer optimum.
    """
    elements = sorted(universe)
    num_sets = len(sets)

    # sum(x_s for s containing e) >= 1  -->  -sum(x_s for s containing e) <= -1
    A_ub = []
    b_ub = []
    for e in elements:
        row = [-1.0 if e in sets[s] else 0.0 for s in range(num_sets)]
        A_ub.append(row)
        b_ub.append(-1.0)

    bounds = [(0.0, 1.0) for _ in range(num_sets)]

    x, lp_cost = solve_lp_minimize(costs, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return x, lp_cost


def deterministic_rounding(universe, sets, lp_solution):
    """
    Frequency based deterministic rounding. Returns the set of chosen set
    indices. Always produces a valid cover in one pass, no randomness, no
    fix up needed, in exchange for an f-approximation (f = the largest
    number of sets any one element belongs to) rather than O(log n).
    """
    frequency = max(
        sum(1 for s in sets if e in s)
        for e in universe
    )
    threshold = 1.0 / frequency if frequency > 0 else 1.0
    return {s for s, x in enumerate(lp_solution) if x >= threshold - 1e-9}


def randomized_rounding(universe, sets, costs, lp_solution, rng, scale=1.0):
    """
    Randomized rounding, independently include each set s with probability
    min(1, x_s* * scale), then a deterministic fix up pass guaranteeing
    every element ends up covered. Returns the set of chosen set indices.

    scale defaults to 1.0, plain unscaled rounding, each set's inclusion
    probability is exactly its LP value, which keeps real randomness in
    the result (and therefore real run to run cost variance) whenever the
    LP relaxation itself is fractional, at the cost of needing the fix up
    step more often. Passing scale=ln(n) recovers the textbook boosted
    version (Vazirani, Approximation Algorithms, ch. 5) whose formal
    guarantee is expected cost O(ln n) * LP_OPT and every element covered
    with high probability on a single pass, but with ln(n) often pushing
    most nonzero LP values' probabilities straight to 1, which in practice
    makes the rounding nearly deterministic on the small instances this
    benchmark actually uses, exactly the sort of gap between asymptotic
    guarantee and small n behavior worth measuring rather than assuming.
    Calling this multiple times with different rng state produces a
    different (but always valid) cover each time, that run to run spread
    is what the benchmark's variance metric is measuring.
    """
    chosen = set()
    for s, x in enumerate(lp_solution):
        probability = min(1.0, x * scale)
        if rng.random() < probability:
            chosen.add(s)

    covered = set()
    for s in chosen:
        covered |= sets[s]
    uncovered = universe - covered

    for e in uncovered:
        candidates = [s for s in range(len(sets)) if e in sets[s]]
        cheapest = min(candidates, key=lambda s: costs[s])
        chosen.add(cheapest)
        covered |= sets[cheapest]

    return chosen


def exact_set_cover_bruteforce(universe, sets, costs, max_size=None):
    """
    Brute force exact solver. This is weighted set cover, so the cheapest
    valid cover is not necessarily the one using the fewest sets, a bigger
    subset of cheap sets can easily beat a smaller subset of expensive
    ones, which means every subset size has to actually be checked, not
    just stopped at the first size that manages to cover everything. Only
    practical for a handful of sets, used purely to measure the real
    approximation ratio of both rounding techniques on small instances.
    max_size caps the largest subset size tried, for when a caller already
    knows the optimum can't need more sets than that.
    """
    import itertools

    num_sets = len(sets)
    search_limit = max_size if max_size is not None else num_sets

    best_cost = float("inf")
    best_chosen = set()

    for size in range(1, search_limit + 1):
        for candidate in itertools.combinations(range(num_sets), size):
            cost = sum(costs[s] for s in candidate)
            if cost >= best_cost:
                continue  # can't possibly beat the best found so far
            covered = set()
            for s in candidate:
                covered |= sets[s]
            if covered >= universe:
                best_cost = cost
                best_chosen = set(candidate)

    return best_chosen, best_cost


def is_valid_cover(chosen, sets, universe):
    covered = set()
    for s in chosen:
        covered |= sets[s]
    return covered >= universe


def solution_cost(chosen, costs):
    return sum(costs[s] for s in chosen)
