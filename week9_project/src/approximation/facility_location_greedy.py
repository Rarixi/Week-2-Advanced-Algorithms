"""
facility_location_greedy.py
------------------------------
Part 1 of week 9. The Uncapacitated Facility Location problem, choose
which facilities to open (each with its own opening cost) so that every
client, connected to whichever open facility is cheapest to reach for it,
minimizes total opening cost plus total connection cost.

The greedy heuristic here opens whichever currently-closed facility would
most reduce total cost next (opening cost plus the resulting drop in
clients' connection costs), repeating until opening another facility
would no longer help. This is a natural, simple greedy, not the more
intricate 1.61-approximation from the Shmoys-Tardos LP rounding result,
polynomial time, and good in practice, which is why
exact_facility_location_bruteforce exists, to actually measure how close
this heuristic lands to optimal on small instances rather than just
asserting it's close.
"""

import itertools


def _total_cost(open_facilities, facility_costs, distances):
    """Opening cost of everything in open_facilities, plus every client's
    cheapest connection cost to the nearest open facility. Infinite if
    open_facilities is empty, since no client has anywhere to connect."""
    if not open_facilities:
        return float("inf")

    opening = sum(facility_costs[f] for f in open_facilities)
    connection = sum(
        min(distances[client][f] for f in open_facilities)
        for client in range(len(distances))
    )
    return opening + connection


def facility_location_greedy(facility_costs, distances):
    """
    Greedy heuristic. facility_costs[f] is the cost of opening facility f,
    distances[c][f] is client c's connection cost to facility f. Returns
    (open_facilities, total_cost).
    """
    num_facilities = len(facility_costs)
    open_facilities = set()
    current_cost = float("inf")

    while True:
        best_facility = None
        best_cost = current_cost

        for f in range(num_facilities):
            if f in open_facilities:
                continue
            candidate_cost = _total_cost(open_facilities | {f}, facility_costs, distances)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_facility = f

        if best_facility is None:
            break

        open_facilities.add(best_facility)
        current_cost = best_cost

    return open_facilities, current_cost


def exact_facility_location_bruteforce(facility_costs, distances, max_facilities=None):
    """
    Brute force exact solver, tries every non-empty subset of facilities
    and returns the cheapest. Only practical for a small number of
    facilities, used to measure the greedy heuristic's real gap from
    optimal rather than just citing a bound. max_facilities caps the
    largest subset size it will even try, mostly to bound runtime when a
    caller already knows a smaller cover exists.
    """
    num_facilities = len(facility_costs)
    search_limit = max_facilities if max_facilities is not None else num_facilities

    best_cost = float("inf")
    best_open = set()

    for size in range(1, search_limit + 1):
        for candidate in itertools.combinations(range(num_facilities), size):
            cost = _total_cost(set(candidate), facility_costs, distances)
            if cost < best_cost:
                best_cost = cost
                best_open = set(candidate)

    return best_open, best_cost
