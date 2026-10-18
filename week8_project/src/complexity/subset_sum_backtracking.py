"""
subset_sum_backtracking.py
----------------------------
Part 2 of week 8. Given a list of positive integers and a target, decide
whether some subset sums to exactly the target.

The exact solver is plain include/exclude backtracking over the list,
expected complexity O(2^n) since in the worst case it has to consider both
choices for every item before it can be sure no subset works. The heuristic
is a greedy approximation, sort descending then take anything that still
fits, expected complexity O(n log n), dominated by the sort, which trades
the exact guarantee away for speed and usually still lands close to target.
"""


def subset_sum_backtracking(numbers, target, collect_stats=False):
    """
    Exact backtracking solver. Returns (found, subset) where subset is a
    list of the chosen numbers if found is True, else (False, None). With
    collect_stats=True, returns a third element, {"nodes_visited": int},
    tracking how many include/exclude branches the search actually explored.
    """
    stats = {"nodes_visited": 0}
    chosen = []

    def backtrack(index, remaining):
        stats["nodes_visited"] += 1

        if remaining == 0:
            return True
        if remaining < 0 or index == len(numbers):
            return False

        # include numbers[index]
        chosen.append(numbers[index])
        if backtrack(index + 1, remaining - numbers[index]):
            return True
        chosen.pop()

        # exclude numbers[index]
        return backtrack(index + 1, remaining)

    found = backtrack(0, target)
    result = list(chosen) if found else None

    if collect_stats:
        return found, result, stats
    return found, result


def subset_sum_greedy_heuristic(numbers, target):
    """
    Fast approximate solver, O(n log n) from the sort. Sorts descending and
    greedily takes each number that still fits under the remaining target.
    This does not backtrack at all, so it can miss a solution that exists
    (a classic case is numbers=[5, 4, 3], target=7, greedy takes 5 first and
    then has no way to reach exactly 7 with what's left, even though 4 + 3
    works), which is exactly the accuracy it's trading away for speed.

    Returns (chosen, achieved_sum, exact), where exact is True only if
    achieved_sum lands exactly on target.
    """
    chosen = []
    remaining = target
    for num in sorted(numbers, reverse=True):
        if num <= remaining:
            chosen.append(num)
            remaining -= num

    achieved_sum = target - remaining
    return chosen, achieved_sum, remaining == 0


def is_valid_subset_sum(chosen, numbers, target):
    """Sanity check that `chosen` is actually a sub-multiset of `numbers`
    (respecting duplicate counts) summing to target exactly."""
    if sum(chosen) != target:
        return False
    remaining_pool = list(numbers)
    for item in chosen:
        if item not in remaining_pool:
            return False
        remaining_pool.remove(item)
    return True
