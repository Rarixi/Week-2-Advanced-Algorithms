"""
randomized_selection.py
-------------------------
Part 2: Randomized Selection (QuickSelect) -- finds the kth smallest element in
expected O(n) by using the same random-pivot partition idea as randomized quicksort,
but only recursing into the ONE side that contains the answer instead of both sides.

k is 1-indexed here: k=1 asks for the minimum, k=n asks for the maximum, and
k=(n+1)//2 asks for the (lower) median, matching how the benchmark and report refer to
"min / median / max" queries.

A note on the "deterministic selection" comparison: there is no deterministic
selection algorithm anywhere else in this repo to compare against (Week 2 only has
sorting, not a dedicated select), so `deterministic_select` below is this file's own
baseline -- sort the array once with Python's built-in sort (Timsort, O(n log n)) and
index into it. That is the standard "deterministic" reference point for selection:
guaranteed O(n log n) with no randomness involved, against which QuickSelect's
expected O(n) is supposed to win once n gets large, at the cost of a worst case that
(rarely) degrades toward O(n^2).

Provides:
    randomized_select(arr, k)                  -- kth smallest, expected O(n)
    deterministic_select(arr, k)                -- kth smallest via sort, O(n log n)
    randomized_select_with_stats(arr, k)        -- (value, stats) -- stats has
                                                    recursion_depth, total_calls, and
                                                    good_split_count (a "good" split
                                                    puts the pivot's rank within the
                                                    middle half of the current
                                                    subarray, the standard definition
                                                    used to reason about QuickSelect's
                                                    expected linear running time)
"""

import random


def randomized_select(arr, k):
    """Returns the kth smallest element of arr (k is 1-indexed). Picks a random pivot
    at each step and recurses into only the side known to contain the kth element."""
    working = arr.copy()
    return _randomized_select(working, 0, len(working) - 1, k, stats=None)


def deterministic_select(arr, k):
    """Returns the kth smallest element of arr (k is 1-indexed) by sorting a copy and
    indexing into it. No randomness, guaranteed O(n log n), used as this file's
    deterministic baseline since none exists elsewhere in the repo."""
    return sorted(arr)[k - 1]


def randomized_select_with_stats(arr, k):
    """Same as randomized_select, but also returns a stats dict:
        recursion_depth   -- how many nested recursive calls were made
        total_calls       -- total number of partition steps performed
        good_split_count  -- how many of those steps picked a pivot whose rank, within
                              the CURRENT subarray being partitioned, fell in the
                              middle half (between the 25th and 75th percentile
                              position) -- the standard "good pivot" definition used
                              to argue QuickSelect's expected O(n) running time.
    """
    working = arr.copy()
    stats = {"recursion_depth": 0, "total_calls": 0, "good_split_count": 0}
    result = _randomized_select(working, 0, len(working) - 1, k, stats=stats, depth=1)
    return result, stats


def _randomized_select(arr, low, high, k, stats, depth=1):
    if stats is not None:
        stats["total_calls"] += 1
        stats["recursion_depth"] = max(stats["recursion_depth"], depth)

    if low == high:
        return arr[low]

    pivot_index = _partition_random(arr, low, high)

    if stats is not None:
        subarray_size = high - low + 1
        relative_rank = (pivot_index - low) / (subarray_size - 1) if subarray_size > 1 else 0.5
        if 0.25 <= relative_rank <= 0.75:
            stats["good_split_count"] += 1

    rank = pivot_index - low + 1  # pivot's 1-indexed rank WITHIN this subarray
    if k == rank:
        return arr[pivot_index]
    elif k < rank:
        return _randomized_select(arr, low, pivot_index - 1, k, stats, depth + 1)
    else:
        # k is measured from the start of the ORIGINAL array, so when recursing right
        # we have to shift it by how many elements we just excluded on the left.
        return _randomized_select(arr, pivot_index + 1, high, k - rank, stats, depth + 1)


def _partition_random(arr, low, high):
    """Same Lomuto partition scheme as randomized_quicksort.py, kept as its own copy
    here rather than imported, since QuickSelect only ever needs the partition step
    on its own (never the full recursive sort)."""
    pivot_choice = random.randint(low, high)
    arr[pivot_choice], arr[high] = arr[high], arr[pivot_choice]
    pivot = arr[high]

    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
