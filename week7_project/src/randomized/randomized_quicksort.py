"""
randomized_quicksort.py
-------------------------
Part 1: Randomized QuickSort -- explores how picking a random pivot at every step
avoids the worst-case input patterns that sink a fixed-pivot QuickSort.

A note on the "deterministic QuickSort from Week 2" comparison this file's benchmark
makes: Week 2's actual quick_sort.py already picks a RANDOM pivot (random.randint, then
swapped to the end) plus an insertion-sort cutoff for small subarrays, so it is not a
true deterministic baseline -- there isn't one anywhere else in the repo either. To make
the "randomized avoids worst-case input" comparison this part is actually about, this
file implements its own `deterministic_quicksort`, a textbook Lomuto-partition
QuickSort that always pivots on the LAST element of each subarray, the classic
textbook choice that degrades to O(n^2) on already-sorted or reverse-sorted input.
`randomized_quicksort` uses the same Lomuto partition scheme, just with a pivot chosen
uniformly at random each call, so the two functions differ in exactly one thing: how
the pivot is picked. That is what isolates randomization's effect instead of mixing it
with some other implementation difference.

Provides:
    randomized_quicksort(arr)                      -- sorted copy, random pivot
    deterministic_quicksort(arr)                   -- sorted copy, last-element pivot
    randomized_quicksort_with_comparisons(arr)      -- (sorted copy, comparison count)
    deterministic_quicksort_with_comparisons(arr)   -- (sorted copy, comparison count)
"""

import random


def randomized_quicksort(arr):
    """Returns a sorted copy of arr. Picks a uniformly random pivot at every
    partition step, so no single input pattern (sorted, reverse-sorted, all equal,
    etc.) can reliably trigger the O(n^2) worst case -- the pivot choice that would
    cause it only happens with low, identically small probability regardless of what
    the input looks like."""
    result = arr.copy()
    _quicksort(result, 0, len(result) - 1, pivot_strategy="random", counter=None)
    return result


def deterministic_quicksort(arr):
    """Returns a sorted copy of arr. Always pivots on the last element of the current
    subarray -- fine on "generic" input, but already-sorted or reverse-sorted input
    makes every partition maximally unbalanced (one side empty, one side n-1), giving
    this version's true O(n^2) worst case."""
    result = arr.copy()
    _quicksort(result, 0, len(result) - 1, pivot_strategy="last", counter=None)
    return result


def randomized_quicksort_with_comparisons(arr):
    """Same as randomized_quicksort, but also returns the total number of element
    comparisons made, for checking against the theoretical ~2n ln(n) expectation."""
    result = arr.copy()
    counter = [0]
    _quicksort(result, 0, len(result) - 1, pivot_strategy="random", counter=counter)
    return result, counter[0]


def deterministic_quicksort_with_comparisons(arr):
    """Same as deterministic_quicksort, but also returns the total comparison count."""
    result = arr.copy()
    counter = [0]
    _quicksort(result, 0, len(result) - 1, pivot_strategy="last", counter=counter)
    return result, counter[0]


def _quicksort(arr, low, high, pivot_strategy, counter):
    """Shared QuickSort body for both pivot strategies. `counter`, when not None, is
    a one-element list used as a mutable box so nested calls can accumulate into the
    same comparison count.

    Always recurses into the SMALLER of the two partitions and loops (instead of
    recursing) into the larger one. This is just an implementation detail for keeping
    recursion depth at O(log n) even in the worst case -- without it, the
    deterministic variant's worst case (already-sorted input, maximally unbalanced
    partitions every time) would need recursion depth proportional to n itself, which
    blows past Python's default recursion limit well before n reaches the thousands.
    It does not change which pivot gets picked or how many comparisons get made, so
    the time/comparison measurements this file is actually about are unaffected."""
    while low < high:
        pivot_index = _partition(arr, low, high, pivot_strategy, counter)
        if pivot_index - low < high - pivot_index:
            _quicksort(arr, low, pivot_index - 1, pivot_strategy, counter)
            low = pivot_index + 1
        else:
            _quicksort(arr, pivot_index + 1, high, pivot_strategy, counter)
            high = pivot_index - 1


def _partition(arr, low, high, pivot_strategy, counter):
    """Lomuto partition scheme: moves the chosen pivot to the end, then scans left to
    right, swapping anything <= the pivot into the growing "small" region. Returns the
    pivot's final index. `pivot_strategy` is the only thing that differs between the
    randomized and deterministic variants."""
    if pivot_strategy == "random":
        pivot_choice = random.randint(low, high)
    else:
        pivot_choice = high

    arr[pivot_choice], arr[high] = arr[high], arr[pivot_choice]
    pivot = arr[high]

    i = low - 1
    for j in range(low, high):
        if counter is not None:
            counter[0] += 1
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
