"""
test_sorting_comparison.py
---------------------------
Cross-algorithm agreement tests for all five Week 2 sorting algorithms
(Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort).

Each individual test_*.py file (test_basic_sorts.py, test_merge_sort.py,
test_quick_sort.py) already checks each algorithm in isolation against a
handful of hand-picked cases. What those files can't catch is a bug that
happens to affect two algorithms the same way, or a case none of them
happened to think of. This file instead checks a much simpler, much
stronger property: for ANY input, every algorithm here must agree with
each other and with Python's own built-in sorted() -- there's only one
correct sorted order for a given list, so if even one algorithm disagrees,
that's a real bug.

Note: bubble_sort/selection_sort/insertion_sort (src/sorting/basic_sorts.py)
sort their input list IN PLACE and also return it, while merge_sort and
quick_sort (src/sorting/merge_sort.py, src/sorting/quick_sort.py) return a
new sorted list without touching the original. Every test below hands each
algorithm its own fresh copy of the input, so one algorithm's in-place
mutation can never bleed into another's.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sorting.basic_sorts import bubble_sort, selection_sort, insertion_sort
from src.sorting.merge_sort import merge_sort
from src.sorting.quick_sort import quick_sort

ALGORITHMS = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "merge_sort": merge_sort,
    "quick_sort": quick_sort,
}


def run_all(data):
    """Run every algorithm on its own fresh copy of `data`, return
    {name: result} for all five."""
    return {name: fn(data.copy()) for name, fn in ALGORITHMS.items()}


class TestAllImplementationsAgree:
    def test_agree_on_sample_arrays(self, sample_arrays):
        for label, data in sample_arrays.items():
            expected = sorted(data)
            results = run_all(data)
            for name, result in results.items():
                assert result == expected, (
                    f"{name} disagreed with sorted() on the '{label}' case: "
                    f"got {result}, expected {expected}"
                )

    def test_agree_on_large_random_array(self, large_random_array):
        expected = sorted(large_random_array)
        results = run_all(large_random_array)
        for name, result in results.items():
            assert result == expected, f"{name} disagreed with sorted() on the large random array"

    def test_agree_across_many_random_trials(self):
        random.seed(1234)
        for _ in range(30):
            size = random.randint(0, 200)
            data = [random.randint(-500, 500) for _ in range(size)]
            expected = sorted(data)
            results = run_all(data)
            for name, result in results.items():
                assert result == expected, (
                    f"{name} disagreed with sorted() on a random trial of size {size}"
                )

    def test_agree_on_many_duplicate_values(self):
        # Few unique values, lots of repeats -- the case week2_report.md flags as the
        # one where Quick Sort's partitioning is most likely to show trouble.
        random.seed(7)
        data = [random.randint(0, 2) for _ in range(150)]
        expected = sorted(data)
        results = run_all(data)
        for name, result in results.items():
            assert result == expected, f"{name} disagreed with sorted() on the duplicate-heavy case"

    def test_no_algorithm_mutates_its_result_relative_to_input_elements(self, sample_arrays):
        # Every algorithm's output must contain exactly the same elements (including
        # duplicates) as the input -- nothing added, dropped, or silently coerced.
        for data in sample_arrays.values():
            results = run_all(data)
            for name, result in results.items():
                assert sorted(result) == sorted(data), f"{name} lost or added an element"


class TestOriginalListIsNotCorrupted:
    """merge_sort and quick_sort promise NOT to touch the caller's original list
    (see their own test files); this test doesn't re-check that promise here, but
    does confirm run_all()'s copy-per-algorithm pattern is doing its job -- i.e.
    that comparing five algorithms in the same test never lets one algorithm's
    in-place sort corrupt the input another algorithm is about to see."""

    def test_bubble_sort_in_place_mutation_does_not_affect_other_algorithms(self):
        original = [5, 3, 4, 1, 2]
        results = run_all(original)
        # bubble_sort/selection_sort/insertion_sort mutate their OWN copy in place,
        # never the shared `original` list itself, since run_all() passes .copy().
        assert original == [5, 3, 4, 1, 2]
        for result in results.values():
            assert result == [1, 2, 3, 4, 5]
