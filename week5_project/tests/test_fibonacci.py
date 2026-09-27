"""
test_fibonacci.py
------------------
pytest tests for src/dp/fibonacci.py (Week 5, Part 1).

Expected interface (document this in fibonacci.py's own docstring when you write it):
    fibonacci_recursive(n: int) -> int   -- naive recursion, O(2^n)
    fibonacci_memo(n: int) -> int        -- top-down DP (memoization), O(n)
    fibonacci_tabulation(n: int) -> int  -- bottom-up DP (tabulation), O(n)
All three use fib(0)=0, fib(1)=1, fib(n)=fib(n-1)+fib(n-2), and must agree on every
input -- these tests check CORRECTNESS, not performance. The execution-time /
recursive-call-count comparison the assignment asks for belongs in
benchmarks/week5_dp_benchmark.py, not here: timing inside a unit test is flaky, and
n=45 under naive recursion would make the test suite unusably slow to even run.

IMPORTANT: fibonacci_recursive is only exercised up to n=20 below (2^20 is already
~1M calls). Never call fibonacci_recursive with the larger n values from the
assignment (30, 35, 40, 45) inside a test -- that blowup is exactly what the
benchmark script is supposed to demonstrate, not something a test suite should sit
through.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp.fibonacci import fibonacci_recursive, fibonacci_memo, fibonacci_tabulation


# The known correct sequence, index = n. Every test class checks against this same
# ground truth instead of re-deriving expected values three separate times.
KNOWN_FIBONACCI = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
    55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
]  # index 20 -> fib(20) = 6765


class TestFibonacciRecursive:
    """Naive recursive version -- kept to small n (<=20) so the suite stays fast."""

    @pytest.mark.parametrize("n,expected", list(enumerate(KNOWN_FIBONACCI)))
    def test_matches_known_values(self, n, expected):
        assert fibonacci_recursive(n) == expected

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError):
            fibonacci_recursive(-1)


class TestFibonacciMemo:
    """Top-down DP -- safe to test at larger n since it's O(n), not O(2^n)."""

    @pytest.mark.parametrize("n,expected", list(enumerate(KNOWN_FIBONACCI)))
    def test_matches_known_values(self, n, expected):
        assert fibonacci_memo(n) == expected

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError):
            fibonacci_memo(-1)

    def test_handles_large_n_quickly(self):
        # n=200 would take naive recursion vastly longer than is practical; memoization
        # should return instantly. This is the whole point of Part 1. (Value confirmed
        # independently via an iterative reference computation, not hand-typed.)
        assert fibonacci_memo(200) == 280571172992510140037611932413038677189525

    def test_repeated_calls_are_independent(self):
        # A memo cache implemented as a mutable default argument (a classic Python
        # footgun -- `def f(n, cache={})`) would leak state between separate calls.
        # Calling fibonacci_memo(10) twice in a row must give the same answer both
        # times, not "correct the first time, wrong after the cache is polluted."
        assert fibonacci_memo(10) == fibonacci_memo(10) == 55


class TestFibonacciTabulation:
    """Bottom-up DP -- iterative, so there's no recursion-depth concern at all."""

    @pytest.mark.parametrize("n,expected", list(enumerate(KNOWN_FIBONACCI)))
    def test_matches_known_values(self, n, expected):
        assert fibonacci_tabulation(n) == expected

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError):
            fibonacci_tabulation(-1)

    def test_handles_large_n_quickly(self):
        assert fibonacci_tabulation(200) == 280571172992510140037611932413038677189525


class TestAllImplementationsAgree:
    """The three versions are three ways of computing the SAME function -- they must
    never disagree. If memo or tabulation ever diverges from the naive recursive
    version, that's a bug in the DP formulation, not a stylistic difference."""

    @pytest.mark.parametrize("n", range(0, 21))
    def test_memo_matches_recursive(self, n):
        assert fibonacci_memo(n) == fibonacci_recursive(n)

    @pytest.mark.parametrize("n", range(0, 21))
    def test_tabulation_matches_recursive(self, n):
        assert fibonacci_tabulation(n) == fibonacci_recursive(n)

    @pytest.mark.parametrize("n", [30, 50, 100])
    def test_memo_matches_tabulation_beyond_recursive_range(self, n):
        # Beyond what's safe to check against naive recursion, memo and tabulation
        # must still agree with each other.
        assert fibonacci_memo(n) == fibonacci_tabulation(n)
