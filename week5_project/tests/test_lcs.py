"""
test_lcs.py
-------------
pytest tests for src/dp/lcs.py (Week 5, Part 3): Longest Common Subsequence.

Expected interface (document this in lcs.py's own docstring when you write it):
    lcs_recursive(x: str, y: str) -> int
    lcs_memo(x: str, y: str) -> int
    lcs_tabulation(x: str, y: str) -> int
    lcs_reconstruct(x: str, y: str) -> str
        -- one actual longest common subsequence, not just its length. More than one
        valid LCS can exist for the same inputs (e.g. both "BCBA" and "BDAB" are valid
        length-4 LCSs of "ABCBDAB" and "BDCABA"), so these tests check that whatever
        string comes back really IS a common subsequence of both inputs and has the
        right length -- not that it matches one specific textbook answer.

lcs_recursive is O(2^(n+m)), so it's only exercised with short strings here -- the
10-1000 character performance sweep the assignment asks for belongs in
benchmarks/week5_dp_benchmark.py, and should skip the recursive version past the
smallest sizes, same as fibonacci_recursive/knapsack_recursive.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp.lcs import lcs_recursive, lcs_memo, lcs_tabulation, lcs_reconstruct


# (x, y, expected_length) -- classic worked examples plus edge cases, shared across
# every test class below. Lengths verified independently with a reference DP
# implementation, not hand-derived.
CASES = [
    ("ABCBDAB", "BDCABA", 4),
    ("AGGTAB", "GXTXAYB", 4),
    ("ABC", "ABC", 3),
    ("ABC", "DEF", 0),
    ("", "", 0),
    ("", "ABC", 0),
    ("A", "A", 1),
    ("A", "B", 0),
]


def is_subsequence(sub, s):
    """True if `sub` can be obtained from `s` by deleting characters without
    reordering -- the actual definition lcs_reconstruct's output must satisfy
    against BOTH input strings, not just "contains the same letters"."""
    it = iter(s)
    return all(ch in it for ch in sub)


class TestLcsRecursive:
    @pytest.mark.parametrize("x,y,expected", CASES)
    def test_matches_known_length(self, x, y, expected):
        assert lcs_recursive(x, y) == expected


class TestLcsMemo:
    @pytest.mark.parametrize("x,y,expected", CASES)
    def test_matches_known_length(self, x, y, expected):
        assert lcs_memo(x, y) == expected

    def test_handles_longer_strings_than_recursive_would_survive(self):
        # A 37-character string against itself would be far too slow for
        # lcs_recursive's O(2^(n+m)), but trivial for O(n*m) DP. Two identical
        # strings have an LCS equal to their own length.
        x = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
        assert lcs_memo(x, x) == len(x)


class TestLcsTabulation:
    @pytest.mark.parametrize("x,y,expected", CASES)
    def test_matches_known_length(self, x, y, expected):
        assert lcs_tabulation(x, y) == expected

    def test_handles_longer_strings_than_recursive_would_survive(self):
        x = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
        assert lcs_tabulation(x, x) == len(x)


class TestAllImplementationsAgree:
    @pytest.mark.parametrize("x,y,expected", CASES)
    def test_memo_matches_recursive(self, x, y, expected):
        assert lcs_memo(x, y) == lcs_recursive(x, y)

    @pytest.mark.parametrize("x,y,expected", CASES)
    def test_tabulation_matches_recursive(self, x, y, expected):
        assert lcs_tabulation(x, y) == lcs_recursive(x, y)


class TestLcsReconstruct:
    """The reconstructed string has to actually BE a valid common subsequence of
    both inputs, at the correct (optimal) length -- not match one specific expected
    string, since ties can have multiple correct answers."""

    @pytest.mark.parametrize("x,y,expected_length", CASES)
    def test_reconstruction_is_valid_and_optimal_length(self, x, y, expected_length):
        result = lcs_reconstruct(x, y)
        assert len(result) == expected_length
        assert is_subsequence(result, x)
        assert is_subsequence(result, y)

    def test_empty_inputs_give_empty_result(self):
        assert lcs_reconstruct("", "") == ""
