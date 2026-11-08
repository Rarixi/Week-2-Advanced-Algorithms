"""
conftest.py
--------------
Shared pytest fixtures and reference solvers for the string search test
suite. The reference solvers here are deliberately naive, brute force
substring search, a full double loop over every candidate substring for
distinct-substring and longest-repeated-substring, so that KMP,
Rabin-Karp, the suffix array, and the suffix tree all get cross checked
against something simple enough to trust, not just against each other.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def rng():
    return random.Random(42)


def brute_force_positions(text, pattern):
    """Every start index of `pattern` in `text`, found the dumb way, a
    direct slice comparison at every possible position."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if text[i:i + m] == pattern]


def brute_distinct_substrings(text):
    """The set of every distinct substring of `text`, O(n^3) total
    characters, fine for the short strings the test suite uses this on."""
    substrings = set()
    n = len(text)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(text[i:j])
    return substrings


def brute_longest_repeated_substring(text):
    """The longest substring occurring at least twice, found by
    checking every length from longest to shortest and returning as
    soon as one length has a repeated substring."""
    n = len(text)
    for length in range(n, 0, -1):
        seen = set()
        for i in range(n - length + 1):
            sub = text[i:i + length]
            if sub in seen:
                return sub
            seen.add(sub)
    return ""
