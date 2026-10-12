"""
conftest.py
------------
Shared pytest fixtures for Week 7's tests. Puts the project root on sys.path (tests
run as `pytest tests/` from inside week7_project/, same convention as every other
week in this course) and provides a couple of small seeded-RNG fixtures so individual
test files don't each have to set that up themselves.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def rng():
    """A fresh random.Random instance seeded for reproducibility."""
    import random
    return random.Random(42)


@pytest.fixture
def small_random_array():
    """A fixed, deliberately non-sorted small array of distinct integers, useful for
    tests that need to check correctness against a known sorted answer."""
    return [8, 3, 5, 1, 9, 2, 7, 0, 6, 4]
