"""
conftest.py
------------
Shared pytest fixtures for Week 9's tests. Puts the project root on sys.path
(tests run as `pytest tests/` from inside week9_project/, same convention as
every other week in this course) and provides a couple of small seeded-RNG
fixtures so individual test files don't each have to set that up themselves.
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
