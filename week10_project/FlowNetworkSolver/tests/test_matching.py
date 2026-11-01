"""
test_matching.py
-------------------
Tests for applications/matching.py. Checks a hand built instance with a
known maximum matching size, that the returned matching never reuses a
left or right node, and cross checks against a brute force reference
matcher on small random instances.
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from applications.matching import bipartite_matching
from utils.generators import random_bipartite_graph


def _brute_force_max_matching(left_nodes, right_nodes, allowed_pairs):
    """Tries every subset of allowed_pairs, from largest to smallest,
    and returns the size of the first one that forms a valid matching
    (no left or right node repeated). Only used on small test instances,
    the search space is 2^|allowed_pairs|."""
    for size in range(len(allowed_pairs), 0, -1):
        for combo in itertools.combinations(allowed_pairs, size):
            lefts = [l for l, _r in combo]
            rights = [r for _l, r in combo]
            if len(set(lefts)) == len(lefts) and len(set(rights)) == len(rights):
                return size
    return 0


class TestKnownInstance:
    def test_triangle_shaped_bipartite_graph(self):
        # L0 can go to R0 or R1, L1 only to R0, L2 only to R1
        # max matching size is 2 (e.g. L1-R0, L2-R1)
        left = ["L0", "L1", "L2"]
        right = ["R0", "R1"]
        allowed = [("L0", "R0"), ("L0", "R1"), ("L1", "R0"), ("L2", "R1")]

        matching = bipartite_matching(left, right, allowed)
        assert len(matching) == 2

    def test_no_allowed_pairs_gives_empty_matching(self):
        matching = bipartite_matching(["L0"], ["R0"], [])
        assert matching == []


class TestMatchingValidity:
    def test_no_left_or_right_node_used_twice(self):
        rng = random.Random(1)
        for _ in range(6):
            left, right, allowed = random_bipartite_graph(5, 5, 0.4, rng=rng)
            matching = bipartite_matching(left, right, allowed)
            lefts_used = [l for l, _r in matching]
            rights_used = [r for _l, r in matching]
            assert len(lefts_used) == len(set(lefts_used))
            assert len(rights_used) == len(set(rights_used))
            for pair in matching:
                assert pair in allowed


class TestAgainstBruteForce:
    def test_matches_brute_force_on_small_random_instances(self):
        rng = random.Random(2)
        for _ in range(5):
            left, right, allowed = random_bipartite_graph(4, 4, 0.5, rng=rng)
            matching = bipartite_matching(left, right, allowed)
            expected_size = _brute_force_max_matching(left, right, allowed)
            assert len(matching) == expected_size
