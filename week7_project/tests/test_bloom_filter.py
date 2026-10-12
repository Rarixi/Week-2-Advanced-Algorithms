"""
test_bloom_filter.py
-----------------------
Correctness tests for Part 4: bloom_filter.py.
"""

import random

from src.randomized.bloom_filter import BloomFilter, optimal_parameters


class TestOptimalParameters:
    def test_returns_positive_m_and_k(self):
        m, k = optimal_parameters(1000, 0.01)
        assert m > 0
        assert k >= 1

    def test_lower_target_fp_rate_needs_more_bits(self):
        m_loose, _ = optimal_parameters(1000, 0.10)
        m_tight, _ = optimal_parameters(1000, 0.001)
        assert m_tight > m_loose

    def test_more_expected_items_needs_more_bits(self):
        m_small, _ = optimal_parameters(100, 0.01)
        m_large, _ = optimal_parameters(10000, 0.01)
        assert m_large > m_small


class TestBloomFilterCorrectness:
    def test_no_false_negatives(self):
        bf = BloomFilter(expected_items=500, target_fp_rate=0.01)
        items = [f"item-{i}" for i in range(500)]
        for item in items:
            bf.add(item)
        for item in items:
            assert bf.check(item), f"{item} was added but check() says absent"

    def test_check_before_add_mostly_false(self):
        bf = BloomFilter(expected_items=100, target_fp_rate=0.001)
        # With a tiny load factor and a strict target rate, an unrelated item
        # should (almost always) read as absent before anything has been added.
        assert bf.check("never-added") is False

    def test_add_increments_items_added(self):
        bf = BloomFilter(expected_items=100, target_fp_rate=0.01)
        assert bf.items_added == 0
        bf.add("x")
        bf.add("y")
        assert bf.items_added == 2

    def test_load_factor_increases_with_adds(self):
        bf = BloomFilter(expected_items=1000, target_fp_rate=0.01)
        before = bf.current_load_factor
        for i in range(100):
            bf.add(f"item-{i}")
        after = bf.current_load_factor
        assert after > before


class TestBloomFilterFalsePositiveRate:
    def test_measured_fp_rate_in_reasonable_range_of_target(self):
        rng = random.Random(21)
        target = 0.01
        bf = BloomFilter(expected_items=2000, target_fp_rate=target)
        for i in range(2000):
            bf.add(f"present-{i}")

        trials = [f"absent-{i}" for i in range(50000)]
        false_positives = sum(1 for t in trials if bf.check(t))
        observed = false_positives / len(trials)
        # Loose bound -- this is a probabilistic structure, not an exact one; just
        # checking it's in the right ballpark (within a factor of 3) of the target.
        assert observed < target * 3

    def test_higher_load_factor_means_higher_fp_rate(self):
        bf = BloomFilter(expected_items=200, target_fp_rate=0.05)
        rate_early = bf.estimated_false_positive_rate
        for i in range(400):  # intentionally overload past expected_items
            bf.add(f"item-{i}")
        rate_late = bf.estimated_false_positive_rate
        assert rate_late > rate_early
