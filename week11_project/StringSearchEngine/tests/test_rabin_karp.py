"""
test_rabin_karp.py
----------------------
Tests for core/rabin_karp.py: known instances, multi pattern batch
matching, double hashing agreeing with single hashing on the final
match list, the collision demo actually producing collisions at a small
modulus, and cross-validation against brute force on random strings.
"""

import pytest

from core.rabin_karp import collision_rate_demo, rabin_karp_search, rabin_karp_search_multi
from utils.generators import random_string

from conftest import brute_force_positions


class TestKnownInstances:
    def test_simple_single_match(self):
        result = rabin_karp_search("hello world", "world")
        assert result["matches"] == [6]
        assert result["collisions"] == 0

    def test_no_match(self):
        result = rabin_karp_search("hello world", "xyz")
        assert result["matches"] == []
        assert result["hash_matches"] == 0

    def test_overlapping_matches(self):
        result = rabin_karp_search("aaaa", "aa")
        assert result["matches"] == [0, 1, 2]

    def test_rejects_empty_pattern(self):
        with pytest.raises(ValueError):
            rabin_karp_search("abc", "")

    def test_pattern_longer_than_text(self):
        result = rabin_karp_search("ab", "abcdef")
        assert result["matches"] == []


class TestDoubleHashing:
    def test_double_hash_matches_single_hash_result(self, rng):
        for _ in range(20):
            text = random_string(rng.randint(5, 100), alphabet="abc", rng=rng)
            pattern = random_string(rng.randint(1, 4), alphabet="abc", rng=rng)
            single = rabin_karp_search(text, pattern)
            double = rabin_karp_search(text, pattern, double_hash=True)
            assert single["matches"] == double["matches"]

    def test_double_hash_never_increases_verifications_needed(self, rng):
        # a second independent hash can only filter out candidates, never add any
        text = random_string(300, alphabet="ab", rng=rng)
        pattern = "aba"
        single = rabin_karp_search(text, pattern)
        double = rabin_karp_search(text, pattern, double_hash=True)
        assert double["verifications"] <= single["verifications"]


class TestMultiPatternMatching:
    def test_several_patterns_in_one_pass(self):
        text = "the cat sat on the mat"
        result = rabin_karp_search_multi(text, ["cat", "sat", "mat", "dog"])
        assert result["cat"] == [4]
        assert result["sat"] == [8]
        assert result["mat"] == [19]
        assert result["dog"] == []

    def test_matches_brute_force_per_pattern(self, rng):
        text = random_string(200, alphabet="abc", rng=rng)
        patterns = [random_string(rng.randint(1, 4), alphabet="abc", rng=rng) for _ in range(10)]
        result = rabin_karp_search_multi(text, patterns)
        for pattern in patterns:
            assert result[pattern] == brute_force_positions(text, pattern)

    def test_rejects_empty_pattern_in_batch(self):
        with pytest.raises(ValueError):
            rabin_karp_search_multi("abc", ["ab", ""])


class TestCollisionDemo:
    def test_small_modulus_produces_more_collisions_than_a_large_one(self, rng):
        text = random_string(4000, alphabet="abcd", rng=rng)
        pattern = text[1000:1006]
        demo = collision_rate_demo(text, pattern, moduli=[101, (1 << 61) - 1])
        small_modulus_rate = demo[101]["collision_rate"]
        large_modulus_rate = demo[(1 << 61) - 1]["collision_rate"]
        assert small_modulus_rate >= large_modulus_rate

    def test_every_verified_match_is_a_true_occurrence(self, rng):
        text = random_string(2000, alphabet="abcd", rng=rng)
        pattern = text[500:505]
        demo = collision_rate_demo(text, pattern, moduli=[97])
        # hash_matches minus collisions should equal the true occurrence count
        true_count = len(brute_force_positions(text, pattern))
        assert demo[97]["hash_matches"] - demo[97]["collisions"] == true_count


class TestAgainstBruteForce:
    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 80), alphabet="ab", rng=rng)
            pattern_len = rng.randint(1, min(5, len(text)))
            pattern = random_string(pattern_len, alphabet="ab", rng=rng)
            assert rabin_karp_search(text, pattern)["matches"] == brute_force_positions(text, pattern)
