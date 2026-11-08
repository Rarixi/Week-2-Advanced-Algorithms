"""
test_kmp.py
--------------
Tests for core/kmp.py: known instances, the edge cases the assignment
calls out by name (pattern longer than text, repeated characters,
periodic strings), and cross-validation against a brute force reference
on random strings.
"""

from core.kmp import compute_lps, kmp_search, kmp_search_with_trace
from utils.generators import periodic_string, random_string, repeated_char_string

from conftest import brute_force_positions


class TestComputeLps:
    def test_no_repeated_prefix_suffix_is_all_zero(self):
        assert compute_lps("abcd") == [0, 0, 0, 0]

    def test_fully_periodic_pattern(self):
        assert compute_lps("aaaa") == [0, 1, 2, 3]

    def test_abab_pattern(self):
        assert compute_lps("abab") == [0, 0, 1, 2]

    def test_rejects_empty_pattern(self):
        import pytest
        with pytest.raises(ValueError):
            compute_lps("")


class TestKnownInstances:
    def test_simple_single_match(self):
        assert kmp_search("hello world", "world") == [6]

    def test_overlapping_matches_are_all_found(self):
        # "aaaa" contains "aa" starting at 0, 1, and 2, overlapping matches
        assert kmp_search("aaaa", "aa") == [0, 1, 2]

    def test_no_match_returns_empty_list(self):
        assert kmp_search("hello world", "xyz") == []

    def test_pattern_equals_text(self):
        assert kmp_search("abc", "abc") == [0]


class TestEdgeCases:
    def test_pattern_longer_than_text_returns_empty(self):
        assert kmp_search("ab", "abcdef") == []

    def test_repeated_characters(self):
        text = repeated_char_string(20, "a")
        pattern = repeated_char_string(5, "a")
        assert kmp_search(text, pattern) == list(range(16))

    def test_periodic_string(self):
        text = periodic_string(20, "abc")
        matches = kmp_search(text, "cab")
        expected = brute_force_positions(text, "cab")
        assert matches == expected

    def test_rejects_empty_pattern(self):
        import pytest
        with pytest.raises(ValueError):
            kmp_search("abc", "")


class TestAgainstBruteForce:
    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 80), alphabet="ab", rng=rng)
            pattern_len = rng.randint(1, min(5, len(text)))
            pattern = random_string(pattern_len, alphabet="ab", rng=rng)
            assert kmp_search(text, pattern) == brute_force_positions(text, pattern)


class TestTrace:
    def test_trace_reports_the_same_matches_as_plain_search(self):
        text = "abcabcabd"
        pattern = "abcabd"
        plain = kmp_search(text, pattern)
        traced = kmp_search_with_trace(text, pattern)
        assert traced["matches"] == plain

    def test_trace_contains_at_least_one_mismatch_fallback_on_a_partial_match(self):
        # "abcabd" against "abcabcabd" mismatches partway through after
        # matching "abcab", this is exactly the fallback KMP's LPS table
        # exists to make cheap instead of restarting from scratch
        traced = kmp_search_with_trace("abcabcabd", "abcabd")
        fallback_ops = [t for t in traced["trace"] if t["op"] == "mismatch_fallback"]
        assert len(fallback_ops) >= 1
        assert all(op["to_j"] < op["from_j"] for op in fallback_ops)
