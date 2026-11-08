"""
test_suffix_array.py
------------------------
Tests for core/suffix_array.py: the suffix array itself matches a
brute force sort of every suffix, the LCP array matches a direct
character by character comparison of consecutive suffixes, substring
search over the array matches brute force, and the LCP-based longest
repeated substring matches an independent brute force version.
"""

from core.suffix_array import (
    build_lcp_array,
    build_suffix_array,
    longest_repeated_substring_via_lcp,
    sa_search,
)
from utils.generators import periodic_string, random_string, repeated_char_string

from conftest import brute_force_positions, brute_longest_repeated_substring


def _brute_suffix_array(text):
    return sorted(range(len(text)), key=lambda i: text[i:])


def _brute_lcp(text, sa):
    lcp = [0]
    for i in range(1, len(sa)):
        a, b = text[sa[i - 1]:], text[sa[i]:]
        common = 0
        while common < len(a) and common < len(b) and a[common] == b[common]:
            common += 1
        lcp.append(common)
    return lcp


class TestBuildSuffixArray:
    def test_empty_text(self):
        assert build_suffix_array("") == []

    def test_banana_matches_known_order(self):
        sa = build_suffix_array("banana")
        assert [("banana"[i:]) for i in sa] == ["a", "ana", "anana", "banana", "na", "nana"]

    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 60), alphabet="abc", rng=rng)
            assert build_suffix_array(text) == _brute_suffix_array(text)

    def test_periodic_and_repeated_char_edge_cases(self):
        for text in (periodic_string(40, "abc"), repeated_char_string(35, "x")):
            assert build_suffix_array(text) == _brute_suffix_array(text)


class TestLcpArray:
    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(40):
            text = random_string(rng.randint(1, 60), alphabet="abc", rng=rng)
            sa = build_suffix_array(text)
            assert build_lcp_array(text, sa) == _brute_lcp(text, sa)

    def test_empty_text(self):
        assert build_lcp_array("", []) == []


class TestSaSearch:
    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 80), alphabet="ab", rng=rng)
            sa = build_suffix_array(text)
            pattern_len = rng.randint(1, min(5, len(text)))
            pattern = random_string(pattern_len, alphabet="ab", rng=rng)
            assert sa_search(text, sa, pattern) == brute_force_positions(text, pattern)

    def test_pattern_not_present(self):
        text = "banana"
        sa = build_suffix_array(text)
        assert sa_search(text, sa, "xyz") == []

    def test_pattern_longer_than_text(self):
        text = "ab"
        sa = build_suffix_array(text)
        assert sa_search(text, sa, "abcdef") == []


class TestLongestRepeatedSubstring:
    def test_banana(self):
        text = "banana"
        sa = build_suffix_array(text)
        lcp = build_lcp_array(text, sa)
        assert longest_repeated_substring_via_lcp(text, sa, lcp) == "ana"

    def test_matches_brute_force_length_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 50), alphabet="ab", rng=rng)
            sa = build_suffix_array(text)
            lcp = build_lcp_array(text, sa)
            got = longest_repeated_substring_via_lcp(text, sa, lcp)
            expected = brute_longest_repeated_substring(text)
            assert len(got) == len(expected)

    def test_no_repeats_returns_empty_string(self):
        text = "abcdef"
        sa = build_suffix_array(text)
        lcp = build_lcp_array(text, sa)
        assert longest_repeated_substring_via_lcp(text, sa, lcp) == ""
