"""
test_suffix_tree.py
-----------------------
Tests for core/suffix_tree.py: substring and prefix queries matched
against brute force, distinct substring counting matched against a
brute force set, and longest repeated substring cross checked against
both a brute force version and the suffix array's own LCP-based answer
(core/suffix_array.py), since two independent data structures over the
same text should always agree.
"""

from core.suffix_array import build_lcp_array, build_suffix_array, longest_repeated_substring_via_lcp
from core.suffix_tree import (
    build_suffix_trie,
    count_distinct_substrings,
    count_nodes,
    longest_repeated_substring,
    prefix_positions,
    substring_positions,
)
from utils.generators import periodic_string, random_string, repeated_char_string

from conftest import brute_distinct_substrings, brute_force_positions, brute_longest_repeated_substring


class TestSubstringPositions:
    def test_banana_known_positions(self):
        text = "banana"
        root = build_suffix_trie(text)
        assert substring_positions(text, root, "ana") == [1, 3]
        assert substring_positions(text, root, "na") == [2, 4]
        assert substring_positions(text, root, "z") == []

    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 60), alphabet="ab", rng=rng)
            root = build_suffix_trie(text)
            pattern_len = rng.randint(1, min(5, len(text)))
            pattern = random_string(pattern_len, alphabet="ab", rng=rng)
            assert substring_positions(text, root, pattern) == brute_force_positions(text, pattern)

    def test_pattern_longer_than_text(self):
        text = "ab"
        root = build_suffix_trie(text)
        assert substring_positions(text, root, "abcdef") == []

    def test_rejects_empty_pattern(self):
        import pytest
        root = build_suffix_trie("abc")
        with pytest.raises(ValueError):
            substring_positions("abc", root, "")


class TestPrefixPositions:
    def test_prefix_matching_is_consistent_with_substring_positions(self, rng):
        for _ in range(20):
            text = random_string(rng.randint(5, 40), alphabet="abc", rng=rng)
            root = build_suffix_trie(text)
            prefix = text[:rng.randint(1, 3)]
            assert prefix_positions(text, root, prefix) == substring_positions(text, root, prefix)


class TestDistinctSubstrings:
    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 40), alphabet="abc", rng=rng)
            root = build_suffix_trie(text)
            assert count_distinct_substrings(text, root) == len(brute_distinct_substrings(text))

    def test_repeated_character_string_has_n_distinct_substrings(self):
        text = repeated_char_string(10, "x")
        root = build_suffix_trie(text)
        # "x", "xx", "xxx", ..., up to the full length, n distinct substrings
        assert count_distinct_substrings(text, root) == 10


class TestLongestRepeatedSubstring:
    def test_banana(self):
        text = "banana"
        root = build_suffix_trie(text)
        assert longest_repeated_substring(text, root) == "ana"

    def test_matches_brute_force_length(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 50), alphabet="ab", rng=rng)
            root = build_suffix_trie(text)
            got = longest_repeated_substring(text, root)
            expected = brute_longest_repeated_substring(text)
            assert len(got) == len(expected)

    def test_agrees_with_suffix_array_lcp_version(self, rng):
        for _ in range(60):
            text = random_string(rng.randint(1, 50), alphabet="abc", rng=rng)
            root = build_suffix_trie(text)
            sa = build_suffix_array(text)
            lcp = build_lcp_array(text, sa)
            tree_answer = longest_repeated_substring(text, root)
            sa_answer = longest_repeated_substring_via_lcp(text, sa, lcp)
            assert len(tree_answer) == len(sa_answer)

    def test_no_repeats_returns_empty_string(self):
        text = "abcdef"
        root = build_suffix_trie(text)
        assert longest_repeated_substring(text, root) == ""

    def test_periodic_string_repeats_the_whole_period_many_times(self):
        text = periodic_string(21, "abc")  # "abcabcabc...abc", 7 repeats
        root = build_suffix_trie(text)
        # the longest repeat can't be the full string, it has to leave
        # room to actually repeat, 18 characters (6 repeats of "abc") is
        # long enough to occur starting at index 0 and again at index 3
        assert len(longest_repeated_substring(text, root)) >= 18


class TestNodeCount:
    def test_node_count_is_at_most_twice_the_number_of_suffixes(self):
        # a compressed trie's leaves equal the number of suffixes, and
        # internal branching nodes can't exceed that, so total nodes is
        # bounded by roughly 2n
        text = random_string(200, alphabet="abcd")
        root = build_suffix_trie(text)
        assert count_nodes(root) <= 2 * len(text) + 1
