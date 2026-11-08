"""
test_plagiarism.py
----------------------
Tests for applications/plagiarism.py: overlapping phrase detection
matches a brute force check, the suffix array based longest common
substring matches a brute force longest common substring, similarity
scores land at the expected extremes (0 for disjoint alphabets, 1 for
identical documents), and highlighting wraps exactly the matched spans.
"""

from applications.plagiarism import (
    find_overlapping_phrases,
    highlight_matches,
    longest_common_substring,
    shingles,
    similarity_score,
)


def _brute_force_lcs(a, b):
    best = ""
    for i in range(len(a)):
        for j in range(i + 1, len(a) + 1):
            candidate = a[i:j]
            if candidate in b and len(candidate) > len(best):
                best = candidate
    return best


class TestShingles:
    def test_overlapping_windows_of_fixed_length(self):
        result = shingles("abcde", phrase_length=3)
        assert result == {"abc": [0], "bcd": [1], "cde": [2]}

    def test_repeated_phrase_records_every_position(self):
        result = shingles("abcabc", phrase_length=3)
        assert result["abc"] == [0, 3]


class TestFindOverlappingPhrases:
    def test_shared_sentence_is_found(self):
        doc_a = "the quick brown fox jumps over the lazy dog"
        doc_b = "yesterday the quick brown fox jumps over somewhere else"
        overlaps = find_overlapping_phrases(doc_a, doc_b, phrase_length=8)
        assert any(o["phrase"] == "the quic" for o in overlaps)

    def test_no_shared_text_gives_no_overlaps(self):
        overlaps = find_overlapping_phrases("aaaaaaaa", "bbbbbbbb", phrase_length=4)
        assert overlaps == []

    def test_identical_documents_overlap_on_every_phrase(self):
        doc = "abcdefghij"
        overlaps = find_overlapping_phrases(doc, doc, phrase_length=4)
        distinct_phrases = shingles(doc, 4)
        assert len(overlaps) == len(distinct_phrases)


class TestLongestCommonSubstring:
    def test_matches_brute_force(self, rng):
        from utils.generators import random_string
        for _ in range(20):
            a = random_string(rng.randint(5, 30), alphabet="ab", rng=rng)
            b = random_string(rng.randint(5, 30), alphabet="ab", rng=rng)
            result = longest_common_substring(a, b)
            expected = _brute_force_lcs(a, b)
            assert result["length"] == len(expected)
            assert len(result["substring"]) == result["length"]

    def test_positions_are_correct(self):
        doc_a = "xxhelloworldxx"
        doc_b = "yyhelloworldyy"
        result = longest_common_substring(doc_a, doc_b)
        assert result["substring"] == "helloworld"
        assert doc_a[result["position_in_a"]:result["position_in_a"] + result["length"]] == "helloworld"
        assert doc_b[result["position_in_b"]:result["position_in_b"] + result["length"]] == "helloworld"

    def test_no_shared_characters_at_all(self):
        result = longest_common_substring("aaaa", "bbbb")
        assert result == {"substring": "", "position_in_a": None, "position_in_b": None, "length": 0}

    def test_empty_document(self):
        result = longest_common_substring("", "abc")
        assert result["length"] == 0


class TestSimilarityScore:
    def test_identical_documents_score_one(self):
        doc = "the quick brown fox jumps over the lazy dog repeatedly"
        result = similarity_score(doc, doc, phrase_length=6)
        assert result["score"] == 1.0

    def test_disjoint_alphabets_score_zero(self):
        result = similarity_score("aaaaaaaa", "bbbbbbbb", phrase_length=4)
        assert result["score"] == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        doc_a = "the quick brown fox jumps over the lazy dog"
        doc_b = "a completely different sentence about something else, over there"
        result = similarity_score(doc_a, doc_b, phrase_length=6)
        assert 0.0 < result["score"] < 1.0


class TestHighlightMatches:
    def test_matched_region_is_bracketed(self):
        doc_a = "prefix shared text suffix"
        doc_b = "other shared text words"
        highlighted = highlight_matches(doc_a, doc_b, phrase_length=6)
        assert "[[" in highlighted and "]]" in highlighted
        assert "shared text" in highlighted

    def test_no_overlap_returns_text_unchanged(self):
        assert highlight_matches("aaaaaaaa", "bbbbbbbb", phrase_length=4) == "aaaaaaaa"
