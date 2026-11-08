"""
test_autocomplete.py
------------------------
Tests for applications/autocomplete.py: prefix queries only return
words that actually start with the prefix, top-k respects frequency
ordering (with an alphabetical tiebreak), repeated inserts accumulate
frequency rather than overwriting it, and both accepted bulk-load shapes
(dict, and a list of words or (word, weight) pairs) behave the same.
"""

import pytest

from applications.autocomplete import AutocompleteEngine, build_autocomplete_engine


class TestInsertAndPrefixQuery:
    def test_prefix_query_only_returns_matching_words(self):
        engine = build_autocomplete_engine(["cat", "car", "dog"])
        matches = {word for word, _ in engine.all_matches("ca")}
        assert matches == {"cat", "car"}

    def test_empty_prefix_returns_everything(self):
        engine = build_autocomplete_engine(["cat", "car", "dog"])
        matches = {word for word, _ in engine.all_matches("")}
        assert matches == {"cat", "car", "dog"}

    def test_prefix_with_no_matches_returns_empty_list(self):
        engine = build_autocomplete_engine(["cat", "car"])
        assert engine.all_matches("zzz") == []

    def test_rejects_empty_word(self):
        engine = AutocompleteEngine()
        with pytest.raises(ValueError):
            engine.add_query("")


class TestTopKCompletions:
    def test_ranked_by_frequency_descending(self):
        engine = build_autocomplete_engine({"cat": 5, "car": 10, "cart": 3})
        top = engine.complete("ca", k=3)
        assert top == [("car", 10), ("cat", 5), ("cart", 3)]

    def test_k_limits_result_count(self):
        engine = build_autocomplete_engine({"cat": 5, "car": 10, "cart": 3, "care": 7})
        assert len(engine.complete("ca", k=2)) == 2

    def test_ties_break_alphabetically(self):
        engine = build_autocomplete_engine({"bat": 4, "ant": 4})
        # same frequency, "ant" should sort before "bat"
        top = engine.complete("", k=2)
        assert top == [("ant", 4), ("bat", 4)]


class TestFrequencyWeighting:
    def test_repeated_inserts_accumulate_rather_than_overwrite(self):
        engine = AutocompleteEngine()
        engine.add_query("hello")
        engine.add_query("hello")
        engine.add_query("hello")
        assert engine.complete("hello", k=1) == [("hello", 3)]

    def test_explicit_weight_is_additive_with_later_inserts(self):
        engine = AutocompleteEngine()
        engine.add_query("hello", weight=10)
        engine.add_query("hello", weight=5)
        assert engine.complete("hello", k=1) == [("hello", 15)]


class TestBulkLoadShapes:
    def test_dict_shape(self):
        engine = build_autocomplete_engine({"cat": 3, "car": 1})
        assert engine.complete("ca", k=2) == [("cat", 3), ("car", 1)]

    def test_plain_word_list_defaults_to_weight_one(self):
        engine = build_autocomplete_engine(["cat", "cat", "car"])
        assert engine.complete("ca", k=2) == [("cat", 2), ("car", 1)]

    def test_word_weight_pair_list(self):
        engine = build_autocomplete_engine([("cat", 9), ("car", 1)])
        assert engine.complete("ca", k=2) == [("cat", 9), ("car", 1)]
