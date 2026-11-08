"""
test_search_engine.py
-------------------------
Tests for core/search_engine.py: all four methods agree with each other
and with brute force on the same text and pattern, preprocessing time is
0 before a method's first use and recorded after it, stats/recommend
return sensible structures, and find_multi matches per-pattern results
from find().
"""

from core.search_engine import StringSearchEngine
from utils.generators import random_string

from conftest import brute_force_positions


class TestAllMethodsAgree:
    def test_matches_brute_force_on_random_strings(self, rng):
        for _ in range(20):
            text = random_string(rng.randint(5, 100), alphabet="abc", rng=rng)
            engine = StringSearchEngine(text)
            pattern_len = rng.randint(1, min(5, len(text)))
            pattern = random_string(pattern_len, alphabet="abc", rng=rng)
            expected = brute_force_positions(text, pattern)
            for method in ("kmp", "rk", "sa", "tree"):
                result = engine.find(pattern, method=method)
                assert result["positions"] == expected, (method, text, pattern)
                assert result["count"] == len(expected)

    def test_rejects_unknown_method(self):
        engine = StringSearchEngine("abcabc")
        import pytest
        with pytest.raises(ValueError):
            engine.find("abc", method="not_a_real_method")


class TestPreprocessingBookkeeping:
    def test_sa_and_tree_preprocessing_only_happens_once(self):
        engine = StringSearchEngine("the quick brown fox jumps over the lazy dog")
        first = engine.find("fox", method="sa")
        second = engine.find("dog", method="sa")
        assert first["preprocessing_time"] > 0.0
        assert first["preprocessing_time"] == second["preprocessing_time"]

    def test_kmp_and_rk_never_report_preprocessing_time(self):
        engine = StringSearchEngine("the quick brown fox")
        for method in ("kmp", "rk"):
            result = engine.find("fox", method=method)
            assert result["preprocessing_time"] == 0.0

    def test_preprocessing_is_zero_before_first_use(self):
        engine = StringSearchEngine("the quick brown fox")
        stats = engine.stats()
        assert stats["sa"]["preprocessing_time"] == 0.0
        assert stats["tree"]["preprocessing_time"] == 0.0


class TestStatsAndRecommend:
    def test_stats_tracks_call_counts_and_match_totals(self):
        engine = StringSearchEngine("abcabcabc")
        engine.find("abc", method="kmp")
        engine.find("abc", method="kmp")
        stats = engine.stats()
        assert stats["kmp"]["calls"] == 2
        assert stats["kmp"]["total_matches"] == 6  # 3 matches per call, 2 calls

    def test_recommend_returns_nonempty_string_before_and_after_use(self):
        engine = StringSearchEngine("abcabcabc")
        assert isinstance(engine.recommend(), str) and len(engine.recommend()) > 0
        engine.find("abc", method="sa")
        engine.find("abc", method="sa")
        after = engine.recommend()
        assert "sa" in after


class TestFindMulti:
    def test_matches_individual_find_calls(self, rng):
        text = random_string(300, alphabet="abc", rng=rng)
        engine = StringSearchEngine(text)
        patterns = [random_string(rng.randint(1, 4), alphabet="abc", rng=rng) for _ in range(8)]
        multi_result = engine.find_multi(patterns)
        for pattern in patterns:
            single_result = engine.find(pattern, method="rk")
            assert sorted(multi_result[pattern]) == sorted(single_result["positions"])
