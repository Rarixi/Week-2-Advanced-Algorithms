"""
test_benchmark_comparison.py
-------------------------------
Tests for week9_approx_benchmark.py. Doesn't re-check the algorithms
themselves (their own test files already do that), just checks that each
benchmark function returns well formed rows with the fields the
visualization and csv export code actually expects, and that export_csv
produces a readable file with every row's fields present.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.week9_approx_benchmark import (
    benchmark_vertex_cover,
    benchmark_facility_location,
    benchmark_set_cover,
    benchmark_tsp,
    benchmark_maxcut,
    export_csv,
)


class TestVertexCoverBenchmark:
    def test_ratio_rows_have_expected_fields(self):
        ratio_rows, _runtime_rows = benchmark_vertex_cover()
        assert ratio_rows
        for row in ratio_rows:
            assert row["algorithm"] == "vertex_cover"
            assert row["ratio"] >= 1.0 - 1e-9
            assert row["ratio"] <= 2.0 + 1e-9

    def test_runtime_rows_are_nonnegative_and_cover_every_size(self):
        _ratio_rows, runtime_rows = benchmark_vertex_cover()
        sizes_seen = {row["n"] for row in runtime_rows}
        assert sizes_seen
        for row in runtime_rows:
            assert row["time_us"] >= 0.0


class TestFacilityLocationBenchmark:
    def test_greedy_cost_never_beats_optimal(self):
        ratio_rows, _runtime_rows = benchmark_facility_location()
        assert ratio_rows
        for row in ratio_rows:
            assert row["greedy_cost"] >= row["optimal_cost"] - 1e-6
            assert row["ratio"] >= 1.0 - 1e-9


class TestSetCoverBenchmark:
    def test_both_variants_present_with_expected_fields(self):
        ratio_rows, _runtime_rows = benchmark_set_cover()
        variants = {row["variant"] for row in ratio_rows}
        assert variants == {"deterministic", "randomized"}

        det_rows = [r for r in ratio_rows if r["variant"] == "deterministic"]
        for row in det_rows:
            assert "ratio" in row
            assert row["ratio"] >= 1.0 - 1e-6

        rand_rows = [r for r in ratio_rows if r["variant"] == "randomized"]
        for row in rand_rows:
            assert "mean_ratio" in row
            assert "ratio_variance" in row
            assert row["ratio_variance"] >= 0.0


class TestTSPBenchmark:
    def test_ratio_rows_respect_the_two_approximation_bound(self):
        ratio_rows, _runtime_rows = benchmark_tsp()
        assert ratio_rows
        for row in ratio_rows:
            assert row["ratio"] >= 1.0 - 1e-9
            assert row["ratio"] <= 2.0 + 1e-9


class TestMaxCutBenchmark:
    def test_local_search_never_beats_its_own_random_start_backwards(self):
        # local_cut comes from running local search starting at the same
        # assignment as random_cut, so it should never be worse.
        ratio_rows, _runtime_rows = benchmark_maxcut()
        assert ratio_rows
        for row in ratio_rows:
            assert row["local_cut"] >= row["random_cut"]
            assert row["random_cut"] <= row["optimal_cut"]
            assert row["local_cut"] <= row["optimal_cut"]

    def test_runtime_rows_have_both_variants(self):
        _ratio_rows, runtime_rows = benchmark_maxcut()
        variants = {row["variant"] for row in runtime_rows}
        assert variants == {"random", "local_search"}


class TestExportCSV:
    def test_writes_a_readable_csv_with_every_field(self, tmp_path):
        rows = [
            {"algorithm": "vertex_cover", "ratio": 1.5, "n": 5},
            {"algorithm": "set_cover", "mean_ratio": 1.1, "n": 6, "variant": "randomized"},
        ]
        out_path = tmp_path / "test_table.csv"
        export_csv(rows, str(out_path))

        assert out_path.exists()
        with open(out_path) as f:
            read_rows = list(csv.DictReader(f))
        assert len(read_rows) == 2
        assert read_rows[0]["algorithm"] == "vertex_cover"
        assert read_rows[1]["variant"] == "randomized"
