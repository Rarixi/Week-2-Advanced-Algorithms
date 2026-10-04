"""
test_benchmark_comparison.py
--------------------------------
pytest tests for benchmarks/week6_dp_advanced_benchmark.py itself: not the DP
algorithms' correctness (that's covered by the other four test files), but that the
benchmark harness runs cleanly end-to-end, produces well-formed rows, and that CSV
export and every save_*_figure function actually complete without error.

Uses a small, fast configuration throughout (run_all_benchmarks accepts overrides for
exactly this reason) so this file runs in well under a second rather than reproducing
the full benchmark sweep every test run.
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from week6_dp_advanced_benchmark import run_all_benchmarks, export_csv, benchmark_variant
from src.utils.visualization import (
    save_mcm_figure,
    save_knapsack_space_figure,
    save_floyd_warshall_figure,
    save_tsp_figure,
)

# A small, fast configuration -- mirrors the shape of the real one in
# week6_dp_advanced_benchmark.py, just at sizes that finish almost instantly.
SMALL_KWARGS = dict(
    mcm_sizes=[3, 5, 8], mcm_cutoff=8,
    knap_capacities=[5, 10, 20], knap_n_items=5,
    fw_sizes=[3, 5, 8],
    tsp_sizes=[3, 4, 5], tsp_cutoff=5,
    verbose=False,
)

REQUIRED_FIELDS = {"algorithm", "variant", "input_size", "time_us", "peak_memory_kb"}


class TestRunAllBenchmarks:
    def test_produces_rows_for_every_algorithm(self):
        rows = run_all_benchmarks(**SMALL_KWARGS)
        algorithms = {r["algorithm"] for r in rows}
        assert algorithms == {"mcm", "knapsack_space", "floyd_warshall", "tsp"}

    def test_every_row_has_the_required_fields(self):
        rows = run_all_benchmarks(**SMALL_KWARGS)
        assert rows  # sanity check the run actually produced something
        for row in rows:
            assert REQUIRED_FIELDS.issubset(row.keys())

    def test_times_and_memory_are_non_negative(self):
        rows = run_all_benchmarks(**SMALL_KWARGS)
        for row in rows:
            assert row["time_us"] >= 0
            assert row["peak_memory_kb"] >= 0

    def test_recursive_and_brute_force_variants_respect_their_cutoffs(self):
        rows = run_all_benchmarks(**SMALL_KWARGS)
        mcm_recursive_sizes = {r["input_size"] for r in rows if r["algorithm"] == "mcm" and r["variant"] == "recursive"}
        assert all(size <= SMALL_KWARGS["mcm_cutoff"] for size in mcm_recursive_sizes)

        tsp_brute_sizes = {r["input_size"] for r in rows if r["algorithm"] == "tsp" and r["variant"] == "brute_force"}
        assert all(size <= SMALL_KWARGS["tsp_cutoff"] for size in tsp_brute_sizes)

    def test_dp_variants_run_at_every_configured_size(self):
        # Unlike recursive/brute_force, the DP variants have no cutoff -- every
        # configured size should produce a row.
        rows = run_all_benchmarks(**SMALL_KWARGS)
        mcm_tab_sizes = {r["input_size"] for r in rows if r["algorithm"] == "mcm" and r["variant"] == "tabulation"}
        assert mcm_tab_sizes == set(SMALL_KWARGS["mcm_sizes"])

        tsp_dp_sizes = {r["input_size"] for r in rows if r["algorithm"] == "tsp" and r["variant"] == "bitmask_dp"}
        assert tsp_dp_sizes == set(SMALL_KWARGS["tsp_sizes"])


class TestBenchmarkVariant:
    def test_appends_exactly_one_row(self):
        rows = []
        benchmark_variant(lambda: sum(range(100)), "demo", "sum", 100, rows, verbose=False)
        assert len(rows) == 1
        assert rows[0]["algorithm"] == "demo"
        assert rows[0]["variant"] == "sum"
        assert rows[0]["input_size"] == 100


class TestExportCSV:
    def test_writes_a_readable_csv_with_every_row(self):
        rows = run_all_benchmarks(**SMALL_KWARGS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "comparison_table.csv")
            export_csv(rows, path)

            with open(path, newline="") as f:
                reread = list(csv.DictReader(f))

            assert len(reread) == len(rows)
            for row in reread:
                assert REQUIRED_FIELDS.issubset(row.keys())


class TestVisualizationFunctionsRunCleanly:
    """Each save_*_figure function should complete without error given the small
    benchmark configuration's rows, and actually produce a non-empty file -- this
    doesn't check the PICTURE is correct (that's a human judgment call made by
    looking at the real, full-size benchmark's output), just that the plotting code
    itself doesn't crash and does write something."""

    def test_all_four_figures_are_written(self):
        rows = run_all_benchmarks(**SMALL_KWARGS)
        with tempfile.TemporaryDirectory() as tmp:
            figures = [
                (save_mcm_figure, "mcm.png"),
                (save_knapsack_space_figure, "knapsack.png"),
                (save_floyd_warshall_figure, "fw.png"),
                (save_tsp_figure, "tsp.png"),
            ]
            for save_fn, filename in figures:
                path = os.path.join(tmp, filename)
                save_fn(rows, path)
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0
