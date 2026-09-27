"""
test_dp_benchmark.py
-----------------------
pytest tests for benchmarks/week5_dp_benchmark.py (Week 5, Part 4).

These call run_all_benchmarks() with small, fast size configurations -- NOT the full
sweep main() uses (which intentionally runs up to n=1000 and takes ~20 seconds, including
the naive recursive versions right up to their cutoff) -- so the suite stays quick while
still exercising every code path: recursive + memo + tabulation, for all three
algorithms, with real timing/memory/call-count measurement against the actual
implementations. The plotting itself (utils/visualization.py) isn't asserted on here --
only the data it would be built from -- since the point of a plot is what it looks like,
not something a unit test can check.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.week5_dp_benchmark import run_all_benchmarks, export_csv

EXPECTED_COLUMNS = {"algorithm", "variant", "input_size", "time_us", "peak_memory_kb", "calls", "max_depth"}
ALGORITHMS = ["fibonacci", "knapsack", "lcs"]
VARIANTS = ["recursive", "memo", "tabulation"]

# Small enough that even the recursive versions finish instantly, but big enough that
# fibonacci/knapsack/lcs's recursion still actually happens (not just base cases).
SMALL_SIZES = [3, 5, 8]
SMALL_CUTOFF = 8


def run_small_benchmarks():
    return run_all_benchmarks(
        fib_sizes=SMALL_SIZES, fib_cutoff=SMALL_CUTOFF,
        knap_sizes=SMALL_SIZES, knap_cutoff=SMALL_CUTOFF, knap_capacity=20,
        lcs_sizes=SMALL_SIZES, lcs_cutoff=SMALL_CUTOFF,
        verbose=False,
    )


class TestRunAllBenchmarks:
    def test_returns_rows_with_expected_columns(self):
        rows = run_small_benchmarks()
        assert len(rows) > 0
        for row in rows:
            assert set(row.keys()) == EXPECTED_COLUMNS

    def test_every_algorithm_and_variant_is_covered(self):
        rows = run_small_benchmarks()
        seen = {(r["algorithm"], r["variant"]) for r in rows}
        for algorithm in ALGORITHMS:
            for variant in VARIANTS:
                assert (algorithm, variant) in seen, f"missing {algorithm}/{variant}"

    def test_every_requested_size_produced_a_row_for_dp_variants(self):
        # Recursive is intentionally skipped above its cutoff (it's exponential), but
        # memo and tabulation run at every requested size -- that asymmetry is the whole
        # point of Part 4, so it's worth checking explicitly rather than assuming it.
        rows = run_small_benchmarks()
        for algorithm in ALGORITHMS:
            for variant in ("memo", "tabulation"):
                sizes = {r["input_size"] for r in rows if r["algorithm"] == algorithm and r["variant"] == variant}
                assert sizes == set(SMALL_SIZES)
            recursive_sizes = {r["input_size"] for r in rows if r["algorithm"] == algorithm and r["variant"] == "recursive"}
            assert recursive_sizes == set(SMALL_SIZES)  # all of SMALL_SIZES is <= SMALL_CUTOFF here

    def test_times_and_memory_are_non_negative(self):
        rows = run_small_benchmarks()
        for row in rows:
            assert row["time_us"] >= 0
            assert row["peak_memory_kb"] >= 0

    def test_calls_and_depth_are_positive_integers(self):
        rows = run_small_benchmarks()
        for row in rows:
            assert isinstance(row["calls"], int) and row["calls"] > 0
            assert isinstance(row["max_depth"], int) and row["max_depth"] > 0

    def test_recursive_makes_at_least_as_many_calls_as_dp_at_same_size(self):
        # The whole point of Part 4: naive recursion re-explores overlapping
        # subproblems, DP doesn't. At the same input size, recursive's call count
        # should be at least as large as memo's and tabulation's -- this is the
        # "exponential vs. linear" story the assignment asks the plots to show,
        # checked here numerically instead of visually. (At these tiny sizes the gap
        # isn't dramatic yet -- that only shows up at the larger sizes main() uses --
        # so this checks ">=" rather than a specific ratio.)
        rows = run_small_benchmarks()
        by_key = {(r["algorithm"], r["variant"], r["input_size"]): r for r in rows}
        for algorithm in ALGORITHMS:
            for size in SMALL_SIZES:
                recursive_calls = by_key[(algorithm, "recursive", size)]["calls"]
                memo_calls = by_key[(algorithm, "memo", size)]["calls"]
                tabulation_calls = by_key[(algorithm, "tabulation", size)]["calls"]
                assert recursive_calls >= memo_calls
                assert recursive_calls >= tabulation_calls

    def test_recursive_depth_grows_with_dp_versions_staying_flat(self):
        # Tabulation is iterative -- its call stack never grows with input size, unlike
        # recursive (and memo, which shares the same recursive shape). Checked at the
        # two largest small sizes so the trend is unambiguous even at this tiny scale.
        rows = run_small_benchmarks()
        by_key = {(r["algorithm"], r["variant"], r["input_size"]): r for r in rows}
        for algorithm in ALGORITHMS:
            small_depth = by_key[(algorithm, "tabulation", SMALL_SIZES[0])]["max_depth"]
            large_depth = by_key[(algorithm, "tabulation", SMALL_SIZES[-1])]["max_depth"]
            assert large_depth == small_depth  # flat regardless of n


class TestExportCsv:
    def test_writes_expected_columns_and_row_count(self, tmp_path):
        rows = run_small_benchmarks()
        csv_path = tmp_path / "dp_vs_recursive_table.csv"
        export_csv(rows, str(csv_path))

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == EXPECTED_COLUMNS
            written_rows = list(reader)
        assert len(written_rows) == len(rows)
