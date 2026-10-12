"""
test_randomized_benchmark.py
-------------------------------
Smoke tests for benchmarks/week7_randomized_benchmark.py. These run each benchmark
function with small, fast parameters (not the full sweep main() uses) and check the
SHAPE and basic sanity of what comes back, not exact numbers -- timing numbers are
inherently noisy, so these tests check structural correctness and qualitative
relationships (e.g. "deterministic's worst case has way more comparisons than
randomized's"), the same things the report draws its conclusions from.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import week7_randomized_benchmark as bm


class TestQuicksortBenchmark:
    def test_returns_rows_with_expected_keys(self):
        rows = bm.run_quicksort_benchmarks(sizes=[50, 100], worst_case_sizes=[50], seed=1, verbose=False)
        assert len(rows) > 0
        for row in rows:
            assert {"variant", "n", "time_us", "comparisons", "input"} <= row.keys()

    def test_worst_case_deterministic_has_far_more_comparisons(self):
        rows = bm.run_quicksort_benchmarks(sizes=[50], worst_case_sizes=[500], seed=1, verbose=False)
        det_worst = next(r for r in rows if r["variant"] == "deterministic_worst_case" and r["n"] == 500)
        rand_worst = next(r for r in rows if r["variant"] == "randomized_worst_case" and r["n"] == 500)
        assert det_worst["comparisons"] > rand_worst["comparisons"] * 5


class TestSelectionBenchmark:
    def test_returns_rows_with_expected_keys(self):
        rows = bm.run_selection_benchmarks(sizes=[50], k_types=["min", "median", "max"], seed=2, verbose=False)
        assert len(rows) == 1 * 3 * 2  # 1 size * 3 k_types * 2 variants
        for row in rows:
            assert {"variant", "n", "k_type", "k", "time_us"} <= row.keys()

    def test_randomized_rows_have_stats_fields(self):
        rows = bm.run_selection_benchmarks(sizes=[50], k_types=["median"], seed=2, verbose=False)
        randomized_row = next(r for r in rows if r["variant"] == "randomized")
        assert "recursion_depth" in randomized_row
        assert "good_split_fraction" in randomized_row


class TestPrimalityBenchmark:
    def test_returns_fp_rate_and_speed_rows(self):
        rows = bm.run_primality_benchmarks(
            k_values=[5], pool_size=10, trials_per_composite=3,
            large_ns=[1_000_003], seed=3, verbose=False,
        )
        fp_rows = [r for r in rows if "false_positive_rate" in r]
        speed_rows = [r for r in rows if "time_us" in r]
        assert len(fp_rows) > 0
        assert len(speed_rows) > 0

    def test_carmichael_fermat_rate_is_at_least_general_pool_rate(self):
        # Fermat's known weak spot is Carmichael numbers specifically -- its false
        # positive rate there should never be LOWER than on a pool of generic
        # composites, at a small k where the effect is visible.
        rows = bm.run_primality_benchmarks(
            k_values=[5], pool_size=30, trials_per_composite=5,
            large_ns=[1_000_003], seed=3, verbose=False,
        )
        fermat_general = next(r for r in rows if r["test"] == "fermat" and r["pool"] == "general" and r["k"] == 5)
        fermat_carmichael = next(r for r in rows if r["test"] == "fermat" and r["pool"] == "carmichael" and r["k"] == 5)
        assert fermat_carmichael["false_positive_rate"] >= fermat_general["false_positive_rate"]


class TestBloomFilterBenchmark:
    def test_returns_measured_and_theoretical_rows(self):
        rows = bm.run_bloom_filter_benchmarks(
            expected_items=200, target_fp_rate=0.05, load_factor_checkpoints=[0.25, 1.0],
            fp_trials=500, seed=4, verbose=False,
        )
        measured = [r for r in rows if r["metric"] == "measured"]
        theoretical = [r for r in rows if r["metric"] == "theoretical"]
        assert len(measured) == 2
        assert len(theoretical) == 2

    def test_higher_load_factor_gives_higher_fp_rate(self):
        rows = bm.run_bloom_filter_benchmarks(
            expected_items=200, target_fp_rate=0.05, load_factor_checkpoints=[0.1, 2.0],
            fp_trials=2000, seed=4, verbose=False,
        )
        measured = sorted(
            [r for r in rows if r["metric"] == "measured"], key=lambda r: r["load_factor"]
        )
        assert measured[0]["false_positive_rate"] < measured[-1]["false_positive_rate"]


class TestMinhashBenchmark:
    def test_returns_one_error_row_per_k(self):
        # speed_set_sizes=[] skips the (slower) exact-vs-minhash speed comparison --
        # that part is covered by TestMinhashSpeedComparison below instead, so this
        # test stays fast and focused on just the error-vs-k rows.
        rows = bm.run_minhash_benchmarks(k_values=[16, 64], trials_per_k=5, speed_set_sizes=[], seed=5, verbose=False)
        error_rows = [r for r in rows if r["metric"] == "error"]
        assert len(error_rows) == 2
        assert {r["num_hashes"] for r in error_rows} == {16, 64}

    def test_larger_k_has_lower_or_equal_mean_error_on_average(self):
        # Not guaranteed on any single run (it's an average over random trials), but
        # with enough trials and a decent k gap this should hold comfortably.
        rows = bm.run_minhash_benchmarks(
            k_values=[16, 256], trials_per_k=40, speed_set_sizes=[], seed=5, verbose=False
        )
        small_k = next(r for r in rows if r["metric"] == "error" and r["num_hashes"] == 16)
        large_k = next(r for r in rows if r["metric"] == "error" and r["num_hashes"] == 256)
        assert large_k["mean_abs_error"] < small_k["mean_abs_error"]


class TestMinhashSpeedComparison:
    def test_returns_exact_and_minhash_rows_per_size(self):
        rows = bm.run_minhash_benchmarks(
            k_values=[16], trials_per_k=3, speed_set_sizes=[50, 200], speed_k=16,
            speed_repeats=2, seed=5, verbose=False,
        )
        speed_rows = [r for r in rows if r["metric"] == "speed"]
        assert len(speed_rows) == 4  # 2 sizes * 2 variants (exact, minhash)
        assert {r["variant"] for r in speed_rows} == {"exact", "minhash"}
        assert {r["set_size"] for r in speed_rows} == {50, 200}

    def test_exact_jaccard_is_faster_than_minhash_for_one_off_comparison(self):
        # MinHash does strictly more hashing work than one direct set intersection,
        # so for a single one-off comparison it should lose on raw speed -- its real
        # payoff is a fixed-size signature, not CPU time, see the module docstring.
        rows = bm.run_minhash_benchmarks(
            k_values=[16], trials_per_k=3, speed_set_sizes=[500], speed_k=32,
            speed_repeats=3, seed=5, verbose=False,
        )
        exact_row = next(r for r in rows if r["metric"] == "speed" and r["variant"] == "exact")
        minhash_row = next(r for r in rows if r["metric"] == "speed" and r["variant"] == "minhash")
        assert exact_row["time_us"] < minhash_row["time_us"]


class TestPrimalityErrorBounds:
    def test_fp_rate_rows_carry_theoretical_bound(self):
        rows = bm.run_primality_benchmarks(
            k_values=[5], pool_size=10, trials_per_composite=3, large_ns=[1_000_003], seed=3, verbose=False,
        )
        fp_rows = [r for r in rows if "false_positive_rate" in r]
        assert all("theoretical_bound" in r for r in fp_rows)
        miller_rabin_row = next(r for r in fp_rows if r["test"] == "miller_rabin" and r["pool"] == "general")
        assert miller_rabin_row["theoretical_bound"] == pytest.approx(0.25 ** 5)

    def test_speed_rows_carry_variance(self):
        rows = bm.run_primality_benchmarks(
            k_values=[5], pool_size=10, trials_per_composite=3, large_ns=[1_000_003], seed=3, verbose=False,
        )
        speed_rows = [r for r in rows if "time_us" in r]
        assert all("time_variance_us2" in r for r in speed_rows)


class TestBloomFilterMemoryComparison:
    def test_bloom_memory_stays_flat_while_set_memory_grows(self):
        rows = bm.run_bloom_filter_benchmarks(
            expected_items=200, target_fp_rate=0.05, load_factor_checkpoints=[0.1, 1.0, 2.0],
            fp_trials=500, seed=4, verbose=False,
        )
        bloom_rows = sorted((r for r in rows if r["metric"] == "bloom_memory"), key=lambda r: r["items_added"])
        set_rows = sorted((r for r in rows if r["metric"] == "set_memory"), key=lambda r: r["items_added"])
        assert len(bloom_rows) == 3
        assert len({r["memory_bytes"] for r in bloom_rows}) == 1  # bit array size never changes
        assert set_rows[0]["memory_bytes"] < set_rows[-1]["memory_bytes"]  # the set keeps growing
        assert bloom_rows[-1]["memory_bytes"] < set_rows[-1]["memory_bytes"]


class TestExportCsv:
    def test_writes_combined_csv_with_algorithm_column(self, tmp_path):
        rows_by_algorithm = {
            "quicksort": [{"variant": "randomized", "n": 10, "time_us": 1.0}],
            "bloom_filter": [{"metric": "measured", "load_factor": 0.5, "false_positive_rate": 0.01}],
        }
        out_path = tmp_path / "summary.csv"
        bm.export_csv(rows_by_algorithm, str(out_path))

        content = out_path.read_text()
        assert "algorithm" in content.splitlines()[0]
        assert "quicksort" in content
        assert "bloom_filter" in content
