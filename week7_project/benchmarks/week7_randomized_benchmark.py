"""
week7_randomized_benchmark.py
--------------------------------
Benchmarks and visualizes the five Week 7 algorithms:
    - Randomized vs. deterministic QuickSort (time, comparisons, worst-case input)
    - Randomized vs. deterministic Selection (QuickSelect), by k (min/median/max)
    - Monte Carlo Primality: Fermat vs. Miller-Rabin error rates, and both vs.
      deterministic trial division for speed on large n
    - Bloom Filter: false-positive rate vs. load factor
    - MinHash: estimation error vs. signature size (number of hash functions)

Each algorithm's benchmark function builds and returns its OWN list of row dicts,
kept separate from the other algorithms' rows -- several of this week's algorithms
reuse field names like "variant" for different meanings (randomized/deterministic
means something different for QuickSort than for Selection), so keeping each
algorithm's rows in their own list avoids one algorithm's rows accidentally feeding
into another's plot. export_csv() combines everything into one tidy summary_table.csv
at the end, tagged with an "algorithm" column, purely for a single combined file to
hand in -- the plotting functions never see the combined version.

Run with:
    python3 week7_randomized_benchmark.py

(Needs matplotlib in addition to the stdlib.)
"""

import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ^ This file lives in benchmarks/, one level below the project root where src/
#   actually is -- ".." goes up to that root so the imports below can resolve.

from src.randomized.randomized_quicksort import (
    randomized_quicksort, deterministic_quicksort,
    randomized_quicksort_with_comparisons, deterministic_quicksort_with_comparisons,
)
from src.randomized.randomized_selection import randomized_select, deterministic_select, randomized_select_with_stats
from src.randomized.montecarlo_primality import fermat_test, miller_rabin_test, deterministic_is_prime
from src.randomized.bloom_filter import BloomFilter
from src.randomized.minhash_similarity import generate_hash_functions, minhash_signature, estimate_jaccard_from_signatures
from src.utils.random_utils import (
    make_rng, random_int_array, worst_case_sorted_array,
    random_token_set, related_token_set, large_random_token_set,
)
from src.utils.probability_analysis import variance, true_jaccard, miller_rabin_error_bound, fermat_error_bound
from src.utils.visualization import (
    save_quicksort_figure, save_selection_figure, save_primality_figure,
    save_bloom_filter_figure, save_minhash_figure,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TIME_REPEATS = 5

# ---- Per-part input configuration ----
QUICKSORT_SIZES = [100, 500, 1000, 5000, 10000, 50000, 100000]
QUICKSORT_WORST_CASE_SIZES = [100, 500, 1000, 2000, 4000]  # kept small: O(n^2) really is slow

SELECTION_SIZES = [100, 1000, 10000, 100000]
SELECTION_K_TYPES = ["min", "median", "max"]

PRIMALITY_K_VALUES = [5, 10, 20]
PRIMALITY_TRIALS_PER_COMPOSITE = 10
PRIMALITY_COMPOSITE_POOL_SIZE = 150
CARMICHAEL_NUMBERS = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341]
LARGE_N_FOR_SPEED = [1_000_003, 10_000_019, 100_000_007, 1_000_000_007, 10_000_000_019, 100_000_000_003]
# ^ all six are verified prime (confirmed by both deterministic_is_prime and
#   miller_rabin_test agreeing). Deliberately spans > 10^6 up to > 10^11 -- trial
#   division's O(sqrt(n)) cost keeps climbing across that whole range, while the
#   probabilistic tests' O(k * log^3 n) cost barely moves, so the gap between them
#   only becomes dramatic once n gets well past the assignment's literal "> 10^6"
#   floor.
PRIMALITY_SPEED_K = 20

BLOOM_EXPECTED_ITEMS = 5000
BLOOM_TARGET_FP_RATE = 0.01
BLOOM_LOAD_FACTOR_CHECKPOINTS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
BLOOM_FP_TRIALS = 20000

MINHASH_K_VALUES = [16, 32, 64, 128, 256, 512]
MINHASH_TRIALS_PER_K = 30
MINHASH_SET_SIZE = 200
MINHASH_OVERLAP_FRACTION = 0.5
MINHASH_SPEED_SET_SIZES = [100, 1000, 10000, 30000]
MINHASH_SPEED_K = 128  # the middle of MINHASH_K_VALUES, used for the speed comparison
MINHASH_SPEED_REPEATS = 3  # fewer than TIME_REPEATS -- a signature at the largest size
                            # here already takes over a second, so 5 full repeats of
                            # BOTH variants at every size would make this one part of
                            # the benchmark dominate the whole script's runtime


def time_execution(fn, repeats=TIME_REPEATS):
    """Mean wall-clock time (seconds) of `fn()` over `repeats` calls, plus the sample
    variance of the individual timings (also in seconds^2), since Part 1 specifically
    asks for runtime variance, not just the mean."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return sum(samples) / len(samples), variance(samples)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n### Part 1: Randomized vs. Deterministic QuickSort ###")
    quicksort_rows = run_quicksort_benchmarks()

    print("\n### Part 2: Randomized vs. Deterministic Selection ###")
    selection_rows = run_selection_benchmarks()

    print("\n### Part 3: Monte Carlo Primality Testing ###")
    primality_rows = run_primality_benchmarks()

    print("\n### Part 4: Bloom Filter ###")
    bloom_rows = run_bloom_filter_benchmarks()

    print("\n### Part 5: MinHash & Similarity Estimation ###")
    minhash_rows = run_minhash_benchmarks()

    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    export_csv(
        {
            "quicksort": quicksort_rows,
            "selection": selection_rows,
            "primality": primality_rows,
            "bloom_filter": bloom_rows,
            "minhash": minhash_rows,
        },
        csv_path,
    )
    print(f"\nWrote {csv_path}")

    save_quicksort_figure(quicksort_rows, os.path.join(RESULTS_DIR, "quicksort_comparison.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'quicksort_comparison.png')}")

    save_selection_figure(selection_rows, os.path.join(RESULTS_DIR, "selection_runtime.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'selection_runtime.png')}")

    # save_primality_figure now draws two panels: false-positive rate (from the
    # fp-rate rows) AND the large-n speed comparison (from the speed rows), so it
    # needs the WHOLE primality_rows list, not a pre-filtered subset -- it does its
    # own filtering internally per panel.
    save_primality_figure(primality_rows, os.path.join(RESULTS_DIR, "primality_accuracy.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'primality_accuracy.png')}")

    save_bloom_filter_figure(bloom_rows, os.path.join(RESULTS_DIR, "bloom_filter_fp_rate.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'bloom_filter_fp_rate.png')}")

    save_minhash_figure(minhash_rows, os.path.join(RESULTS_DIR, "minhash_similarity_plot.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'minhash_similarity_plot.png')}")

    print(f"\nAll results written to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Part 1: Randomized vs. Deterministic QuickSort
# ---------------------------------------------------------------------------

def run_quicksort_benchmarks(sizes=QUICKSORT_SIZES, worst_case_sizes=QUICKSORT_WORST_CASE_SIZES, seed=1, verbose=True):
    rows = []
    rng = make_rng(seed)

    # Average-case comparison: both variants sorting the SAME random input at each
    # size. This is the main "randomized vs. deterministic" comparison the report
    # focuses on, since on generic (non-adversarial) input both run in expected
    # O(n log n) -- the interesting part is the comparison/variance numbers, not a
    # huge runtime gap.
    for n in sizes:
        arr = random_int_array(n, 0, n * 10, rng)
        for variant, sort_fn, sort_with_count_fn in [
            ("randomized", randomized_quicksort, randomized_quicksort_with_comparisons),
            ("deterministic", deterministic_quicksort, deterministic_quicksort_with_comparisons),
        ]:
            mean_time, time_var = time_execution(lambda f=sort_fn, a=arr: f(a))
            _, comparisons = sort_with_count_fn(arr)
            row = {
                "variant": variant, "n": n,
                "time_us": round(mean_time * 1e6, 3),
                "time_variance_us2": round(time_var * 1e12, 3),
                "comparisons": comparisons,
                "input": "random",
            }
            rows.append(row)
            if verbose:
                print(f"quicksort({variant:13s}, random)    n={n:7d}  time={row['time_us']:12.2f}us  "
                      f"comparisons={comparisons:10d}")

    # Worst-case demonstration: already-sorted input, where deterministic's
    # fixed last-element pivot gives the maximally unbalanced partition every single
    # time, while randomized is unaffected by the input pattern itself.
    for n in worst_case_sizes:
        arr = worst_case_sorted_array(n)
        for variant, sort_fn, sort_with_count_fn in [
            ("randomized", randomized_quicksort, randomized_quicksort_with_comparisons),
            ("deterministic", deterministic_quicksort, deterministic_quicksort_with_comparisons),
        ]:
            mean_time, time_var = time_execution(lambda f=sort_fn, a=arr: f(a), repeats=3)
            _, comparisons = sort_with_count_fn(arr)
            row = {
                "variant": f"{variant}_worst_case", "n": n,
                "time_us": round(mean_time * 1e6, 3),
                "time_variance_us2": round(time_var * 1e12, 3),
                "comparisons": comparisons,
                "input": "sorted",
            }
            rows.append(row)
            if verbose:
                print(f"quicksort({variant:13s}, sorted)    n={n:7d}  time={row['time_us']:12.2f}us  "
                      f"comparisons={comparisons:10d}")

    return rows


# ---------------------------------------------------------------------------
# Part 2: Randomized vs. Deterministic Selection
# ---------------------------------------------------------------------------

def run_selection_benchmarks(sizes=SELECTION_SIZES, k_types=SELECTION_K_TYPES, seed=2, verbose=True):
    rows = []
    rng = make_rng(seed)

    for n in sizes:
        arr = random_int_array(n, 0, n * 10, rng)
        for k_type in k_types:
            k = {"min": 1, "median": (n + 1) // 2, "max": n}[k_type]

            for variant, select_fn in [
                ("randomized", randomized_select), ("deterministic", deterministic_select),
            ]:
                mean_time, time_var = time_execution(lambda f=select_fn, a=arr, kk=k: f(a, kk))
                row = {
                    "variant": variant, "n": n, "k_type": k_type, "k": k,
                    "time_us": round(mean_time * 1e6, 3),
                    "time_variance_us2": round(time_var * 1e12, 3),
                }
                if variant == "randomized":
                    _, stats = randomized_select_with_stats(arr, k)
                    row["recursion_depth"] = stats["recursion_depth"]
                    row["good_split_fraction"] = round(
                        stats["good_split_count"] / stats["total_calls"], 4
                    ) if stats["total_calls"] else None
                rows.append(row)
                if verbose:
                    print(f"selection({variant:13s}, {k_type:6s})  n={n:7d}  time={row['time_us']:12.2f}us")

    return rows


# ---------------------------------------------------------------------------
# Part 3: Monte Carlo Primality Testing
# ---------------------------------------------------------------------------

def _empirical_false_positive_rate(composites, test_fn, k, trials_per_composite, rng):
    """Runs test_fn(composite, k, rng) `trials_per_composite` times per composite in
    `composites` and returns the overall fraction of trials that were FOOLED (test_fn
    returned True on a number that is actually composite)."""
    fooled = 0
    total = 0
    for c in composites:
        for _ in range(trials_per_composite):
            if test_fn(c, k, rng):
                fooled += 1
            total += 1
    return fooled / total if total else 0.0


def run_primality_benchmarks(
    k_values=PRIMALITY_K_VALUES,
    pool_size=PRIMALITY_COMPOSITE_POOL_SIZE,
    trials_per_composite=PRIMALITY_TRIALS_PER_COMPOSITE,
    large_ns=LARGE_N_FOR_SPEED,
    seed=3,
    verbose=True,
):
    rows = []
    rng = make_rng(seed)

    general_composites = [
        c for c in rng.sample(range(100, 100_000), pool_size * 3) if not deterministic_is_prime(c)
    ][:pool_size]

    error_bound_fn = {"fermat": fermat_error_bound, "miller_rabin": miller_rabin_error_bound}

    for k in k_values:
        for test_name, test_fn in [("fermat", fermat_test), ("miller_rabin", miller_rabin_test)]:
            rate = _empirical_false_positive_rate(general_composites, test_fn, k, trials_per_composite, rng)
            rows.append({
                "test": test_name, "k": k, "pool": "general", "false_positive_rate": round(rate, 6),
                "theoretical_bound": round(error_bound_fn[test_name](k), 10),
            })
            if verbose:
                print(f"primality({test_name:12s}, pool=general)    k={k:3d}  false_positive_rate={rate:.6f}  "
                      f"theoretical_bound={error_bound_fn[test_name](k):.10f}")

        for test_name, test_fn in [("fermat", fermat_test), ("miller_rabin", miller_rabin_test)]:
            rate = _empirical_false_positive_rate(CARMICHAEL_NUMBERS, test_fn, k, trials_per_composite, rng)
            rows.append({
                "test": test_name, "k": k, "pool": "carmichael", "false_positive_rate": round(rate, 6),
                "theoretical_bound": round(error_bound_fn[test_name](k), 10),
            })
            if verbose:
                print(f"primality({test_name:12s}, pool=carmichael)  k={k:3d}  false_positive_rate={rate:.6f}  "
                      f"theoretical_bound={error_bound_fn[test_name](k):.10f}")

    # Speed comparison for n > 10^6: deterministic trial division vs. both
    # probabilistic tests at a fixed, reasonably strong k. Variance is kept this time
    # (not discarded), since Part 6 asks for mean/variance of EVERY runtime measured,
    # not just quicksort and selection's.
    for n in large_ns:
        for test_name, fn in [
            ("deterministic", lambda nn=n: deterministic_is_prime(nn)),
            ("fermat", lambda nn=n: fermat_test(nn, PRIMALITY_SPEED_K, rng)),
            ("miller_rabin", lambda nn=n: miller_rabin_test(nn, PRIMALITY_SPEED_K, rng)),
        ]:
            mean_time, time_var = time_execution(fn, repeats=5)
            rows.append({
                "test": test_name, "n": n, "k": PRIMALITY_SPEED_K if test_name != "deterministic" else None,
                "time_us": round(mean_time * 1e6, 3),
                "time_variance_us2": round(time_var * 1e12, 3),
            })
            if verbose:
                print(f"primality speed({test_name:13s})  n={n:12d}  time={mean_time * 1e6:12.2f}us")

    return rows


# ---------------------------------------------------------------------------
# Part 4: Bloom Filter
# ---------------------------------------------------------------------------

def run_bloom_filter_benchmarks(
    expected_items=BLOOM_EXPECTED_ITEMS,
    target_fp_rate=BLOOM_TARGET_FP_RATE,
    load_factor_checkpoints=BLOOM_LOAD_FACTOR_CHECKPOINTS,
    fp_trials=BLOOM_FP_TRIALS,
    seed=4,
    verbose=True,
):
    rows = []
    rng = make_rng(seed)
    bf = BloomFilter(expected_items=expected_items, target_fp_rate=target_fp_rate)

    # A real set holding the exact same items, kept in lockstep with the filter, so
    # the memory comparison below is always "the same n items, two structures."
    equivalent_set = set()

    next_item_id = 0
    for load_factor in load_factor_checkpoints:
        target_count = int(round(load_factor * bf.m))
        while bf.items_added < target_count:
            item = f"present-{next_item_id}"
            bf.add(item)
            equivalent_set.add(item)
            next_item_id += 1

        trial_items = [f"absent-{i}" for i in range(fp_trials)]
        false_positives = sum(1 for item in trial_items if bf.check(item))
        measured_rate = false_positives / len(trial_items)

        rows.append({
            "metric": "measured", "load_factor": round(bf.current_load_factor, 4),
            "false_positive_rate": round(measured_rate, 6), "items_added": bf.items_added, "m": bf.m, "k": bf.k,
        })
        rows.append({
            "metric": "theoretical", "load_factor": round(bf.current_load_factor, 4),
            "false_positive_rate": round(bf.estimated_false_positive_rate, 6), "items_added": bf.items_added,
            "m": bf.m, "k": bf.k,
        })

        # Deterministic counterpart: memory footprint, bloom filter vs. a plain
        # python set holding the exact same items. The bloom filter side uses the
        # THEORETICAL bit-packed size (m bits = m/8 bytes), the textbook space
        # complexity a bloom filter is actually claiming -- this project's own
        # BloomFilter class stores its bits as a plain python list of 0/1 ints
        # (documented in bloom_filter.py as a deliberate simplification for
        # transparency), which actually costs roughly 8 bytes per bit due to list
        # pointer overhead, not 1/8th of a byte. Comparing against that literal
        # in-memory list would unfairly make bloom filters look worse than they
        # structurally are, so m/8 bytes is used here as the honest, standard
        # bit-packed figure instead.
        bloom_bytes = bf.m / 8
        set_bytes = sys.getsizeof(equivalent_set) + sum(sys.getsizeof(item) for item in equivalent_set)
        rows.append({"metric": "bloom_memory", "items_added": bf.items_added, "memory_bytes": round(bloom_bytes, 1)})
        rows.append({"metric": "set_memory", "items_added": bf.items_added, "memory_bytes": set_bytes})

        if verbose:
            print(f"bloom_filter  load_factor={bf.current_load_factor:.3f}  "
                  f"measured_fp={measured_rate:.6f}  theoretical_fp={bf.estimated_false_positive_rate:.6f}  "
                  f"bloom_bytes={bloom_bytes:.0f}  set_bytes={set_bytes}")

    return rows


# ---------------------------------------------------------------------------
# Part 5: MinHash & Similarity Estimation
# ---------------------------------------------------------------------------

def run_minhash_benchmarks(
    k_values=MINHASH_K_VALUES,
    trials_per_k=MINHASH_TRIALS_PER_K,
    set_size=MINHASH_SET_SIZE,
    overlap_fraction=MINHASH_OVERLAP_FRACTION,
    speed_set_sizes=MINHASH_SPEED_SET_SIZES,
    speed_k=MINHASH_SPEED_K,
    speed_repeats=MINHASH_SPEED_REPEATS,
    seed=5,
    verbose=True,
):
    rows = []
    rng = make_rng(seed)

    base_set = random_token_set(set_size, rng)
    related_set = related_token_set(base_set, set_size, overlap_fraction, rng)
    true_sim = true_jaccard(base_set, related_set)

    for num_hashes in k_values:
        errors = []
        estimates = []
        for trial in range(trials_per_k):
            hash_functions = generate_hash_functions(num_hashes, rng)
            sig_a = minhash_signature(base_set, hash_functions)
            sig_b = minhash_signature(related_set, hash_functions)
            estimate = estimate_jaccard_from_signatures(sig_a, sig_b)
            estimates.append(estimate)
            errors.append(abs(estimate - true_sim))

        mean_abs_error = sum(errors) / len(errors)
        mean_estimate = sum(estimates) / len(estimates)
        rows.append({
            "metric": "error", "num_hashes": num_hashes,
            "mean_abs_error": round(mean_abs_error, 6),
            "mean_estimate": round(mean_estimate, 6),
            "true_jaccard": round(true_sim, 6),
        })
        if verbose:
            print(f"minhash  k={num_hashes:4d}  true={true_sim:.4f}  "
                  f"mean_estimate={mean_estimate:.4f}  mean_abs_error={mean_abs_error:.4f}")

    # Deterministic counterpart: speed, computing exact Jaccard directly (a single
    # set intersection and union) vs. MinHash (build a signature for each set, then
    # compare signatures), at a representative signature size (speed_k). MinHash is
    # NOT expected to win here -- it does strictly more hashing work than one direct
    # set intersection -- its real payoff is a signature of fixed size speed_k,
    # independent of how big the original sets are, which matters for STORAGE and
    # for repeated comparisons without the original sets on hand, not raw one-off
    # CPU time. Both get their own fresh random set pair at each size.
    for size in speed_set_sizes:
        set_a = large_random_token_set(size, rng)
        set_b = large_random_token_set(size, rng)
        hash_functions = generate_hash_functions(speed_k, rng)

        exact_mean, exact_var = time_execution(lambda a=set_a, b=set_b: true_jaccard(a, b), repeats=speed_repeats)
        minhash_mean, minhash_var = time_execution(
            lambda a=set_a, b=set_b, hf=hash_functions: estimate_jaccard_from_signatures(
                minhash_signature(a, hf), minhash_signature(b, hf)
            ),
            repeats=speed_repeats,
        )
        rows.append({
            "metric": "speed", "variant": "exact", "set_size": size,
            "time_us": round(exact_mean * 1e6, 3), "time_variance_us2": round(exact_var * 1e12, 3),
        })
        rows.append({
            "metric": "speed", "variant": "minhash", "set_size": size, "num_hashes": speed_k,
            "time_us": round(minhash_mean * 1e6, 3), "time_variance_us2": round(minhash_var * 1e12, 3),
        })
        if verbose:
            print(f"minhash speed  set_size={size:7d}  exact={exact_mean * 1e6:10.2f}us  "
                  f"minhash(k={speed_k})={minhash_mean * 1e6:10.2f}us")

    return rows


def export_csv(rows_by_algorithm, path):
    """Flattens every algorithm's row list into one CSV, tagged with an "algorithm"
    column. The union of every possible field across all five algorithms becomes the
    fieldnames list, with missing fields left blank for rows that don't use them --
    this week's five algorithms have too little in common to share one tidy schema
    the way Week 6's four DP benchmarks could."""
    fieldnames = ["algorithm"]
    seen = set(fieldnames)
    for algo_rows in rows_by_algorithm.values():
        for row in algo_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, algo_rows in rows_by_algorithm.items():
            for row in algo_rows:
                writer.writerow({"algorithm": algorithm, **row})


if __name__ == "__main__":
    main()
