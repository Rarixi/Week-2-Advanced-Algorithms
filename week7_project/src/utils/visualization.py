"""
visualization.py
------------------
Shared plotting helpers for benchmarks/week7_randomized_benchmark.py. Each of the five
algorithms benchmarked this week needs a different comparison shape, so each gets its
own save_*_figure function rather than sharing one generic plot.

Every runtime line plotted with error bars uses the timing VARIANCE each benchmark
function records (not just the mean) -- the bars show plus/minus one standard
deviation (sqrt of the variance), so "how much did repeated runs actually vary" is
visible directly on the plot, not just sitting in a CSV column.

Provides:
    save_quicksort_figure(rows, out_path)       -- randomized vs. deterministic
                                                     quicksort: time (with variance
                                                     error bars) and comparison count,
                                                     2 panels
    save_selection_figure(rows, out_path)        -- randomized vs. deterministic
                                                     selection: runtime (with variance
                                                     error bars) across k=min/median/
                                                     max, 1 panel
    save_primality_figure(rows, out_path)        -- empirical false-positive rate vs.
                                                     k (with the theoretical (1/4)^k
                                                     and (1/2)^k error bounds overlaid)
                                                     PLUS deterministic vs. probabilistic
                                                     speed vs. n (with variance error
                                                     bars), 2 panels
    save_bloom_filter_figure(rows, out_path)     -- measured false-positive rate vs.
                                                     load factor PLUS memory footprint,
                                                     bloom filter vs. an equivalent
                                                     plain Python set, 2 panels
    save_minhash_figure(rows, out_path)          -- estimation error vs. signature size
                                                     PLUS speed, MinHash vs. computing
                                                     exact Jaccard directly, 2 panels
"""

import math

import matplotlib
matplotlib.use("Agg")  # write PNGs directly, no need for a display
import matplotlib.pyplot as plt


def _series_for(rows, key_field, key_value, x_field, y_field):
    """Sorted (x, y) for every row matching rows[key_field] == key_value, pulled out
    of the flat `rows` list each benchmark function returns. Uses .get() for the
    filter, not rows[key_field] -- several of this week's row lists mix rows from
    different metrics that don't all share the same fields (an "error" row has no
    "variant" field, a "measured" row has no "num_hashes" field, etc.), and a plain
    lookup would raise KeyError on every row that happens not to have key_field at
    all, instead of just skipping it the way a non-matching value should."""
    matching = [r for r in rows if r.get(key_field) == key_value]
    matching.sort(key=lambda r: r[x_field])
    xs = [r[x_field] for r in matching]
    ys = [r[y_field] for r in matching]
    return xs, ys


def _series_with_stddev(rows, key_field, key_value, x_field, y_field, var_field):
    """Same as _series_for, but also returns the standard deviation (sqrt of the
    variance field, floored at 0 to guard against tiny negative values from floating
    point rounding) alongside each y value, for plotting as error bars."""
    matching = [r for r in rows if r.get(key_field) == key_value]
    matching.sort(key=lambda r: r[x_field])
    xs = [r[x_field] for r in matching]
    ys = [r[y_field] for r in matching]
    stddevs = [math.sqrt(max(r.get(var_field, 0.0), 0.0)) for r in matching]
    return xs, ys, stddevs


def save_quicksort_figure(rows, out_path):
    """Panel 1: mean time vs. array size (log-log) with plus/minus one standard
    deviation error bars, randomized vs. deterministic -- the deterministic line
    should bend upward sharply once the worst-case input kicks in, while the
    randomized line should track n*log(n) smoothly. Panel 2: comparison count vs.
    array size, measured against the theoretical 2n*ln(n) reference curve for
    randomized quicksort."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for variant, style in [("randomized", "-o"), ("deterministic", "-s")]:
        xs, ys, stddevs = _series_with_stddev(rows, "variant", variant, "n", "time_us", "time_variance_us2")
        if xs:
            ax.errorbar(xs, ys, yerr=stddevs, fmt=style, label=variant, capsize=3, alpha=0.85)
    ax.set_xlabel("array size (n)")
    ax.set_ylabel("mean time, +/- 1 std dev (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("QuickSort: time vs. array size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for variant, style in [("randomized", "-o"), ("deterministic", "-s")]:
        xs, ys = _series_for(rows, "variant", variant, "n", "comparisons")
        if xs:
            ax.plot(xs, ys, style, label=f"{variant} (measured)")
    ref_xs, _ = _series_for(rows, "variant", "randomized", "n", "comparisons")
    if ref_xs:
        ref_ys = [2 * n * math.log(n) for n in ref_xs]
        ax.plot(ref_xs, ref_ys, "--", label="2n ln(n) reference", alpha=0.6)
    ax.set_xlabel("array size (n)")
    ax.set_ylabel("comparisons")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("QuickSort: comparisons vs. array size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Randomized QuickSort vs. Deterministic QuickSort")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_selection_figure(rows, out_path):
    """Single panel: mean time vs. array size with plus/minus one standard deviation
    error bars, one line per (variant, k-type) combination -- six lines total
    (randomized/deterministic x min/median/max). Lets the plot show both the
    randomized-vs-deterministic gap AND whether the choice of k (min, median, or max)
    changes anything."""
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {
        ("randomized", "min"): "-o", ("randomized", "median"): "-s", ("randomized", "max"): "-^",
        ("deterministic", "min"): "--o", ("deterministic", "median"): "--s", ("deterministic", "max"): "--^",
    }
    for (variant, k_type), style in styles.items():
        matching = [r for r in rows if r["variant"] == variant and r["k_type"] == k_type]
        matching.sort(key=lambda r: r["n"])
        xs = [r["n"] for r in matching]
        ys = [r["time_us"] for r in matching]
        stddevs = [math.sqrt(max(r.get("time_variance_us2", 0.0), 0.0)) for r in matching]
        if xs:
            ax.errorbar(xs, ys, yerr=stddevs, fmt=style, label=f"{variant} / {k_type}", capsize=2, alpha=0.8)

    ax.set_xlabel("array size (n)")
    ax.set_ylabel("mean time, +/- 1 std dev (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Selection (QuickSelect): randomized vs. deterministic, by k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_primality_figure(rows, out_path):
    """Panel 1: empirical false-positive rate vs. k (number of rounds), one line per
    (test, pool) combination -- "general" (random composites, which essentially never
    fool either test, even at k=5) and "carmichael" (numbers specifically constructed
    to fool Fermat's test, where the real gap between Fermat and Miller-Rabin actually
    shows up) -- with each test's THEORETICAL error bound ((1/4)^k for Miller-Rabin,
    (1/2)^k for Fermat) drawn as a dashed reference curve. Panel 2: deterministic
    trial division vs. both probabilistic tests, mean speed vs. n with plus/minus one
    standard deviation error bars, log-log, showing trial division's O(sqrt(n)) cost
    pulling away from the probabilistic tests' near-flat O(k log^3 n) cost as n grows.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    combos = [
        ("fermat", "general", "-o", "tab:blue"),
        ("fermat", "carmichael", "-o", "tab:red"),
        ("miller_rabin", "general", "--s", "tab:blue"),
        ("miller_rabin", "carmichael", "--s", "tab:red"),
    ]
    for test_name, pool, style, color in combos:
        matching = [r for r in rows if r.get("test") == test_name and r.get("pool") == pool]
        matching.sort(key=lambda r: r["k"])
        xs = [r["k"] for r in matching]
        ys = [max(r["false_positive_rate"], 1e-6) for r in matching]
        if xs:
            ax.plot(xs, ys, style, color=color, label=f"{test_name} / {pool}", alpha=0.85)

    bound_xs = sorted({
        r["k"] for r in rows
        if r.get("test") in ("fermat", "miller_rabin") and "false_positive_rate" in r
    })
    # ^ restricted to fp-rate rows specifically, not just "has a k field" -- the speed
    #   comparison rows also carry a "k" value (PRIMALITY_SPEED_K) and would otherwise
    #   sneak into this reference line's x-axis too if that constant ever diverged
    #   from the k values used for the error-rate sweep.
    if bound_xs:
        mr_bounds = [max(0.25 ** k, 1e-6) for k in bound_xs]
        fermat_bounds = [max(0.5 ** k, 1e-6) for k in bound_xs]
        ax.plot(bound_xs, mr_bounds, ":", color="tab:green", label="miller-rabin bound (1/4)^k", alpha=0.7)
        ax.plot(bound_xs, fermat_bounds, ":", color="tab:orange", label="fermat loose bound (1/2)^k", alpha=0.7)

    ax.set_xlabel("k (number of rounds)")
    ax.set_ylabel("false-positive rate")
    ax.set_yscale("log")
    ax.set_title("Empirical rate vs. k, with theoretical bounds")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for test_name, style in [("deterministic", "-o"), ("fermat", "-s"), ("miller_rabin", "-^")]:
        matching = [r for r in rows if r.get("test") == test_name and "n" in r]
        matching.sort(key=lambda r: r["n"])
        xs = [r["n"] for r in matching]
        ys = [r["time_us"] for r in matching]
        stddevs = [math.sqrt(max(r.get("time_variance_us2", 0.0), 0.0)) for r in matching]
        if xs:
            ax.errorbar(xs, ys, yerr=stddevs, fmt=style, label=test_name, capsize=3, alpha=0.85)
    ax.set_xlabel("n")
    ax.set_ylabel("mean time, +/- 1 std dev (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Speed: trial division vs. probabilistic tests")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Monte Carlo Primality Testing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_bloom_filter_figure(rows, out_path):
    """Panel 1: measured false-positive rate vs. load factor (items inserted / bit
    array capacity), with the theoretical (1 - e^(-kn/m))^k curve included for
    reference. Panel 2: memory footprint vs. items inserted, bloom filter (a flat
    line -- its bit array size m never changes no matter how many items go in) vs. an
    equivalent plain Python set holding the same items (which grows steadily), the
    "deterministic counterpart" comparison -- a set never has false positives, but
    costs far more memory per item than the bloom filter's fixed-size bit array."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    xs, ys = _series_for(rows, "metric", "measured", "load_factor", "false_positive_rate")
    if xs:
        ax.plot(xs, ys, "-o", label="measured")
    tx, ty = _series_for(rows, "metric", "theoretical", "load_factor", "false_positive_rate")
    if tx:
        ax.plot(tx, ty, "--", label="theoretical", alpha=0.7)
    ax.set_xlabel("load factor (n items / m bits)")
    ax.set_ylabel("false-positive rate")
    ax.set_yscale("log")
    ax.set_title("False-positive rate vs. load factor")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    bx, by = _series_for(rows, "metric", "bloom_memory", "items_added", "memory_bytes")
    if bx:
        ax.plot(bx, by, "-o", label="bloom filter (bit array)")
    sx, sy = _series_for(rows, "metric", "set_memory", "items_added", "memory_bytes")
    if sx:
        ax.plot(sx, sy, "-s", label="plain python set (exact)")
    ax.set_xlabel("items inserted")
    ax.set_ylabel("memory (bytes)")
    ax.set_yscale("log")
    ax.set_title("Memory: bloom filter vs. an exact set")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Bloom Filter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_minhash_figure(rows, out_path):
    """Panel 1: mean absolute estimation error vs. number of hash functions
    (signature size k), log-log, since error should shrink roughly as 1/sqrt(k).
    Panel 2: speed, MinHash (signature build + compare) vs. computing exact Jaccard
    directly, vs. set size -- the "deterministic counterpart" comparison. MinHash is
    NOT expected to win this race for a single one-off comparison (it does strictly
    more hashing work than a direct set intersection), its real advantage is a much
    smaller signature to store or send once computed, not raw CPU time for one
    comparison -- see the report for the fuller explanation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    xs, ys = _series_for(rows, "metric", "error", "num_hashes", "mean_abs_error")
    if xs:
        ax.plot(xs, ys, "-o", label="measured mean error")
    ax.set_xlabel("number of hash functions (k)")
    ax.set_ylabel("mean absolute error vs. true Jaccard")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Estimation error vs. signature size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for variant, style in [("exact", "-o"), ("minhash", "-s")]:
        xs, ys = _series_for(rows, "variant", variant, "set_size", "time_us")
        if xs:
            ax.plot(xs, ys, style, label=variant)
    ax.set_xlabel("set size")
    ax.set_ylabel("mean time (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Speed: MinHash vs. exact Jaccard")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("MinHash & Similarity Estimation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
