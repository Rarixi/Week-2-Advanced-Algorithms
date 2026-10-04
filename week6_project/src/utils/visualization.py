"""
visualization.py
------------------
Shared plotting helpers for benchmarks/week6_dp_advanced_benchmark.py. Each of the
four algorithms benchmarked this week has a different shape of comparison to make
(3-way recursive/memo/tabulation, 2D-vs-1D memory, single-algorithm scaling, or
brute-force-vs-DP), so each gets its own save_*_figure function rather than sharing
one generic plot shape.

Provides:
    save_mcm_figure(rows, out_path)                 -- recursive/memo/tabulation: time
                                                        and speedup, 2 panels
    save_knapsack_space_figure(rows, out_path)       -- 2D vs. 1D: memory and time, 2 panels
    save_floyd_warshall_figure(rows, out_path)       -- single-algorithm time vs. n,
                                                        with a fitted O(n^3) reference curve
    save_tsp_figure(rows, out_path)                  -- brute force vs. bitmask DP: time
                                                        vs. n, 1 panel (log y-axis, since
                                                        both variants are exponential)
"""

import matplotlib
matplotlib.use("Agg")  # write PNGs directly, no need for a display
import matplotlib.pyplot as plt


def _series_for(rows, algorithm, variant, field="time_us"):
    """Sorted (sizes, values) for one (algorithm, variant) pair, pulled out of the
    flat `rows` list every benchmark run produces."""
    matching = [r for r in rows if r["algorithm"] == algorithm and r["variant"] == variant]
    matching.sort(key=lambda r: r["input_size"])
    sizes = [r["input_size"] for r in matching]
    values = [r[field] for r in matching]
    return sizes, values


def save_mcm_figure(rows, out_path):
    """Panel 1: time vs. chain length (log-log), recursive/memo/tabulation -- the
    recursive line stops at its cutoff (Catalan-number growth makes it impractical
    beyond a small chain length), memo and tabulation continue across the full range.
    Panel 2: speedup factor (recursive time / DP time), only at chain lengths where a
    recursive measurement exists.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for variant, style in [("recursive", "-o"), ("memo", "-s"), ("tabulation", "-^")]:
        sizes, times = _series_for(rows, "mcm", variant)
        if sizes:
            ax.plot(sizes, times, style, label=variant)
    ax.set_xlabel("chain length (number of matrices)")
    ax.set_ylabel("mean time (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("MCM: time vs. chain length")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    rec_sizes, rec_times = _series_for(rows, "mcm", "recursive")
    rec_by_size = dict(zip(rec_sizes, rec_times))
    for variant, style in [("memo", "-s"), ("tabulation", "-^")]:
        dp_sizes, dp_times = _series_for(rows, "mcm", variant)
        xs, ys = [], []
        for size, dp_time in zip(dp_sizes, dp_times):
            if size in rec_by_size and dp_time > 0:
                xs.append(size)
                ys.append(rec_by_size[size] / dp_time)
        if xs:
            ax.plot(xs, ys, style, label=f"recursive / {variant}")
    ax.set_xlabel("chain length (number of matrices)")
    ax.set_ylabel("speedup factor (x)")
    ax.set_yscale("log")
    ax.set_title("MCM: speedup, recursive vs. DP")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Matrix Chain Multiplication: recursive vs. memo vs. tabulation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_knapsack_space_figure(rows, out_path):
    """Panel 1: peak memory vs. capacity (log-log), 2D table vs. 1D array -- the whole
    point of knapsack_1d, so this is the figure that should show the biggest, cleanest
    gap in the entire benchmark. Panel 2: time vs. capacity -- included to confirm the
    space optimization does NOT come at a time cost (both are O(n * capacity)), so the
    two lines should track closely together rather than diverge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for variant, style in [("2d", "-o"), ("1d", "-s")]:
        sizes, mems = _series_for(rows, "knapsack_space", variant, field="peak_memory_kb")
        if sizes:
            ax.plot(sizes, mems, style, label=variant)
    ax.set_xlabel("capacity")
    ax.set_ylabel("peak traced memory (KB)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Peak memory: 2D table vs. 1D array")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for variant, style in [("2d", "-o"), ("1d", "-s")]:
        sizes, times = _series_for(rows, "knapsack_space", variant, field="time_us")
        if sizes:
            ax.plot(sizes, times, style, label=variant)
    ax.set_xlabel("capacity")
    ax.set_ylabel("mean time (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Time: 2D table vs. 1D array")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("0/1 Knapsack: space-optimized (1D) vs. full table (2D)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_floyd_warshall_figure(rows, out_path):
    """Single panel: time vs. number of nodes (log-log), with a reference curve
    (scaled to match at the smallest measured size) showing what pure O(n^3) growth
    would look like, so the measured curve's closeness to that reference line is a
    visual confirmation of the algorithm's cubic time complexity.
    """
    sizes, times = _series_for(rows, "floyd_warshall", "tabulation")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, times, "-o", label="measured")

    if sizes and times:
        # Scale the O(n^3) reference curve to match the measured curve at the
        # smallest size, so both lines start at the same point and any divergence
        # at larger sizes reflects a real difference in growth RATE, not just a
        # different additive constant.
        n0, t0 = sizes[0], times[0]
        reference = [t0 * (n / n0) ** 3 for n in sizes]
        ax.plot(sizes, reference, "--", label="O(n^3) reference", alpha=0.6)

    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("mean time (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Floyd-Warshall: time vs. graph size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_tsp_figure(rows, out_path):
    """Single panel: time vs. number of cities (log y-axis) -- brute force's line
    stops at its cutoff (O(n!) makes anything past ~10 cities impractical), while
    bitmask DP continues much further (O(n^2 * 2^n) is still exponential, but with a
    dramatically smaller base), making the gap between "impractical" and "merely
    exponential" visually obvious.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for variant, style in [("brute_force", "-o"), ("bitmask_dp", "-s")]:
        sizes, times = _series_for(rows, "tsp", variant)
        if sizes:
            ax.plot(sizes, times, style, label=variant)

    ax.set_xlabel("number of cities")
    ax.set_ylabel("mean time (microseconds)")
    ax.set_yscale("log")
    ax.set_title("TSP: brute force vs. bitmask DP (Held-Karp)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
