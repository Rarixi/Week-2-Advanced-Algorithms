"""
visualization.py
-----------------
Shared plotting helpers for benchmarks/week5_dp_benchmark.py.

Each algorithm (fibonacci, knapsack, lcs) gets one PNG with three side-by-side panels:
    1. Time vs. input size (log-log) -- the naive recursive version's curve stops at its
       cutoff size (it's exponential -- going further isn't practical), while memo and
       tabulation continue across the full range, so the plot visually shows recursion
       becoming impractical while DP keeps scaling.
    2. Function calls vs. input size (log-log) -- same idea, but counting calls instead
       of time (see utils/timer.py's count_calls_and_depth).
    3. Speedup factor (recursive time / DP time) vs. input size, computed only at the
       input sizes where a recursive measurement actually exists.
"""

import matplotlib
matplotlib.use("Agg")  # write PNGs directly, no need for a display
import matplotlib.pyplot as plt


def series_for(rows, algorithm, variant):
    """Sorted (sizes, times, calls) lists for one (algorithm, variant) pair, pulled out
    of the flat `rows` list every benchmark run produces."""
    matching = [r for r in rows if r["algorithm"] == algorithm and r["variant"] == variant]
    matching.sort(key=lambda r: r["input_size"])
    sizes = [r["input_size"] for r in matching]
    times = [r["time_us"] for r in matching]
    calls = [r["calls"] for r in matching]
    return sizes, times, calls


def save_comparison_figure(rows, algorithm, out_path, title, xlabel):
    """Builds and saves the three-panel comparison PNG described above for one
    algorithm's rows (already filtered by algorithm elsewhere isn't required -- this
    filters by `algorithm` itself, so `rows` can be the full combined benchmark table)."""
    variants = [("recursive", "-o"), ("memo", "-s"), ("tabulation", "-^")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: time vs. input size
    ax = axes[0]
    for variant, style in variants:
        sizes, times, _ = series_for(rows, algorithm, variant)
        if sizes:
            ax.plot(sizes, times, style, label=variant)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean time (microseconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Time vs. input size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: calls vs. input size
    ax = axes[1]
    for variant, style in variants:
        sizes, _, calls = series_for(rows, algorithm, variant)
        pairs = [(s, c) for s, c in zip(sizes, calls) if c is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, style, label=variant)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("function calls")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Calls vs. input size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: speedup factor (recursive time / DP time), only where recursive exists
    ax = axes[2]
    rec_sizes, rec_times, _ = series_for(rows, algorithm, "recursive")
    rec_by_size = dict(zip(rec_sizes, rec_times))
    for variant, style in [("memo", "-s"), ("tabulation", "-^")]:
        dp_sizes, dp_times, _ = series_for(rows, algorithm, variant)
        xs, ys = [], []
        for size, dp_time in zip(dp_sizes, dp_times):
            if size in rec_by_size and dp_time > 0:
                xs.append(size)
                ys.append(rec_by_size[size] / dp_time)
        if xs:
            ax.plot(xs, ys, style, label=f"recursive / {variant}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("speedup factor (x)")
    ax.set_yscale("log")
    ax.set_title("Speedup: recursive vs. DP")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
