"""
visualization.py
------------------
Plotting helpers for week 8's benchmark. Two figures come out of this file,
matching the two png paths the project structure calls for.

runtime_growth.png
    panel 1, subset sum backtracking (exponential) vs its heuristic
    (polynomial), runtime vs n, log scale on the time axis since the
    backtracking side grows so fast a linear axis would flatten everything
    else to the floor.
    panel 2, sat backtracking runtime vs number of variables, same idea,
    one exponential curve on its own since there's no polynomial sat solver
    to compare it against.
    panel 3, vertex cover approx and tsp nearest neighbor runtime vs n,
    both polynomial, shown on a linear scale so the contrast with the first
    two panels (which need log scale just to stay on the page) makes the
    point about growth rates on its own.

approximation_quality.png
    panel 1, vertex cover's measured approximation ratio (approx size over
    true optimal size, from brute force on small instances) vs n.
    panel 2, tsp's measured approximation ratio (nearest neighbor tour
    length over optimal tour length, from brute force on small instances)
    vs n.
    panel 3, the runtime vs accuracy trade off curve, approximation ratio
    plotted against runtime for both algorithms so slower/more thorough
    settings and their accuracy payoff sit on one chart.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series_for(rows, key_field, key_value, x_field, y_field):
    matching = [r for r in rows if r.get(key_field) == key_value and y_field in r]
    matching.sort(key=lambda r: r[x_field])
    xs = [r[x_field] for r in matching]
    ys = [r[y_field] for r in matching]
    return xs, ys


def _series_with_stddev(rows, key_field, key_value, x_field, y_field, var_field):
    matching = [r for r in rows if r.get(key_field) == key_value and y_field in r]
    matching.sort(key=lambda r: r[x_field])
    xs = [r[x_field] for r in matching]
    ys = [r[y_field] for r in matching]
    stddevs = [math.sqrt(max(r.get(var_field, 0.0), 0.0)) for r in matching]
    return xs, ys, stddevs


def save_runtime_growth_figure(subset_sum_rows, sat_rows, vertex_cover_rows, tsp_rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    for variant, style, color in [
        ("backtracking", "o-", "tab:red"),
        ("heuristic", "s-", "tab:blue"),
    ]:
        xs, ys, stddevs = _series_with_stddev(
            subset_sum_rows, "variant", variant, "n", "time_us", "time_variance_us2"
        )
        if xs:
            ax.errorbar(xs, ys, yerr=stddevs, fmt=style, color=color, label=variant, capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("n (number of items)")
    ax.set_ylabel("time (microseconds, log scale)")
    ax.set_title("Subset Sum, Backtracking vs Heuristic")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    xs, ys = _series_for(sat_rows, "variant", "backtracking", "num_vars", "time_us")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:purple", label="sat backtracking")
    ax.set_yscale("log")
    ax.set_xlabel("number of variables")
    ax.set_ylabel("time (microseconds, log scale)")
    ax.set_title("SAT Solver Runtime Growth")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    xs, ys = _series_for(vertex_cover_rows, "variant", "approx", "n", "time_us")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:green", label="vertex cover approx")
    xs, ys = _series_for(tsp_rows, "variant", "nearest_neighbor", "n", "time_us")
    if xs:
        ax.plot(xs, ys, "s-", color="tab:orange", label="tsp nearest neighbor")
    ax.set_xlabel("n (vertices / cities)")
    ax.set_ylabel("time (microseconds, linear scale)")
    ax.set_title("Approximation Algorithms, Polynomial Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mean_per_x(xs, ys):
    """Collapse possibly-repeated x values (several trials at the same n)
    down to one mean y per distinct x, sorted by x, for drawing a clean
    trend line through a scatter of individual trials."""
    buckets = {}
    for x, y in zip(xs, ys):
        buckets.setdefault(x, []).append(y)
    distinct_xs = sorted(buckets)
    mean_ys = [sum(buckets[x]) / len(buckets[x]) for x in distinct_xs]
    return distinct_xs, mean_ys


def save_approximation_quality_figure(vertex_cover_ratio_rows, tsp_ratio_rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    xs, ys = _series_for(vertex_cover_ratio_rows, "metric", "ratio", "n", "ratio")
    if xs:
        ax.scatter(xs, ys, color="tab:green", alpha=0.5, label="individual trials")
        mean_xs, mean_ys = _mean_per_x(xs, ys)
        ax.plot(mean_xs, mean_ys, "o-", color="tab:green", linewidth=2, label="mean ratio")
    ax.axhline(2.0, linestyle=":", color="tab:gray", label="2-approximation bound")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("approx size / optimal size")
    ax.set_title("Vertex Cover Approximation Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs, ys = _series_for(tsp_ratio_rows, "metric", "ratio", "n", "ratio")
    if xs:
        ax.scatter(xs, ys, color="tab:orange", alpha=0.5, label="individual trials")
        mean_xs, mean_ys = _mean_per_x(xs, ys)
        ax.plot(mean_xs, mean_ys, "s-", color="tab:orange", linewidth=2, label="mean ratio")
    ax.axhline(1.0, linestyle=":", color="tab:gray", label="optimal (ratio = 1)")
    ax.set_xlabel("n (cities)")
    ax.set_ylabel("heuristic tour / optimal tour")
    ax.set_title("TSP Nearest Neighbor Approximation Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    xs, ys = _series_for(vertex_cover_ratio_rows, "metric", "ratio", "approx_time_us", "ratio")
    if xs:
        ax.scatter(xs, ys, color="tab:green", label="vertex cover")
    xs, ys = _series_for(tsp_ratio_rows, "metric", "ratio", "approx_time_us", "ratio")
    if xs:
        ax.scatter(xs, ys, color="tab:orange", label="tsp")
    ax.set_xlabel("approximation algorithm time (microseconds)")
    ax.set_ylabel("approximation ratio")
    ax.set_title("Runtime vs Accuracy Trade Off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
