"""
visualization.py
------------------
Plotting helpers for week 9's benchmark. Three figures come out of this
file, matching the three png paths the project structure calls for.

approx_ratios.png
    one panel per algorithm (vertex cover, set cover, metric tsp, max-cut),
    each showing the measured approximation ratio against n, with that
    algorithm's theoretical bound drawn in for reference where one exists
    (2x for vertex cover and tsp, 1/2 for max-cut's cut value ratio).

cost_vs_optimal.png
    facility location and metric tsp both explicitly ask to be shown as
    raw cost against the true optimal cost, not just a ratio, so this
    figure gives each its own panel, approximate cost and optimal cost
    plotted together against n.

randomized_maxcut_results.png
    max-cut gets its own figure since it's the one randomized algorithm
    this week, cut value against n (individual random trials, their mean,
    local search's mean, the true optimum, and the theoretical |E| / 2
    expectation all together), the variance of the random cut value
    against n, and runtime against n for the random assignment versus
    local search, covering ratio, variance, and scalability all for the
    one algorithm where repeated trials actually mean something.
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


def _mean_per_x(xs, ys):
    """Collapse possibly repeated x values (several trials at the same n)
    down to one mean y per distinct x, sorted by x."""
    buckets = {}
    for x, y in zip(xs, ys):
        buckets.setdefault(x, []).append(y)
    distinct_xs = sorted(buckets)
    mean_ys = [sum(buckets[x]) / len(buckets[x]) for x in distinct_xs]
    return distinct_xs, mean_ys


def save_approx_ratios_figure(vertex_cover_rows, set_cover_rows, tsp_rows, maxcut_rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0][0]
    xs, ys = _series_for(vertex_cover_rows, "metric", "ratio", "n", "ratio")
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

    ax = axes[0][1]
    det_xs, det_ys = _series_for(set_cover_rows, "variant", "deterministic", "n", "ratio")
    if det_xs:
        ax.plot(det_xs, det_ys, "o-", color="tab:red", label="deterministic rounding")
    rand_xs, rand_ys = _series_for(set_cover_rows, "variant", "randomized", "n", "mean_ratio")
    if rand_xs:
        ax.plot(rand_xs, rand_ys, "s-", color="tab:blue", label="randomized rounding (mean)")
    ax.set_xlabel("n (universe size)")
    ax.set_ylabel("rounded cost / optimal cost")
    ax.set_title("Set Cover Rounding Approximation Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    xs, ys = _series_for(tsp_rows, "metric", "ratio", "n", "ratio")
    if xs:
        ax.scatter(xs, ys, color="tab:orange", alpha=0.5, label="individual trials")
        mean_xs, mean_ys = _mean_per_x(xs, ys)
        ax.plot(mean_xs, mean_ys, "s-", color="tab:orange", linewidth=2, label="mean ratio")
    ax.axhline(2.0, linestyle=":", color="tab:gray", label="2-approximation bound")
    ax.set_xlabel("n (cities)")
    ax.set_ylabel("approx tour / optimal tour")
    ax.set_title("Metric TSP (MST Doubling) Approximation Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    random_xs, random_ys = _series_for(maxcut_rows, "metric", "ratio", "n", "random_ratio")
    if random_xs:
        ax.scatter(random_xs, random_ys, color="tab:purple", alpha=0.4, label="random trials")
        mean_xs, mean_ys = _mean_per_x(random_xs, random_ys)
        ax.plot(mean_xs, mean_ys, "o-", color="tab:purple", linewidth=2, label="random (mean)")
    local_xs, local_ys = _series_for(maxcut_rows, "metric", "ratio", "n", "local_ratio")
    if local_xs:
        mean_xs, mean_ys = _mean_per_x(local_xs, local_ys)
        ax.plot(mean_xs, mean_ys, "^-", color="tab:brown", linewidth=2, label="local search (mean)")
    ax.axhline(0.5, linestyle=":", color="tab:gray", label="1/2-approximation bound")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("cut value / optimal cut")
    ax.set_title("Randomized Max-Cut Approximation Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cost_vs_optimal_figure(facility_location_rows, tsp_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    xs, greedy_ys = _series_for(facility_location_rows, "metric", "ratio", "n", "greedy_cost")
    _xs2, optimal_ys = _series_for(facility_location_rows, "metric", "ratio", "n", "optimal_cost")
    if xs:
        ax.plot(xs, greedy_ys, "o-", color="tab:green", label="greedy cost")
        ax.plot(xs, optimal_ys, "s--", color="tab:gray", label="true optimal cost")
    ax.set_xlabel("n (clients, small instances)")
    ax.set_ylabel("total cost (opening + connection)")
    ax.set_title("Facility Location, Greedy vs Optimal Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs, approx_ys = _series_for(tsp_rows, "metric", "ratio", "n", "approx_length")
    _xs2, optimal_ys = _series_for(tsp_rows, "metric", "ratio", "n", "optimal_length")
    if xs:
        mean_xs, mean_approx = _mean_per_x(xs, approx_ys)
        _mx, mean_optimal = _mean_per_x(xs, optimal_ys)
        ax.plot(mean_xs, mean_approx, "o-", color="tab:orange", label="mst doubling tour length")
        ax.plot(mean_xs, mean_optimal, "s--", color="tab:gray", label="true optimal tour length")
    ax.set_xlabel("n (cities, small instances)")
    ax.set_ylabel("tour length")
    ax.set_title("Metric TSP, Approx vs Optimal Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_randomized_maxcut_results_figure(maxcut_ratio_rows, maxcut_runtime_rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    xs, random_cuts = _series_for(maxcut_ratio_rows, "metric", "ratio", "n", "random_cut")
    if xs:
        ax.scatter(xs, random_cuts, color="tab:purple", alpha=0.35, label="individual random trials")
        mean_xs, mean_random = _mean_per_x(xs, random_cuts)
        ax.plot(mean_xs, mean_random, "o-", color="tab:purple", linewidth=2, label="random (mean)")
    _xs2, local_cuts = _series_for(maxcut_ratio_rows, "metric", "ratio", "n", "local_cut")
    if xs:
        _mx, mean_local = _mean_per_x(xs, local_cuts)
        ax.plot(mean_xs, mean_local, "^-", color="tab:brown", linewidth=2, label="local search (mean)")
    _xs3, optimal_cuts = _series_for(maxcut_ratio_rows, "metric", "ratio", "n", "optimal_cut")
    if xs:
        _mx, mean_optimal = _mean_per_x(xs, optimal_cuts)
        ax.plot(mean_xs, mean_optimal, "s--", color="tab:gray", linewidth=2, label="true optimum")
    _xs4, expected_cuts = _series_for(maxcut_ratio_rows, "metric", "ratio", "n", "expected_cut")
    if xs:
        _mx, mean_expected = _mean_per_x(xs, expected_cuts)
        ax.plot(mean_xs, mean_expected, ":", color="black", linewidth=2, label="theoretical |E| / 2")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("cut value")
    ax.set_title("Max-Cut Value vs n")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs, random_cuts = _series_for(maxcut_ratio_rows, "metric", "ratio", "n", "random_cut")
    if xs:
        buckets = {}
        for x, y in zip(xs, random_cuts):
            buckets.setdefault(x, []).append(y)
        distinct_xs = sorted(buckets)
        variances = []
        for x in distinct_xs:
            vals = buckets[x]
            mean = sum(vals) / len(vals)
            variances.append(sum((v - mean) ** 2 for v in vals) / len(vals))
        ax.bar([str(x) for x in distinct_xs], variances, color="tab:purple", alpha=0.7)
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("variance of random cut value")
    ax.set_title("Random Max-Cut, Variance Across Trials")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    xs, ys = _series_for(maxcut_runtime_rows, "variant", "random", "n", "time_us")
    if xs:
        ax.plot(xs, ys, "o-", color="tab:purple", label="random assignment")
    xs, ys = _series_for(maxcut_runtime_rows, "variant", "local_search", "n", "time_us")
    if xs:
        ax.plot(xs, ys, "^-", color="tab:brown", label="local search")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("time (microseconds)")
    ax.set_title("Max-Cut Runtime, Random vs Local Search")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
