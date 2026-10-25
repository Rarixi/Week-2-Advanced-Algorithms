"""
week9_approx_benchmark.py
----------------------------
Part 5 of week 9. Benchmarks all four approximation algorithms (Vertex
Cover 2-approx, Facility Location greedy, Set Cover LP rounding, Metric
TSP MST doubling, randomized Max-Cut), comparing approximation ratio,
runtime, scalability, and variance for the randomized pieces, exporting
plots and a comparison table to benchmarks/results/.

Two kinds of sweep for every algorithm, mirroring the week 7 and week 8
pattern:

ratio sweeps
    small instances (n small enough that the exact / brute force solver
    for that problem is still fast), run several times across different n
    and different random seeds, measuring approx cost against the true
    optimal cost so a real approximation ratio can be reported, not just
    cited from the proof.

runtime sweeps
    larger instances where the exact solver is no longer run at all (it
    would take forever), just timing the approximation algorithm itself
    across growing n to show it scales like a polynomial time algorithm
    should.

Set Cover and Max-Cut additionally repeat several randomized trials per
instance (randomized rounding, random_cut) specifically to measure mean
and variance, since a single trial of a randomized algorithm doesn't mean
much on its own.

Run with:
    python3 week9_approx_benchmark.py
"""

import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.approximation.vertex_cover_2approx import (
    vertex_cover_greedy_approx,
    exact_minimum_vertex_cover,
)
from src.approximation.facility_location_greedy import (
    facility_location_greedy,
    exact_facility_location_bruteforce,
)
from src.approximation.set_cover_lpround import (
    solve_set_cover_lp_relaxation,
    deterministic_rounding,
    randomized_rounding,
    exact_set_cover_bruteforce,
    solution_cost,
)
from src.approximation.tsp_metric_approx import (
    tsp_mst_doubling_approx,
    tsp_exact_bruteforce,
    approximation_ratio as tsp_approximation_ratio,
)
from src.approximation.maxcut_randomized import (
    random_cut,
    expected_cut_value,
    local_search_maxcut,
    exact_maxcut_bruteforce,
)
from src.utils.graph_utils import (
    connected_random_graph,
    build_adjacency,
    random_points,
    tour_length,
    random_facility_location_instance,
    random_set_cover_instance,
)
from src.utils.visualization import (
    save_approx_ratios_figure,
    save_cost_vs_optimal_figure,
    save_randomized_maxcut_results_figure,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

RATIO_SIZES = [5, 6, 7, 8, 9, 10]
TRIALS_PER_SIZE = 3
SET_COVER_RANDOM_TRIALS = 20
MAXCUT_RANDOM_TRIALS = 30
RUNTIME_SIZES = [10, 20, 40, 80, 160, 320]


# ---------------------------------------------------------------------------
# Vertex Cover
# ---------------------------------------------------------------------------

def benchmark_vertex_cover():
    ratio_rows = []
    rng = random.Random(100)
    for n in RATIO_SIZES:
        for _ in range(TRIALS_PER_SIZE):
            vertices, edges = connected_random_graph(n, rng.uniform(0.25, 0.5), rng=rng)
            if not edges:
                continue
            approx_cover = vertex_cover_greedy_approx(vertices, edges)
            exact_size, _exact_cover = exact_minimum_vertex_cover(vertices, edges)
            ratio_rows.append({
                "algorithm": "vertex_cover",
                "metric": "ratio",
                "n": n,
                "approx_size": len(approx_cover),
                "optimal_size": exact_size,
                "ratio": len(approx_cover) / exact_size if exact_size > 0 else 1.0,
            })

    runtime_rows = []
    rng = random.Random(101)
    for n in RUNTIME_SIZES:
        vertices, edges = connected_random_graph(n, 0.1, rng=rng)
        start = time.perf_counter()
        vertex_cover_greedy_approx(vertices, edges)
        elapsed = time.perf_counter() - start
        runtime_rows.append({
            "algorithm": "vertex_cover",
            "metric": "runtime",
            "n": n,
            "time_us": elapsed * 1e6,
        })

    return ratio_rows, runtime_rows


# ---------------------------------------------------------------------------
# Facility Location
# ---------------------------------------------------------------------------

def benchmark_facility_location():
    ratio_rows = []
    rng = random.Random(200)
    for n in [4, 5, 6, 7, 8]:
        for _ in range(TRIALS_PER_SIZE):
            costs, distances = random_facility_location_instance(
                rng.randint(3, 5), n, rng=rng
            )
            _open_f, greedy_cost = facility_location_greedy(costs, distances)
            _exact_open, exact_cost = exact_facility_location_bruteforce(costs, distances)
            ratio_rows.append({
                "algorithm": "facility_location",
                "metric": "ratio",
                "n": n,
                "greedy_cost": greedy_cost,
                "optimal_cost": exact_cost,
                "ratio": greedy_cost / exact_cost if exact_cost > 0 else 1.0,
            })

    runtime_rows = []
    rng = random.Random(201)
    for n in RUNTIME_SIZES:
        costs, distances = random_facility_location_instance(max(3, n // 5), n, rng=rng)
        start = time.perf_counter()
        facility_location_greedy(costs, distances)
        elapsed = time.perf_counter() - start
        runtime_rows.append({
            "algorithm": "facility_location",
            "metric": "runtime",
            "n": n,
            "time_us": elapsed * 1e6,
        })

    return ratio_rows, runtime_rows


# ---------------------------------------------------------------------------
# Set Cover
# ---------------------------------------------------------------------------

def benchmark_set_cover():
    ratio_rows = []
    rng = random.Random(300)
    for n in [6, 8, 10, 12, 14]:
        for _ in range(TRIALS_PER_SIZE):
            universe, sets, costs = random_set_cover_instance(n, n + 2, rng=rng)
            x, lp_cost = solve_set_cover_lp_relaxation(universe, sets, costs)
            _exact_chosen, exact_cost = exact_set_cover_bruteforce(universe, sets, costs)

            det_chosen = deterministic_rounding(universe, sets, x)
            det_cost = solution_cost(det_chosen, costs)
            ratio_rows.append({
                "algorithm": "set_cover",
                "metric": "ratio",
                "variant": "deterministic",
                "n": n,
                "lp_cost": lp_cost,
                "cost": det_cost,
                "optimal_cost": exact_cost,
                "ratio": det_cost / exact_cost if exact_cost > 0 else 1.0,
            })

            trial_costs = []
            for _ in range(SET_COVER_RANDOM_TRIALS):
                rand_chosen = randomized_rounding(universe, sets, costs, x, rng)
                trial_costs.append(solution_cost(rand_chosen, costs))
            mean_cost = sum(trial_costs) / len(trial_costs)
            cost_variance = sum((c - mean_cost) ** 2 for c in trial_costs) / len(trial_costs)
            mean_ratio = mean_cost / exact_cost if exact_cost > 0 else 1.0
            ratio_variance = cost_variance / (exact_cost ** 2) if exact_cost > 0 else 0.0
            ratio_rows.append({
                "algorithm": "set_cover",
                "metric": "ratio",
                "variant": "randomized",
                "n": n,
                "lp_cost": lp_cost,
                "mean_cost": mean_cost,
                "cost_variance": cost_variance,
                "optimal_cost": exact_cost,
                "mean_ratio": mean_ratio,
                "ratio_variance": ratio_variance,
            })

    runtime_rows = []
    rng = random.Random(301)
    for n in [20, 40, 80, 160, 320, 640]:
        universe, sets, costs = random_set_cover_instance(n, n + n // 4, rng=rng)
        start = time.perf_counter()
        x, _lp_cost = solve_set_cover_lp_relaxation(universe, sets, costs)
        deterministic_rounding(universe, sets, x)
        elapsed = time.perf_counter() - start
        runtime_rows.append({
            "algorithm": "set_cover",
            "metric": "runtime",
            "n": n,
            "time_us": elapsed * 1e6,
        })

    return ratio_rows, runtime_rows


# ---------------------------------------------------------------------------
# Metric TSP
# ---------------------------------------------------------------------------

def benchmark_tsp():
    ratio_rows = []
    rng = random.Random(400)
    for n in RATIO_SIZES:
        for _ in range(TRIALS_PER_SIZE):
            points = random_points(n, rng=rng)
            tour = tsp_mst_doubling_approx(points)
            approx_length = tour_length(tour, points)
            _opt_tour, opt_length = tsp_exact_bruteforce(points)
            ratio_rows.append({
                "algorithm": "tsp",
                "metric": "ratio",
                "n": n,
                "approx_length": approx_length,
                "optimal_length": opt_length,
                "ratio": tsp_approximation_ratio(approx_length, opt_length),
            })

    runtime_rows = []
    rng = random.Random(401)
    for n in RUNTIME_SIZES:
        points = random_points(n, rng=rng)
        start = time.perf_counter()
        tsp_mst_doubling_approx(points)
        elapsed = time.perf_counter() - start
        runtime_rows.append({
            "algorithm": "tsp",
            "metric": "runtime",
            "n": n,
            "time_us": elapsed * 1e6,
        })

    return ratio_rows, runtime_rows


# ---------------------------------------------------------------------------
# Max-Cut
# ---------------------------------------------------------------------------

def benchmark_maxcut():
    ratio_rows = []
    rng = random.Random(500)
    for n in [6, 7, 8, 9, 10]:
        for _ in range(TRIALS_PER_SIZE):
            vertices, edges = connected_random_graph(n, rng.uniform(0.3, 0.5), rng=rng)
            adjacency = build_adjacency(vertices, edges)
            optimal = exact_maxcut_bruteforce(vertices, edges)
            expected = expected_cut_value(len(edges))

            for _ in range(MAXCUT_RANDOM_TRIALS):
                assignment, random_value = random_cut(vertices, edges, rng)
                _improved, local_value = local_search_maxcut(vertices, edges, assignment, adjacency)
                ratio_rows.append({
                    "algorithm": "maxcut",
                    "metric": "ratio",
                    "n": n,
                    "random_cut": random_value,
                    "local_cut": local_value,
                    "optimal_cut": optimal,
                    "expected_cut": expected,
                    "random_ratio": random_value / optimal if optimal > 0 else 1.0,
                    "local_ratio": local_value / optimal if optimal > 0 else 1.0,
                })

    runtime_rows = []
    rng = random.Random(501)
    for n in RUNTIME_SIZES:
        vertices, edges = connected_random_graph(n, 0.1, rng=rng)
        adjacency = build_adjacency(vertices, edges)

        start = time.perf_counter()
        assignment, _cut = random_cut(vertices, edges, rng)
        elapsed = time.perf_counter() - start
        runtime_rows.append({
            "algorithm": "maxcut",
            "metric": "runtime",
            "variant": "random",
            "n": n,
            "time_us": elapsed * 1e6,
        })

        start = time.perf_counter()
        local_search_maxcut(vertices, edges, assignment, adjacency)
        elapsed = time.perf_counter() - start
        runtime_rows.append({
            "algorithm": "maxcut",
            "metric": "runtime",
            "variant": "local_search",
            "n": n,
            "time_us": elapsed * 1e6,
        })

    return ratio_rows, runtime_rows


# ---------------------------------------------------------------------------
# Comparison table + summary
# ---------------------------------------------------------------------------

def export_csv(all_rows, out_path):
    fieldnames = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def print_summary(vc_ratio, fl_ratio, sc_ratio, tsp_ratio, mc_ratio):
    def mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else float("nan")

    print("=" * 70)
    print("WEEK 9 APPROXIMATION ALGORITHMS - SUMMARY")
    print("=" * 70)

    vc_ratios = [r["ratio"] for r in vc_ratio]
    print(f"Vertex Cover        mean ratio = {mean(vc_ratios):.3f}  (bound: 2.0)")

    fl_ratios = [r["ratio"] for r in fl_ratio]
    print(f"Facility Location   mean ratio = {mean(fl_ratios):.3f}  (heuristic, no fixed bound)")

    det_ratios = [r["ratio"] for r in sc_ratio if r.get("variant") == "deterministic"]
    rand_ratios = [r["mean_ratio"] for r in sc_ratio if r.get("variant") == "randomized"]
    print(f"Set Cover (det.)    mean ratio = {mean(det_ratios):.3f}  (bound: f, frequency dependent)")
    print(f"Set Cover (rand.)   mean ratio = {mean(rand_ratios):.3f}  (target: O(log n))")

    tsp_ratios = [r["ratio"] for r in tsp_ratio]
    print(f"Metric TSP          mean ratio = {mean(tsp_ratios):.3f}  (bound: 2.0)")

    random_ratios = [r["random_ratio"] for r in mc_ratio]
    local_ratios = [r["local_ratio"] for r in mc_ratio]
    print(f"Max-Cut (random)    mean ratio = {mean(random_ratios):.3f}  (bound: 0.5 in expectation)")
    print(f"Max-Cut (local srch)mean ratio = {mean(local_ratios):.3f}  (bound: 0.5, local optimum)")
    print("=" * 70)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Benchmarking Vertex Cover...")
    vc_ratio, vc_runtime = benchmark_vertex_cover()

    print("Benchmarking Facility Location...")
    fl_ratio, fl_runtime = benchmark_facility_location()

    print("Benchmarking Set Cover...")
    sc_ratio, sc_runtime = benchmark_set_cover()

    print("Benchmarking Metric TSP...")
    tsp_ratio, tsp_runtime = benchmark_tsp()

    print("Benchmarking Max-Cut...")
    mc_ratio, mc_runtime = benchmark_maxcut()

    print("Saving figures...")
    save_approx_ratios_figure(
        vc_ratio, sc_ratio, tsp_ratio, mc_ratio,
        os.path.join(RESULTS_DIR, "approx_ratios.png"),
    )
    save_cost_vs_optimal_figure(
        fl_ratio, tsp_ratio,
        os.path.join(RESULTS_DIR, "cost_vs_optimal.png"),
    )
    save_randomized_maxcut_results_figure(
        mc_ratio, mc_runtime,
        os.path.join(RESULTS_DIR, "randomized_maxcut_results.png"),
    )

    print("Writing comparison table...")
    all_rows = (
        vc_ratio + vc_runtime
        + fl_ratio + fl_runtime
        + sc_ratio + sc_runtime
        + tsp_ratio + tsp_runtime
        + mc_ratio + mc_runtime
    )
    export_csv(all_rows, os.path.join(RESULTS_DIR, "comparison_table.csv"))

    print_summary(vc_ratio, fl_ratio, sc_ratio, tsp_ratio, mc_ratio)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
