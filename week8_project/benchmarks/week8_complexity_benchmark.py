"""
week8_complexity_benchmark.py
--------------------------------
Benchmarks and visualizes the week 8 algorithms:
    - Subset Sum, backtracking (exponential) vs a greedy heuristic (O(n log n))
    - SAT, dfs backtracking, runtime growth as the number of variables increases
    - Vertex Cover, the greedy 2-approximation vs an exact brute force solver
      on small instances, for both runtime and approximation ratio
    - TSP, the nearest neighbor heuristic vs an exact brute force solver on
      small instances, same two angles, runtime and approximation ratio

Each algorithm's benchmark function builds and returns its own list of row
dicts, tagged consistently enough that the plotting functions can pull out
just the series they need by field. export_csv() combines everything into
one tidy comparison_table.csv at the end, tagged with an "algorithm" column,
purely for a single combined file to hand in.

Run with:
    python3 week8_complexity_benchmark.py

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

from src.complexity.subset_sum_backtracking import subset_sum_backtracking, subset_sum_greedy_heuristic
from src.complexity.satisfiability_solver import solve_sat
from src.complexity.vertex_cover_approx import vertex_cover_greedy_approx, exact_minimum_vertex_cover
from src.complexity.traveling_salesman_approx import (
    tsp_nearest_neighbor, tsp_exact_bruteforce, approximation_ratio as tsp_approximation_ratio,
)
from src.utils.graph_utils import (
    connected_random_graph, random_points, random_cnf, random_subset_sum_instance, tour_length,
)
from src.utils.timer import time_execution
from src.utils.visualization import save_runtime_growth_figure, save_approximation_quality_figure


SUBSET_SUM_SIZES = [10, 14, 18, 20, 22]
SUBSET_SUM_REPEATS = 3

SAT_NUM_VARS_SIZES = [10, 14, 18, 20, 22, 24, 26]
SAT_TRIALS_PER_SIZE = 8
SAT_CLAUSE_TO_VAR_RATIO = 4.3  # near the empirically "hardest" region for random 3-sat

VERTEX_COVER_RATIO_SIZES = [6, 8, 10, 12, 14, 16, 18]
VERTEX_COVER_RATIO_TRIALS = {6: 5, 8: 5, 10: 5, 12: 5, 14: 3, 16: 3, 18: 2}
VERTEX_COVER_RATIO_EDGE_PROBABILITY = 0.25
VERTEX_COVER_RUNTIME_SIZES = [50, 100, 200, 400, 800]
VERTEX_COVER_RUNTIME_EDGE_PROBABILITY = 0.05
VERTEX_COVER_RUNTIME_REPEATS = 3

TSP_RATIO_SIZES = [5, 6, 7, 8, 9, 10]
TSP_RATIO_TRIALS = 5
TSP_RUNTIME_SIZES = [50, 100, 200, 500, 1000]
TSP_RUNTIME_REPEATS = 3


def run_subset_sum_benchmarks(sizes=SUBSET_SUM_SIZES, repeats=SUBSET_SUM_REPEATS, seed=1, verbose=True):
    """
    Benchmarks the exact backtracking solver against the greedy heuristic on
    the same instances. Uses the genuine worst case for backtracking, a
    target one more than the total sum, so there is no solution to stumble
    onto early and nothing to prune on, forcing the full 2^n tree every
    time, which is what actually makes the exponential growth curve clean
    instead of noisy.
    """
    rng = random.Random(seed)
    rows = []

    for n in sizes:
        numbers = [rng.randint(100, 500) for _ in range(n)]
        target = sum(numbers) + 1  # just out of reach, forces full exploration

        bt_mean, bt_var = time_execution(
            lambda nums=numbers, t=target: subset_sum_backtracking(nums, t), repeats=repeats
        )
        _found, _subset, bt_stats = subset_sum_backtracking(numbers, target, collect_stats=True)

        heuristic_mean, heuristic_var = time_execution(
            lambda nums=numbers, t=target: subset_sum_greedy_heuristic(nums, t), repeats=repeats
        )
        _chosen, achieved_sum, exact = subset_sum_greedy_heuristic(numbers, target)

        rows.append({
            "algorithm": "subset_sum", "variant": "backtracking", "n": n,
            "time_us": round(bt_mean * 1e6, 3), "time_variance_us2": round(bt_var * 1e12, 3),
            "nodes_visited": bt_stats["nodes_visited"], "found_solution": False,
        })
        rows.append({
            "algorithm": "subset_sum", "variant": "heuristic", "n": n,
            "time_us": round(heuristic_mean * 1e6, 3), "time_variance_us2": round(heuristic_var * 1e12, 3),
            "achieved_sum": achieved_sum, "target": target, "exact_match": exact,
        })

        if verbose:
            print(f"  subset sum n={n}, backtracking {bt_mean*1e6:.1f}us ({bt_stats['nodes_visited']} nodes), heuristic {heuristic_mean*1e6:.1f}us")

    return rows


def run_sat_benchmarks(
    num_vars_sizes=SAT_NUM_VARS_SIZES, trials_per_size=SAT_TRIALS_PER_SIZE,
    clause_ratio=SAT_CLAUSE_TO_VAR_RATIO, seed=2, verbose=True,
):
    """
    Benchmarks the dfs backtracking sat solver as the number of variables
    grows, holding the clauses-to-variables ratio fixed near 4.3, the
    region random 3-sat instances are empirically hardest in. Averages
    several random instances per size, since any single random instance can
    get lucky (or unlucky) and finish in a handful of steps regardless of n,
    which would make the growth curve look noisy rather than exponential.
    """
    rng = random.Random(seed)
    rows = []

    for num_vars in num_vars_sizes:
        num_clauses = round(clause_ratio * num_vars)
        times = []
        node_counts = []
        satisfiable_count = 0

        for _ in range(trials_per_size):
            clauses = random_cnf(num_vars, num_clauses, rng=rng)
            start = time.perf_counter()
            satisfiable, _assignment, stats = solve_sat(clauses, num_vars, collect_stats=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            node_counts.append(stats["nodes_visited"])
            if satisfiable:
                satisfiable_count += 1

        mean_time = sum(times) / len(times)
        variance_time = sum((t - mean_time) ** 2 for t in times) / len(times)
        mean_nodes = sum(node_counts) / len(node_counts)

        rows.append({
            "algorithm": "sat", "variant": "backtracking", "num_vars": num_vars,
            "num_clauses": num_clauses, "time_us": round(mean_time * 1e6, 3),
            "time_variance_us2": round(variance_time * 1e12, 3),
            "mean_nodes_visited": round(mean_nodes, 1),
            "satisfiable_fraction": round(satisfiable_count / trials_per_size, 3),
        })

        if verbose:
            print(f"  sat num_vars={num_vars}, mean {mean_time*1e6:.1f}us, mean nodes {mean_nodes:.0f}, satisfiable {satisfiable_count}/{trials_per_size}")

    return rows


def run_vertex_cover_benchmarks(
    ratio_sizes=VERTEX_COVER_RATIO_SIZES, ratio_trials=VERTEX_COVER_RATIO_TRIALS,
    ratio_edge_probability=VERTEX_COVER_RATIO_EDGE_PROBABILITY,
    runtime_sizes=VERTEX_COVER_RUNTIME_SIZES, runtime_edge_probability=VERTEX_COVER_RUNTIME_EDGE_PROBABILITY,
    runtime_repeats=VERTEX_COVER_RUNTIME_REPEATS, seed=3, verbose=True,
):
    """
    Two separate sweeps. The ratio sweep stays small (exact_minimum_vertex_cover
    is brute force) and measures how close the greedy 2-approximation's cover
    size actually comes to the true optimum. The runtime sweep goes much
    bigger, since the approximation algorithm alone is O(E) and has nothing
    exponential to hold it back, showing what polynomial growth actually
    looks like next to the exponential curves from subset sum and sat.
    """
    rng = random.Random(seed)
    rows = []

    for n in ratio_sizes:
        trials = ratio_trials[n] if isinstance(ratio_trials, dict) else ratio_trials
        for _ in range(trials):
            vertices, edges = connected_random_graph(n, ratio_edge_probability, rng=rng)
            if not edges:
                continue

            approx_mean, _approx_var = time_execution(
                lambda v=vertices, e=edges: vertex_cover_greedy_approx(v, e), repeats=3
            )
            approx_cover = vertex_cover_greedy_approx(vertices, edges)

            exact_start = time.perf_counter()
            exact_size, _exact_cover = exact_minimum_vertex_cover(vertices, edges)
            exact_elapsed = time.perf_counter() - exact_start

            ratio = len(approx_cover) / exact_size if exact_size else 1.0
            rows.append({
                "algorithm": "vertex_cover", "metric": "ratio", "n": n,
                "approx_size": len(approx_cover), "optimal_size": exact_size, "ratio": round(ratio, 4),
                "approx_time_us": round(approx_mean * 1e6, 3), "exact_time_us": round(exact_elapsed * 1e6, 3),
            })

        if verbose:
            print(f"  vertex cover ratio n={n}, done")

    for n in runtime_sizes:
        vertices, edges = connected_random_graph(n, runtime_edge_probability, rng=rng)
        mean_time, variance_time = time_execution(
            lambda v=vertices, e=edges: vertex_cover_greedy_approx(v, e), repeats=runtime_repeats
        )
        rows.append({
            "algorithm": "vertex_cover", "variant": "approx", "n": n, "num_edges": len(edges),
            "time_us": round(mean_time * 1e6, 3), "time_variance_us2": round(variance_time * 1e12, 3),
        })
        if verbose:
            print(f"  vertex cover runtime n={n}, {mean_time*1e6:.1f}us, {len(edges)} edges")

    return rows


def run_tsp_benchmarks(
    ratio_sizes=TSP_RATIO_SIZES, ratio_trials=TSP_RATIO_TRIALS,
    runtime_sizes=TSP_RUNTIME_SIZES, runtime_repeats=TSP_RUNTIME_REPEATS, seed=4, verbose=True,
):
    """
    Same two-sweep structure as vertex cover. The ratio sweep stays small
    since the exact solver is (n-1)! brute force, and measures how much
    longer the nearest neighbor heuristic's tour is than the true optimal.
    The runtime sweep goes much bigger, since nearest neighbor alone is
    O(n^2) and has no exponential blowup waiting for it.
    """
    rng = random.Random(seed)
    rows = []

    for n in ratio_sizes:
        for _ in range(ratio_trials):
            points = random_points(n, rng=rng)

            nn_mean, _nn_var = time_execution(lambda p=points: tsp_nearest_neighbor(p), repeats=3)
            nn_tour = tsp_nearest_neighbor(points)
            nn_length = tour_length(nn_tour, points)

            exact_start = time.perf_counter()
            _opt_tour, opt_length = tsp_exact_bruteforce(points)
            exact_elapsed = time.perf_counter() - exact_start

            ratio = tsp_approximation_ratio(nn_length, opt_length)
            rows.append({
                "algorithm": "tsp", "metric": "ratio", "n": n,
                "nn_length": round(nn_length, 3), "optimal_length": round(opt_length, 3),
                "ratio": round(ratio, 4),
                "approx_time_us": round(nn_mean * 1e6, 3), "exact_time_us": round(exact_elapsed * 1e6, 3),
            })

        if verbose:
            print(f"  tsp ratio n={n}, done")

    for n in runtime_sizes:
        points = random_points(n, rng=rng)
        mean_time, variance_time = time_execution(
            lambda p=points: tsp_nearest_neighbor(p), repeats=runtime_repeats
        )
        rows.append({
            "algorithm": "tsp", "variant": "nearest_neighbor", "n": n,
            "time_us": round(mean_time * 1e6, 3), "time_variance_us2": round(variance_time * 1e12, 3),
        })
        if verbose:
            print(f"  tsp runtime n={n}, {mean_time*1e6:.1f}us")

    return rows


def export_csv(rows_by_algorithm, out_path):
    """Combine every algorithm's rows into one csv, tagged with an
    "algorithm" column, covering the union of every field used by any row
    so the file stays readable as one flat table."""
    all_fields = ["algorithm"]
    for rows in rows_by_algorithm.values():
        for row in rows:
            for key in row.keys():
                if key not in all_fields:
                    all_fields.append(key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for algorithm, rows in rows_by_algorithm.items():
            for row in rows:
                full_row = {"algorithm": algorithm}
                full_row.update(row)
                writer.writerow(full_row)


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Running subset sum benchmarks...")
    subset_sum_rows = run_subset_sum_benchmarks()

    print("Running sat solver benchmarks...")
    sat_rows = run_sat_benchmarks()

    print("Running vertex cover benchmarks...")
    vertex_cover_rows = run_vertex_cover_benchmarks()

    print("Running tsp benchmarks...")
    tsp_rows = run_tsp_benchmarks()

    print("Saving figures...")
    save_runtime_growth_figure(
        subset_sum_rows, sat_rows, vertex_cover_rows, tsp_rows,
        os.path.join(results_dir, "runtime_growth.png"),
    )
    save_approximation_quality_figure(
        vertex_cover_rows, tsp_rows,
        os.path.join(results_dir, "approximation_quality.png"),
    )

    print("Exporting comparison_table.csv...")
    export_csv(
        {
            "subset_sum": subset_sum_rows,
            "sat": sat_rows,
            "vertex_cover": vertex_cover_rows,
            "tsp": tsp_rows,
        },
        os.path.join(results_dir, "comparison_table.csv"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
