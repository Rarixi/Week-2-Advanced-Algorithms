"""
week9_demo.py
----------------
A small runnable walkthrough of every part of week 9, meant to be read top
to bottom. Builds one small instance of each problem, runs the
approximation algorithm and its exact counterpart, and prints the
comparison, so you can see every algorithm actually working without
digging through the test suite or the full benchmark.

Run with:
    python3 examples/week9_demo.py
"""

import os
import random
import sys

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
    approximation_ratio,
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


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_vertex_cover():
    section("Part 1a: Vertex Cover, Greedy 2-Approximation")
    rng = random.Random(42)
    vertices, edges = connected_random_graph(9, 0.3, rng=rng)
    print(f"graph: {len(vertices)} vertices, {len(edges)} edges")

    approx_cover = vertex_cover_greedy_approx(vertices, edges)
    exact_size, exact_cover = exact_minimum_vertex_cover(vertices, edges)

    print(f"greedy cover:  {sorted(approx_cover)}  (size {len(approx_cover)})")
    print(f"exact cover:   {sorted(exact_cover)}  (size {exact_size})")
    print(f"ratio: {len(approx_cover) / exact_size:.3f}  (bound: 2.0)")


def demo_facility_location():
    section("Part 1b: Facility Location, Greedy Heuristic")
    rng = random.Random(7)
    costs, distances = random_facility_location_instance(5, 8, rng=rng)

    open_f, greedy_cost = facility_location_greedy(costs, distances)
    exact_open, exact_cost = exact_facility_location_bruteforce(costs, distances)

    print(f"facilities opened by greedy: {sorted(open_f)}, cost {greedy_cost:.2f}")
    print(f"facilities opened by exact:  {sorted(exact_open)}, cost {exact_cost:.2f}")
    print(f"ratio: {greedy_cost / exact_cost:.3f}")


def demo_set_cover():
    section("Part 2: Set Cover, LP Relaxation and Rounding")
    rng = random.Random(1)
    universe, sets, costs = random_set_cover_instance(10, 9, rng=rng)
    print(f"universe size {len(universe)}, {len(sets)} sets available")

    x, lp_cost = solve_set_cover_lp_relaxation(universe, sets, costs)
    print(f"LP relaxation cost: {lp_cost:.3f}  (a lower bound on the true optimum)")

    det_chosen = deterministic_rounding(universe, sets, x)
    det_cost = solution_cost(det_chosen, costs)
    print(f"deterministic rounding: sets {sorted(det_chosen)}, cost {det_cost:.3f}")

    rand_chosen = randomized_rounding(universe, sets, costs, x, rng)
    rand_cost = solution_cost(rand_chosen, costs)
    print(f"randomized rounding:    sets {sorted(rand_chosen)}, cost {rand_cost:.3f}")

    exact_chosen, exact_cost = exact_set_cover_bruteforce(universe, sets, costs)
    print(f"exact optimum:          sets {sorted(exact_chosen)}, cost {exact_cost:.3f}")


def demo_tsp():
    section("Part 3: Metric TSP, MST Doubling 2-Approximation")
    rng = random.Random(3)
    points = random_points(8, rng=rng)

    tour = tsp_mst_doubling_approx(points)
    approx_length = tour_length(tour, points)

    _opt_tour, opt_length = tsp_exact_bruteforce(points)

    print(f"mst doubling tour: {tour}, length {approx_length:.2f}")
    print(f"exact optimal tour length: {opt_length:.2f}")
    print(f"ratio: {approximation_ratio(approx_length, opt_length):.3f}  (bound: 2.0)")


def demo_maxcut():
    section("Part 4: Randomized Max-Cut")
    rng = random.Random(11)
    vertices, edges = connected_random_graph(9, 0.35, rng=rng)
    adjacency = build_adjacency(vertices, edges)
    print(f"graph: {len(vertices)} vertices, {len(edges)} edges")

    assignment, random_value = random_cut(vertices, edges, rng)
    expected = expected_cut_value(len(edges))
    print(f"one random cut: {random_value}  (theoretical expectation: {expected:.2f})")

    _improved, local_value = local_search_maxcut(vertices, edges, assignment, adjacency)
    print(f"after local search: {local_value}")

    optimal = exact_maxcut_bruteforce(vertices, edges)
    print(f"true optimal cut: {optimal}")
    print(f"random ratio: {random_value / optimal:.3f}, local search ratio: {local_value / optimal:.3f}  (bound: 0.5)")


def main():
    demo_vertex_cover()
    demo_facility_location()
    demo_set_cover()
    demo_tsp()
    demo_maxcut()
    print()
    print("Full benchmarks with plots live in benchmarks/week9_approx_benchmark.py")


if __name__ == "__main__":
    main()
