"""
week8_demo.py
----------------
A small runnable walkthrough of every part of week 8, meant to be read top
to bottom. Doesn't replace the benchmark or the tests, just gives a quick
look at each algorithm actually running on one small example apiece.

Run with:
    python3 examples/week8_demo.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.complexity.reduction_demo import reduce_3sat_to_vertex_cover, validate_reduction
from src.complexity.satisfiability_solver import solve_sat, is_assignment_valid
from src.complexity.subset_sum_backtracking import subset_sum_backtracking, subset_sum_greedy_heuristic
from src.complexity.vertex_cover_approx import vertex_cover_greedy_approx, exact_minimum_vertex_cover
from src.complexity.traveling_salesman_approx import (
    tsp_nearest_neighbor, tsp_exact_bruteforce, approximation_ratio,
)
from src.utils.graph_utils import connected_random_graph, random_points, tour_length


def section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def demo_reduction():
    section("Part 1, 3-SAT to Vertex Cover Reduction")
    clauses = [[1, 2, 3], [-1, 2, 3]]
    num_vars = 3
    print(f"formula: {clauses} over {num_vars} variables")
    print()
    reduce_3sat_to_vertex_cover(clauses, num_vars, verbose=True)
    print()
    result = validate_reduction(clauses, num_vars, verbose=False)
    print(f"independent check, satisfiable={result['satisfiable']}, "
          f"cover of target size exists={result['cover_of_target_size_exists']}, "
          f"reduction correct for this instance={result['reduction_correct_for_this_instance']}")


def demo_sat_and_subset_sum():
    section("Part 2, SAT Solver and Subset Sum")
    clauses = [[1, 2, -3], [-1, 2, 3], [1, -2, 3]]
    satisfiable, assignment = solve_sat(clauses, 3)
    print(f"sat solver on {clauses}, satisfiable={satisfiable}, assignment={assignment}")
    if satisfiable:
        print(f"  checks out: {is_assignment_valid(clauses, assignment)}")

    numbers = [3, 34, 4, 12, 5, 2]
    target = 9
    found, subset = subset_sum_backtracking(numbers, target)
    print(f"\nsubset sum backtracking on {numbers}, target={target}, found={found}, subset={subset}")

    chosen, achieved_sum, exact = subset_sum_greedy_heuristic(numbers, target)
    print(f"subset sum greedy heuristic, chosen={chosen}, achieved_sum={achieved_sum}, exact={exact}")


def demo_approximations():
    section("Part 3, Vertex Cover and TSP Approximations")
    rng = random.Random(42)

    vertices, edges = connected_random_graph(10, 0.3, rng=rng)
    approx_cover = vertex_cover_greedy_approx(vertices, edges)
    exact_size, exact_cover = exact_minimum_vertex_cover(vertices, edges)
    print(f"vertex cover on a random 10-vertex graph with {len(edges)} edges")
    print(f"  greedy approx cover size {len(approx_cover)}, true optimal size {exact_size}, "
          f"ratio {len(approx_cover) / exact_size:.2f} (guaranteed at most 2.0)")

    points = random_points(8, rng=rng)
    nn_tour = tsp_nearest_neighbor(points)
    nn_length = tour_length(nn_tour, points)
    opt_tour, opt_length = tsp_exact_bruteforce(points)
    print(f"\ntsp on 8 random cities")
    print(f"  nearest neighbor tour length {nn_length:.1f}, true optimal tour length {opt_length:.1f}, "
          f"ratio {approximation_ratio(nn_length, opt_length):.3f}")


if __name__ == "__main__":
    demo_reduction()
    demo_sat_and_subset_sum()
    demo_approximations()
    print()
