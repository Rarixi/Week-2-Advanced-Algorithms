"""
test_reduction_demo.py
-------------------------
Tests for the 3-SAT to Vertex Cover reduction. Checks the structural shape
of the constructed graph (right number of vertices and edges for a given
formula) and, more importantly, that the reduction's actual claim holds,
a target-size vertex cover exists exactly when the original formula is
satisfiable, verified independently via the sat solver and an exact vertex
cover solver, not just trusted.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.complexity.reduction_demo import reduce_3sat_to_vertex_cover, validate_reduction
from src.complexity.satisfiability_solver import solve_sat, is_assignment_valid


class TestSatisfiabilitySolver:
    # satisfiability_solver.py has no dedicated test file of its own in this
    # project's structure, it's exercised indirectly by TestReductionCorrectness
    # above too, but these check it directly before the reduction ever gets
    # involved.

    def test_finds_a_valid_assignment_when_satisfiable(self):
        clauses = [[1, 2, 3], [-1, 2, 3]]
        satisfiable, assignment = solve_sat(clauses, 3)
        assert satisfiable is True
        assert is_assignment_valid(clauses, assignment)

    def test_reports_unsatisfiable_correctly(self):
        clauses = [[1, 1, 1], [-1, -1, -1]]
        satisfiable, assignment = solve_sat(clauses, 1)
        assert satisfiable is False
        assert assignment is None

    def test_stats_report_fewer_nodes_than_full_2_to_the_n_tree(self):
        # Pruning dead clauses early should mean the search doesn't always
        # have to visit the full 2^n assignment tree.
        clauses = [[1, 2, 3], [-1, -2, -3]]
        satisfiable, assignment, stats = solve_sat(clauses, 6, collect_stats=True)
        assert stats["nodes_visited"] < 2 ** 6


class TestReductionStructure:
    def test_vertex_and_edge_counts_match_formula_size(self):
        clauses = [[1, 2, 3], [-1, 2, -3]]
        num_vars = 3
        result = reduce_3sat_to_vertex_cover(clauses, num_vars, verbose=False)

        # 2 vertices per variable, 3 per clause
        assert len(result["vertices"]) == 2 * num_vars + 3 * len(clauses)
        # 1 edge per variable gadget, 3 triangle edges + 3 literal edges per clause
        assert len(result["edges"]) == num_vars + len(clauses) * (3 + 3)

    def test_target_k_formula(self):
        clauses = [[1, 2, 3], [-1, 2, -3], [1, -2, 3]]
        result = reduce_3sat_to_vertex_cover(clauses, 3, verbose=False)
        assert result["target_k"] == 3 + 2 * 3

    def test_steps_are_logged(self):
        clauses = [[1, 2, 3]]
        result = reduce_3sat_to_vertex_cover(clauses, 3, verbose=False)
        assert len(result["steps"]) > 0
        assert any("Step 1" in step for step in result["steps"])
        assert any("Step 2" in step for step in result["steps"])
        assert any("Step 3" in step for step in result["steps"])


class TestReductionCorrectness:
    def test_satisfiable_formula_has_cover_of_target_size(self):
        # x1=True, x2=True satisfies both clauses regardless of x3
        clauses = [[1, 2, 3], [-1, 2, 3]]
        result = validate_reduction(clauses, 3, verbose=False)
        assert result["satisfiable"] is True
        assert result["cover_of_target_size_exists"] is True
        assert result["min_cover_size"] == result["target_k"]
        assert result["reduction_correct_for_this_instance"] is True

    def test_unsatisfiable_formula_has_no_cover_of_target_size(self):
        # (x1) AND (NOT x1), using repeated literals to fake a 1-variable
        # clause within the 3-literal clause format, is unsatisfiable.
        clauses = [[1, 1, 1], [-1, -1, -1]]
        result = validate_reduction(clauses, 1, verbose=False)
        assert result["satisfiable"] is False
        assert result["cover_of_target_size_exists"] is False
        assert result["min_cover_size"] > result["target_k"]
        assert result["reduction_correct_for_this_instance"] is True

    def test_reduction_correct_across_several_random_instances(self, rng):
        from src.utils.graph_utils import random_cnf

        for _ in range(8):
            num_vars = rng.randint(2, 4)
            num_clauses = rng.randint(1, 3)
            clauses = random_cnf(num_vars, num_clauses, rng=rng)
            result = validate_reduction(clauses, num_vars, verbose=False)
            assert result["reduction_correct_for_this_instance"] is True
