"""
reduction_demo.py
--------------------
Part 1 of week 8. A polynomial time reduction from 3-SAT to Vertex Cover,
the classic textbook construction, built step by step with every step
logged so the mapping from one problem's instance to the other is visible,
not just the end result.

The construction, given a 3-cnf formula over variables 1..n with clauses
c_1..c_m
    Variable gadget, for each variable i, two vertices x{i} and nx{i} with
    an edge between them. Any vertex cover has to include at least one of
    the two, and this reduction reads whichever one gets picked as the
    truth value assigned to that variable.
    Clause gadget, for each clause j = (l1 OR l2 OR l3), three vertices
    forming a triangle (so any cover needs at least 2 of the 3) plus one
    edge from each triangle vertex to the literal vertex it represents.
    Target k = n + 2m.

The claim this reduction makes is that the formula is satisfiable if and
only if the resulting graph has a vertex cover of size exactly k. That
claim is what validate_reduction below actually checks, independently, by
running the sat solver and an exact (brute force) vertex cover solver on
the same instance and confirming they agree.
"""

import itertools

from src.complexity.satisfiability_solver import solve_sat
from src.complexity.vertex_cover_approx import exact_minimum_vertex_cover


def _literal_to_vertex(literal):
    var = abs(literal)
    return f"x{var}" if literal > 0 else f"nx{var}"


def _format_literal(literal):
    var = abs(literal)
    return f"x{var}" if literal > 0 else f"NOT x{var}"


def _format_clause(clause):
    return "(" + " OR ".join(_format_literal(l) for l in clause) + ")"


def reduce_3sat_to_vertex_cover(clauses, num_vars, verbose=True):
    """
    Build the vertex cover instance equivalent to the given 3-cnf formula.
    Returns a dict with "vertices", "edges", "target_k", and "steps" (the
    log of every explanation string produced along the way, in order,
    whether or not verbose printed them).
    """
    steps = []

    def log(message):
        steps.append(message)
        if verbose:
            print(message)

    vertices = []
    edges = []

    log("Step 1, variable gadgets.")
    for i in range(1, num_vars + 1):
        pos_label, neg_label = f"x{i}", f"nx{i}"
        vertices.extend([pos_label, neg_label])
        edges.append((pos_label, neg_label))
        log(
            f"  Variable {i}, added vertices {pos_label} and {neg_label} with an "
            f"edge between them. Any vertex cover has to include at least one of "
            f"the two, and whichever one it picks is read as the truth value this "
            f"reduction assigns to variable {i}."
        )

    log("Step 2, clause gadgets (one triangle per clause).")
    for j, clause in enumerate(clauses, start=1):
        clause_vertices = [f"c{j}_{k}" for k in range(1, len(clause) + 1)]
        vertices.extend(clause_vertices)

        for a, b in itertools.combinations(clause_vertices, 2):
            edges.append((a, b))
        log(
            f"  Clause {j} = {_format_clause(clause)}, added triangle vertices "
            f"{clause_vertices} with an edge between every pair. A triangle always "
            f"needs at least 2 of its 3 vertices in any vertex cover, no matter what."
        )

        for k, literal in enumerate(clause, start=1):
            literal_vertex = _literal_to_vertex(literal)
            clause_vertex = clause_vertices[k - 1]
            edges.append((clause_vertex, literal_vertex))
            log(
                f"    Connected {clause_vertex} to {literal_vertex}, the vertex "
                f"representing literal {_format_literal(literal)}. If {literal_vertex} "
                f"is not in the cover, this edge can only be covered by {clause_vertex} "
                f"itself, which forces the triangle to use its 3rd vertex instead of "
                f"just 2, exactly the extra cost that shows up when a clause isn't "
                f"actually satisfied by this literal."
            )

    target_k = num_vars + 2 * len(clauses)
    log(
        f"Step 3, target vertex cover size is n + 2m = {num_vars} + 2 times "
        f"{len(clauses)} = {target_k}."
    )
    log(
        f"  That's one guaranteed vertex per variable pair (n of them) plus two "
        f"guaranteed vertices per clause triangle (2m of them). The reduction's claim "
        f"is that a cover of exactly this size, covering the triangles and variable "
        f"pairs with nothing extra to spare, can also cover every clause to literal "
        f"edge, exactly when the formula is satisfiable."
    )

    return {
        "vertices": vertices,
        "edges": edges,
        "target_k": target_k,
        "steps": steps,
    }


def validate_reduction(clauses, num_vars, verbose=False):
    """
    Independently checks the reduction's correctness on one instance, by
    running the sat solver on the original formula and an exact vertex
    cover solver on the constructed graph, then confirming "a cover of size
    target_k exists" agrees with "the formula is satisfiable". Returns a
    dict with both answers plus a "reduction_correct_for_this_instance" flag.

    Only practical for small instances, since it calls the brute force
    exact_minimum_vertex_cover under the hood.
    """
    reduction = reduce_3sat_to_vertex_cover(clauses, num_vars, verbose=verbose)
    satisfiable, _assignment = solve_sat(clauses, num_vars)

    # n + 3m is a provable upper bound on the true minimum cover size here
    # (take one vertex from each variable pair, plus all 3 vertices of every
    # clause triangle, that always covers the whole graph), so searching up
    # to it is guaranteed to find the real minimum without having to search
    # the entire vertex set.
    safe_upper_bound = num_vars + 3 * len(clauses)
    min_cover_size, min_cover = exact_minimum_vertex_cover(
        reduction["vertices"], reduction["edges"], max_size=safe_upper_bound
    )

    cover_of_target_size_exists = min_cover_size <= reduction["target_k"]

    return {
        "satisfiable": satisfiable,
        "target_k": reduction["target_k"],
        "min_cover_size": min_cover_size,
        "min_cover": min_cover,
        "cover_of_target_size_exists": cover_of_target_size_exists,
        "reduction_correct_for_this_instance": cover_of_target_size_exists == satisfiable,
    }
