"""
satisfiability_solver.py
--------------------------
Part 2 of week 8, a sat solver for small cnf instances using plain dfs
backtracking over variable assignments. No unit propagation, no clause
learning, nothing clever, this is meant to show the brute force exponential
side of np completeness, not to be a real sat solver. Expected complexity is
O(2^n) in the number of variables, since in the worst case (an unsatisfiable
formula, or one where the solver gets unlucky with variable order) it has to
explore the whole assignment tree.

A cnf formula here is a list of clauses, each clause a list of nonzero
integers, where k means variable k is true and -k means variable k is false.
A clause is satisfied if at least one of its literals is true under the
current assignment.
"""


def _clause_satisfied(clause, assignment):
    for literal in clause:
        var = abs(literal)
        if var not in assignment:
            continue
        value = assignment[var]
        if (literal > 0 and value) or (literal < 0 and not value):
            return True
    return False


def _clause_is_dead(clause, assignment):
    """True if every literal in the clause is already assigned false, so
    there is no way left to satisfy it no matter what unassigned variables
    end up being. Lets the search prune a branch early instead of assigning
    every remaining variable before finding out the clause failed."""
    for literal in clause:
        var = abs(literal)
        if var not in assignment:
            return False
        value = assignment[var]
        if (literal > 0 and value) or (literal < 0 and not value):
            return False
    return True


def solve_sat(clauses, num_vars, collect_stats=False):
    """
    DFS backtracking sat solver. Returns (satisfiable, assignment) where
    assignment is a dict of var -> bool on success, or (False, None) if no
    satisfying assignment exists. If collect_stats is True, returns a third
    element, a dict with "nodes_visited", the number of partial assignments
    the search actually explored, useful for showing how much early pruning
    saves versus the full 2^n tree.
    """
    stats = {"nodes_visited": 0}
    assignment = {}

    def backtrack(next_var):
        stats["nodes_visited"] += 1

        for clause in clauses:
            if _clause_is_dead(clause, assignment):
                return False

        if next_var > num_vars:
            return all(_clause_satisfied(c, assignment) for c in clauses)

        for value in (True, False):
            assignment[next_var] = value
            if backtrack(next_var + 1):
                return True
            del assignment[next_var]

        return False

    satisfied = backtrack(1)
    result_assignment = dict(assignment) if satisfied else None

    if collect_stats:
        return satisfied, result_assignment, stats
    return satisfied, result_assignment


def is_assignment_valid(clauses, assignment):
    """Check a full assignment (dict var -> bool) actually satisfies every
    clause, used to double check the solver's own answer rather than just
    trusting it."""
    return all(_clause_satisfied(clause, assignment) for clause in clauses)
