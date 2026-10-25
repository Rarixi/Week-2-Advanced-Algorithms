"""
lp_solver.py
---------------
A thin, general purpose linear programming wrapper for week 9. Set Cover's
LP relaxation (in set_cover_lpround.py) is the only caller this week, but
this file stays generic, minimize c^T x subject to A_ub x <= b_ub,
A_eq x = b_eq, and per variable bounds, so it isn't tied to Set Cover's
specific constraint shape.

Uses scipy.optimize.linprog under the hood (the "python library if
allowed" option the assignment spec itself suggests), with HiGHS as the
solving method, scipy's modern default and a genuinely reliable simplex
or interior point solver, rather than hand rolling a simplex
implementation that would be much easier to get subtly wrong.
"""

from scipy.optimize import linprog


def solve_lp_minimize(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method="highs"):
    """
    Minimize c^T x subject to the given constraints and bounds. bounds is a
    list of (low, high) pairs per variable, same convention scipy uses,
    None for either side means unbounded in that direction.

    Returns (x, objective_value), where x is a list of floats, the
    optimal (possibly fractional) solution. Raises RuntimeError if the LP
    is infeasible, unbounded, or otherwise fails to solve, rather than
    silently returning a bad result.
    """
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)

    if not result.success:
        raise RuntimeError(f"LP solver failed to find a solution, status {result.status}, {result.message}")

    return list(result.x), float(result.fun)
