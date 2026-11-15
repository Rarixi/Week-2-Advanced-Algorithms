"""
numerical.py
---------------
Part 1.2 of the FFTAnalyzer project. Tools for measuring and reasoning
about floating-point error: machine epsilon, absolute / relative / ULP
error, vector and matrix norms, condition numbers (exact and estimated),
an "is this matrix ill-conditioned?" check, and a set of small,
runnable catastrophic-cancellation examples, each with a naive and a
stable formula side by side.

The single most useful rule of thumb this module is built around:

    relative error in the answer  <~  cond(problem) * relative error in the input

Inputs stored in double precision already carry a relative error of up
to eps ~ 1.1e-16 (unit round-off), so a problem with condition number
1e10 can lose about 10 of its ~16 significant digits before any
algorithm runs. A *stable* algorithm is one that doesn't lose much more
than that; an *unstable* one can lose far more, even on a well
conditioned problem (the naive quadratic formula below is the classic
example).
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# machine precision
# ---------------------------------------------------------------------------

def machine_epsilon(dtype=np.float64):
    """
    Measures machine epsilon empirically: keep halving eps until
    1 + eps/2 rounds back to exactly 1. The answer is the gap between 1
    and the next representable number, 2^-52 for float64 and 2^-23 for
    float32, and should match np.finfo(dtype).eps.
    """
    one = dtype(1.0)
    eps = dtype(1.0)
    while dtype(one + eps / dtype(2.0)) != one:
        eps = dtype(eps / dtype(2.0))
    return eps


def unit_roundoff(dtype=np.float64):
    """u = eps / 2, the max relative error of a single rounding (round to nearest)."""
    return machine_epsilon(dtype) / 2


def ulp(x):
    """Size of one unit in the last place at x (spacing to the next float)."""
    return float(np.spacing(np.abs(np.float64(x))))


# ---------------------------------------------------------------------------
# error measures
# ---------------------------------------------------------------------------

def absolute_error(approx, exact):
    """||approx - exact|| (2-norm for arrays, |.| for scalars)."""
    diff = np.asarray(approx, dtype=np.complex128) - np.asarray(exact, dtype=np.complex128)
    return float(np.sqrt(np.sum(np.abs(diff) ** 2)))


def relative_error(approx, exact, floor=0.0):
    """
    ||approx - exact|| / ||exact||. If the exact value is zero the
    relative error is undefined, so this returns the absolute error
    (or divides by `floor` if one is given).
    """
    abs_err = absolute_error(approx, exact)
    denom = float(np.sqrt(np.sum(np.abs(np.asarray(exact, dtype=np.complex128)) ** 2)))
    denom = max(denom, floor)
    return abs_err if denom == 0.0 else abs_err / denom


def ulp_error(approx, exact):
    """Error measured in units of the last place of the exact value."""
    exact = float(exact)
    return abs(float(approx) - exact) / ulp(exact) if exact != 0 else abs(float(approx))


def correct_digits(approx, exact):
    """Roughly how many significant decimal digits agree: -log10(relative error)."""
    rel = relative_error(approx, exact)
    if rel == 0.0:
        return float(np.finfo(np.float64).precision + 1)
    return max(0.0, -math.log10(rel))


def fft_roundtrip_error(x, fft_func, ifft_func):
    """Relative error of ifft(fft(x)) vs x: a direct measure of an FFT's round-off."""
    return relative_error(ifft_func(fft_func(x)), x)


# ---------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------

def vector_norm(x, p=2):
    """
    L1 = sum |x_i|,  L2 = sqrt(sum x_i^2),  Linf = max |x_i|.

    The L2 norm is computed with scaling (divide by the largest |x_i|
    first), the same trick as BLAS dnrm2, so squaring doesn't overflow
    for entries ~1e200 or underflow to zero for entries ~1e-200.
    """
    x = np.abs(np.asarray(x, dtype=np.complex128)).ravel()
    if x.size == 0:
        return 0.0
    if p == 1:
        return float(np.sum(x))
    if p in (np.inf, "inf"):
        return float(np.max(x))
    if p == 2:
        scale = np.max(x)
        if scale == 0.0:
            return 0.0
        return float(scale * np.sqrt(np.sum((x / scale) ** 2)))
    raise ValueError("p must be 1, 2 or inf")


def matrix_norm(A, p=2):
    """
    Induced matrix norms:
      p=1   -> max column sum of |A|
      p=inf -> max row sum of |A|
      p=2   -> largest singular value, sqrt of the dominant eigenvalue of
               A^T A, found with the power method (no SVD call)
      'fro' -> Frobenius, sqrt(sum a_ij^2) (not induced, but an upper bound on p=2)
    """
    A = np.asarray(A, dtype=np.float64)
    if p == 1:
        return float(np.max(np.sum(np.abs(A), axis=0)))
    if p in (np.inf, "inf"):
        return float(np.max(np.sum(np.abs(A), axis=1)))
    if p == "fro":
        return vector_norm(A, 2)
    if p == 2:
        from .matrix import power_method  # local import, avoids a cycle
        if not np.any(A):
            return 0.0
        lam, _, _, _ = power_method(A.T @ A, tol=1e-14, max_iter=20_000)
        return float(np.sqrt(max(lam, 0.0)))
    raise ValueError("p must be 1, 2, inf or 'fro'")


# ---------------------------------------------------------------------------
# condition numbers
# ---------------------------------------------------------------------------

def _lu_solvers(A):
    """
    One LU of A, returned as two closures: solve(b) for A x = b and
    solve_t(b) for A^T x = b (using A^T = U^T L^T P, so no second
    factorization). Returns None if A is exactly singular.
    """
    from .matrix import back_substitution, forward_substitution, lu_decompose
    P, L, U = lu_decompose(A)
    if np.any(np.diag(U) == 0):
        return None

    def solve(b):
        return back_substitution(U, forward_substitution(L, P @ b, unit_diagonal=True))

    def solve_t(b):
        w = forward_substitution(U.T, b)
        v = back_substitution(L.T, w)
        return P.T @ v

    return solve, solve_t


def smallest_singular_value(A, tol=1e-12, max_iter=500):
    """
    sigma_min(A) by inverse iteration on A^T A, *without forming A^T A*.
    (A^T A)^{-1} x = A^{-1} (A^{-T} x), so each step is two triangular-
    pair solves with A's own LU, and at convergence
    1 / sigma_min = ||A^{-T} x|| for the unit vector x.

    Why not just form A^T A and call inverse_iteration on it? Because
    cond(A^T A) = cond(A)^2. For a Hilbert matrix of size 8,
    cond(A) ~ 1.5e10, so cond(A^T A) ~ 2e20 > 1/eps, and the
    factorization of A^T A is pure noise. (An earlier version of this
    function did exactly that and reported cond(H_8) ~ 5.6e8, off by a
    factor of 27; the test suite now checks against np.linalg.cond.)
    """
    A = np.asarray(A, dtype=np.float64)
    solvers = _lu_solvers(A)
    if solvers is None:
        return 0.0
    solve, solve_t = solvers
    x = np.random.default_rng(2).standard_normal(A.shape[0])
    x /= np.linalg.norm(x)
    inv_sigma = 0.0
    for _ in range(max_iter):
        w = solve_t(x)
        z = solve(w)
        new_inv_sigma = float(np.linalg.norm(w))
        nz = np.linalg.norm(z)
        if not np.isfinite(nz) or nz == 0.0:
            return 0.0
        x = z / nz
        if abs(new_inv_sigma - inv_sigma) <= tol * new_inv_sigma:
            inv_sigma = new_inv_sigma
            break
        inv_sigma = new_inv_sigma
    return 1.0 / inv_sigma if inv_sigma > 0 else 0.0


def condition_number(A, p=2):
    """
    kappa_p(A) = ||A||_p * ||A^{-1}||_p. For p = 1 and inf, A^{-1} comes
    from the LU code in matrix.py. For p = 2 it's sigma_max / sigma_min:
    sigma_max from the power method on A^T A (safe, the largest
    eigenvalue is well determined), sigma_min from
    smallest_singular_value (which avoids squaring the condition
    number, see its docstring). Returns inf for a singular matrix.
    """
    from .matrix import inverse
    A = np.asarray(A, dtype=np.float64)
    if p == 2:
        smax = matrix_norm(A, 2)
        smin = smallest_singular_value(A)
        return math.inf if smin == 0.0 else float(smax / smin)
    try:
        Ainv = inverse(A)
    except np.linalg.LinAlgError:
        return math.inf
    if not np.all(np.isfinite(Ainv)):
        return math.inf
    return matrix_norm(A, p) * matrix_norm(Ainv, p)


def condition_estimate_1norm(A, max_iter=5):
    """
    Hager's (1984) 1-norm condition estimator, the idea behind LAPACK's
    xGECON. Computing A^{-1} costs O(n^3); with an LU in hand, this
    estimates ||A^{-1}||_1 with a handful of O(n^2) solves instead, by
    climbing toward the column of A^{-1} with the largest 1-norm:

        x = (1/n, ..., 1/n)
        repeat:  y = A^{-1} x,  s = sign(y),  z = A^{-T} s
                 if max|z| <= z . x: stop, ||A^{-1}||_1 ~ ||y||_1
                 x = e_j for j = argmax |z|

    It's a lower bound that's almost always within a factor of ~3 of the
    truth, which is all a condition number needs to be.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    solvers = _lu_solvers(A)
    if solvers is None:
        return math.inf
    solve, solve_t = solvers

    x = np.full(n, 1.0 / n)
    est = 0.0
    for _ in range(max_iter):
        y = solve(x)
        est = float(np.sum(np.abs(y)))
        s = np.where(y >= 0, 1.0, -1.0)
        z = solve_t(s)
        j = int(np.argmax(np.abs(z)))
        if np.abs(z[j]) <= z @ x:
            break
        x = np.zeros(n)
        x[j] = 1.0
    return matrix_norm(A, 1) * est


def digits_lost(A, p=2):
    """log10(kappa(A)): about how many decimal digits a solve can lose."""
    k = condition_number(A, p)
    return math.inf if math.isinf(k) else math.log10(max(k, 1.0))


def is_ill_conditioned(A, threshold=None, p=2):
    """
    "Is this matrix ill-conditioned?" Returns (verdict, kappa, digits_lost).

    Default threshold: kappa > 1 / sqrt(eps) ~ 6.7e7, i.e. a solve is
    expected to keep fewer than half of double precision's ~16 digits.
    A matrix with kappa >= 1/eps (~4.5e15) is numerically singular,
    nothing meaningful survives.
    """
    eps = np.finfo(np.float64).eps
    if threshold is None:
        threshold = 1.0 / math.sqrt(eps)
    k = condition_number(A, p)
    lost = math.inf if math.isinf(k) else math.log10(max(k, 1.0))
    return bool(k > threshold), k, lost


def stability_report(A, b=None):
    """
    A one-stop summary for a linear system A x = b: condition number,
    digits expected to survive, and, solving with LU + pivoting, the
    relative residual ||b - A x|| / (||A|| ||x||). A backward stable
    solver keeps that residual near eps even when the forward error
    (which kappa controls) is large; that contrast is the point.
    """
    from .matrix import solve
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if b is None:
        b = A @ np.ones(n)  # so the exact answer is all ones
        x_true = np.ones(n)
    else:
        x_true = None
    verdict, kappa, lost = is_ill_conditioned(A)
    x = solve(A, b)
    residual = vector_norm(b - A @ x) / (matrix_norm(A, "fro") * vector_norm(x) + 1e-300)
    report = {
        "n": n,
        "condition_number": kappa,
        "digits_lost": lost,
        "digits_expected": max(0.0, 15.95 - lost),
        "ill_conditioned": verdict,
        "relative_residual": residual,
    }
    if x_true is not None:
        report["forward_relative_error"] = relative_error(x, x_true)
    return report


# ---------------------------------------------------------------------------
# classic test matrices
# ---------------------------------------------------------------------------

def hilbert_matrix(n):
    """H[i, j] = 1 / (i + j + 1). The textbook ill-conditioned matrix:
    kappa_2 grows like e^{3.5 n}, about 1.5e10 at n=8 and 1.6e13 at n=10."""
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


def vandermonde_matrix(x, n=None):
    """V[i, j] = x_i^j: the matrix behind polynomial interpolation,
    ill-conditioned for equally spaced points."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) if n is None else n
    return x[:, None] ** np.arange(n)[None, :]


# ---------------------------------------------------------------------------
# catastrophic cancellation examples
# ---------------------------------------------------------------------------

def quadratic_roots_naive(a, b, c):
    """
    The school formula (-b +/- sqrt(b^2 - 4ac)) / 2a. When b^2 >> 4ac,
    sqrt(b^2 - 4ac) ~ |b| and one of the two numerators subtracts two
    nearly equal numbers, the leading digits cancel, and what's left is
    mostly rounding error.
    """
    d = math.sqrt(b * b - 4 * a * c)
    return (-b + d) / (2 * a), (-b - d) / (2 * a)


def quadratic_roots_stable(a, b, c):
    """
    Compute the root that *adds* magnitudes, q = -(b + sign(b) sqrt(disc)) / 2,
    x1 = q / a, then get the other root from Vieta's formula x1 * x2 = c / a,
    so x2 = c / q. No subtraction of nearly equal numbers anywhere.
    """
    d = math.sqrt(b * b - 4 * a * c)
    q = -0.5 * (b + math.copysign(d, b))
    x1 = q / a
    x2 = c / q if q != 0 else 0.0
    return tuple(sorted((x1, x2), reverse=True))


def one_minus_cos_naive(x):
    """(1 - cos x) / x^2 -> 1/2 as x -> 0, but for x ~ 1e-8 cos x rounds
    to exactly 1.0 and this returns 0."""
    return (1.0 - math.cos(x)) / (x * x)


def one_minus_cos_stable(x):
    """Same function via the identity 1 - cos x = 2 sin^2(x/2): no cancellation."""
    s = math.sin(x / 2.0)
    return 2.0 * s * s / (x * x)


def exp_minus_one_naive(x):
    """e^x - 1 for small x: exp(x) ~ 1 + x and subtracting 1 throws away
    the digits of x that didn't fit next to the leading 1."""
    return math.exp(x) - 1.0


def exp_minus_one_stable(x):
    """math.expm1 evaluates e^x - 1 without ever forming e^x."""
    return math.expm1(x)


def variance_naive(data):
    """
    One-pass E[x^2] - E[x]^2. For data with a large mean and small
    spread, both terms are huge and nearly equal, and the difference
    can come out negative.
    """
    data = [float(v) for v in data]
    n = len(data)
    s = sum(data)
    s2 = sum(v * v for v in data)
    return (s2 - s * s / n) / n


def variance_welford(data):
    """Welford's online algorithm: updates the mean and the sum of
    squared deviations incrementally, no large-minus-large step."""
    mean = 0.0
    m2 = 0.0
    n = 0
    for v in data:
        n += 1
        delta = v - mean
        mean += delta / n
        m2 += delta * (v - mean)
    return m2 / n if n else 0.0


def naive_sum(values):
    """Left-to-right summation; error can grow like n * eps * sum|x|."""
    total = 0.0
    for v in values:
        total += v
    return total


def kahan_sum(values):
    """
    Compensated (Kahan) summation: keep a running correction term c that
    captures the low-order bits lost in each addition and feeds them
    back into the next one. Error bound ~ 2 eps * sum|x|, independent
    of n.
    """
    total = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = total + y
        c = (t - total) - y  # (t - total) is what actually got added; minus y is the lost part
        total = t
    return total


def cancellation_examples():
    """
    Runs every naive/stable pair above on an input chosen to break the
    naive version, with the exact answer known independently, and
    returns rows of {name, input, exact, naive, stable, naive_rel_err,
    stable_rel_err}. Used by the tests, the report and the demo.
    """
    rows = []

    def add(name, inp, exact, naive, stable):
        rows.append({
            "name": name,
            "input": inp,
            "exact": exact,
            "naive": naive,
            "stable": stable,
            "naive_rel_err": abs(naive - exact) / abs(exact),
            "stable_rel_err": abs(stable - exact) / abs(exact),
        })

    # x^2 + 1e8 x + 1 = 0: small root ~ -1e-8 - 1e-24
    a, b, c = 1.0, 1e8, 1.0
    exact_small = -1.0000000000000000e-08  # -(1e-8 + 1e-24 + ...) to double precision
    add("quadratic small root (b=1e8)", (a, b, c), exact_small,
        quadratic_roots_naive(a, b, c)[0], quadratic_roots_stable(a, b, c)[0])

    x = 1e-8
    add("(1 - cos x) / x^2", x, 0.5 - x * x / 24.0, one_minus_cos_naive(x), one_minus_cos_stable(x))

    x = 1e-12
    add("exp(x) - 1", x, x + x * x / 2.0, exp_minus_one_naive(x), exp_minus_one_stable(x))

    data = [1e9 + v for v in (4.0, 7.0, 13.0, 16.0)]
    add("variance, mean 1e9", "1e9 + [4, 7, 13, 16]", 22.5, variance_naive(data), variance_welford(data))

    values = [1.0] + [1e-16] * 1_000_000
    add("sum 1 + 1e6 * 1e-16", "1 + 1e6 copies of 1e-16", 1.0 + 1e-10,
        naive_sum(values), kahan_sum(values))

    return rows
