"""
matrix.py
------------
Part 1.1 of the FFTAnalyzer project. Dense linear algebra written from
scratch: matrix multiplication (classical and Strassen), LU
decomposition with partial pivoting, solving linear systems and
determinants through LU, QR decomposition three ways (classical
Gram-Schmidt, modified Gram-Schmidt, Householder), and basic eigenvalue
methods (power method, inverse iteration, unshifted QR iteration).

numpy is used for array storage and for vectorizing the innermost loop
of each algorithm (one row update, one dot product), but never for the
algorithm itself: no np.linalg anywhere in this file. The tests use
np.linalg as the independent reference.

The theme running through the module is stability vs performance:

  * LU *without* pivoting is the cheapest elimination there is, and it
    falls apart the moment a pivot is tiny (see lu_no_pivot and the
    test that feeds it [[1e-20, 1], [1, 1]]). Partial pivoting costs
    O(n^2) comparisons on top of O(n^3) work, essentially free, and
    keeps every multiplier |l_ij| <= 1, which is what makes Gaussian
    elimination backward stable in practice.
  * Classical Gram-Schmidt is the most natural way to write QR, and it
    loses orthogonality roughly in proportion to cond(A)^2 * eps. The
    modified version does the same arithmetic in a different order
    and the loss drops to roughly cond(A) * eps. Householder
    reflections are orthogonal to working precision regardless of
    conditioning. The tests and the benchmark measure ||Q^T Q - I|| for
    all three on increasingly ill-conditioned Hilbert matrices.
  * Strassen's algorithm does 7 recursive multiplications instead of 8,
    O(n^2.807) instead of O(n^3), but it adds and subtracts submatrices
    more often, so its error bound is weaker (normwise rather than
    componentwise), and the constant factor means it only wins above a
    cutoff size.
"""

import numpy as np


def _as_float_matrix(A, name="A"):
    A = np.array(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {A.shape}")
    return A


def _check_square(A, name="A"):
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")


# ---------------------------------------------------------------------------
# matrix multiplication
# ---------------------------------------------------------------------------

def matmul_naive(A, B):
    """
    The classical O(n^3) triple loop, C[i][j] = sum_k A[i][k] * B[k][j],
    in pure Python with no vectorization at all. This is the baseline
    every other multiplication here is checked and timed against, and it
    is slow on purpose: it shows how much of "O(n^3)" in practice is
    interpreter overhead rather than arithmetic.
    """
    A = _as_float_matrix(A, "A")
    B = _as_float_matrix(B, "B")
    n, m = A.shape
    m2, p = B.shape
    if m != m2:
        raise ValueError(f"inner dimensions differ: {A.shape} x {B.shape}")
    a, b = A.tolist(), B.tolist()
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai, Ci = a[i], C[i]
        for k in range(m):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = b[k]
            for j in range(p):
                Ci[j] += aik * Bk[j]
    return np.array(C)


def matmul(A, B):
    """
    Still the classical O(n^3) algorithm, i-k-j loop order, but with the
    innermost j loop done as one numpy row operation:
    C[i, :] += A[i, k] * B[k, :]. Same arithmetic as matmul_naive, much
    lower constant factor. Used as the base case for Strassen.
    """
    A = _as_float_matrix(A, "A")
    B = _as_float_matrix(B, "B")
    n, m = A.shape
    if m != B.shape[0]:
        raise ValueError(f"inner dimensions differ: {A.shape} x {B.shape}")
    C = np.zeros((n, B.shape[1]))
    for k in range(m):
        # rank-1 update: column k of A times row k of B, all rows at once
        C += np.outer(A[:, k], B[k, :])
    return C


def strassen(A, B, cutoff=64):
    """
    Strassen's algorithm. Splits each matrix into 2x2 blocks and forms
    the product from 7 block multiplications instead of 8:

        M1 = (A11 + A22)(B11 + B22)     C11 = M1 + M4 - M5 + M7
        M2 = (A21 + A22) B11            C12 = M3 + M5
        M3 = A11 (B12 - B22)            C21 = M2 + M4
        M4 = A22 (B21 - B11)            C22 = M1 - M2 + M3 + M6
        M5 = (A11 + A12) B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)

    T(n) = 7 T(n/2) + O(n^2) = O(n^log2(7)) ~ O(n^2.807). Below `cutoff`
    the extra additions cost more than the saved multiplication, so it
    falls back to the classical algorithm. Non-square or odd-sized inputs
    are zero-padded up to a power-of-two square and the result trimmed.
    """
    A = _as_float_matrix(A, "A")
    B = _as_float_matrix(B, "B")
    n, m = A.shape
    if m != B.shape[0]:
        raise ValueError(f"inner dimensions differ: {A.shape} x {B.shape}")
    p = B.shape[1]
    size = 1
    while size < max(n, m, p):
        size *= 2
    Ap = np.zeros((size, size))
    Bp = np.zeros((size, size))
    Ap[:n, :m] = A
    Bp[:m, :p] = B
    return _strassen_rec(Ap, Bp, max(1, cutoff))[:n, :p]


def _strassen_rec(A, B, cutoff):
    n = A.shape[0]
    if n <= cutoff:
        return matmul(A, B)
    h = n // 2
    A11, A12, A21, A22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    B11, B12, B21, B22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]

    M1 = _strassen_rec(A11 + A22, B11 + B22, cutoff)
    M2 = _strassen_rec(A21 + A22, B11, cutoff)
    M3 = _strassen_rec(A11, B12 - B22, cutoff)
    M4 = _strassen_rec(A22, B21 - B11, cutoff)
    M5 = _strassen_rec(A11 + A12, B22, cutoff)
    M6 = _strassen_rec(A21 - A11, B11 + B12, cutoff)
    M7 = _strassen_rec(A12 - A22, B21 + B22, cutoff)

    C = np.empty((n, n))
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    return C


# ---------------------------------------------------------------------------
# LU decomposition
# ---------------------------------------------------------------------------

def lu_no_pivot(A):
    """
    Doolittle LU *without* pivoting, A = L U. Included only to show why
    pivoting matters: it divides by whatever happens to be on the
    diagonal, so a zero pivot raises and a tiny pivot produces huge
    multipliers that wipe out the information in the rest of the row.
    """
    A = _as_float_matrix(A)
    _check_square(A)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    for k in range(n - 1):
        if U[k, k] == 0.0:
            raise ZeroDivisionError(f"zero pivot at step {k}, pivoting required")
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
    return L, np.triu(U)


def lu_decompose(A):
    """
    LU decomposition with partial pivoting: P A = L U, where P is a
    permutation matrix, L is unit lower triangular with |L[i, j]| <= 1,
    and U is upper triangular.

    At step k, the row with the largest |entry| in column k (on or below
    the diagonal) is swapped up to be the pivot. Every multiplier is then
    (something) / (the largest thing available), so it's at most 1 in
    magnitude, and rounding errors can't be amplified through the
    elimination the way they are with a tiny pivot.

    Returns (P, L, U). A singular matrix still factors (U just gets a
    zero on its diagonal); solving with it is what raises.
    Cost: 2n^3/3 flops.
    """
    A = _as_float_matrix(A)
    _check_square(A)
    n = A.shape[0]
    U = A.copy()
    L = np.zeros((n, n))
    perm = np.arange(n)

    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
            perm[[k, pivot_row]] = perm[[pivot_row, k]]
        L[k, k] = 1.0
        if U[k, k] == 0.0:
            continue  # whole column below is zero too, nothing to eliminate
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
        U[k + 1:, k] = 0.0

    P = np.eye(n)[perm]
    return P, L, U


def forward_substitution(L, b, unit_diagonal=False):
    """Solve L y = b for lower triangular L, top row first. O(n^2)."""
    L = _as_float_matrix(L, "L")
    b = np.array(b, dtype=np.float64)
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        s = b[i] - L[i, :i] @ y[:i]
        y[i] = s if unit_diagonal else s / L[i, i]
    return y


def back_substitution(U, y):
    """Solve U x = y for upper triangular U, bottom row first. O(n^2)."""
    U = _as_float_matrix(U, "U")
    y = np.array(y, dtype=np.float64)
    n = U.shape[0]
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise np.linalg.LinAlgError("matrix is singular (zero pivot in U)")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu_solve(P, L, U, b):
    """
    Solve A x = b given P A = L U: then L U x = P b, so solve L y = P b
    (forward), then U x = y (back). O(n^2) per right-hand side, which is
    the payoff of factoring once and solving many times. `b` may be a
    vector or a matrix of right-hand-side columns.
    """
    b = np.array(b, dtype=np.float64)
    Pb = P @ b
    y = forward_substitution(L, Pb, unit_diagonal=True)
    return back_substitution(U, y)


def solve(A, b):
    """Solve A x = b with LU + partial pivoting."""
    P, L, U = lu_decompose(A)
    return lu_solve(P, L, U, b)


def permutation_sign(P):
    """+1 or -1: the determinant of a permutation matrix (parity of the
    number of swaps needed to sort it), found by counting cycles."""
    perm = np.argmax(np.asarray(P), axis=1)
    n = len(perm)
    seen = np.zeros(n, dtype=bool)
    sign = 1
    for i in range(n):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def determinant(A):
    """
    det(A) from P A = L U: det(P) det(A) = det(L) det(U) = prod(diag(U)),
    since L has a unit diagonal, so det(A) = sign(P) * prod(diag(U)).
    O(n^3) instead of the O(n!) cofactor expansion.
    """
    P, L, U = lu_decompose(A)
    return permutation_sign(P) * float(np.prod(np.diag(U)))


def log_abs_determinant(A):
    """
    (sign, log|det A|). The plain product of n pivots under/overflows
    easily (det of 0.1 * I for n = 400 is 1e-400, below the smallest
    double), the sum of logs doesn't.
    """
    P, L, U = lu_decompose(A)
    d = np.diag(U)
    if np.any(d == 0):
        return 0.0, -np.inf
    sign = permutation_sign(P) * int(np.prod(np.sign(d)))
    return float(sign), float(np.sum(np.log(np.abs(d))))


def inverse(A):
    """A^{-1} by solving A X = I one column at a time from one LU."""
    A = _as_float_matrix(A)
    _check_square(A)
    P, L, U = lu_decompose(A)
    n = A.shape[0]
    return np.column_stack([lu_solve(P, L, U, e) for e in np.eye(n)])


# ---------------------------------------------------------------------------
# QR decomposition
# ---------------------------------------------------------------------------

def qr_classical_gram_schmidt(A):
    """
    Classical Gram-Schmidt: q_j = normalize(a_j - sum_{i<j} (q_i . a_j) q_i).
    Each projection coefficient is computed against the *original*
    column a_j, so any rounding error already in q_1..q_{j-1} is never
    corrected, and orthogonality decays like cond(A)^2 * eps. Returns the
    reduced factorization Q (m x n), R (n x n).
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[:j, j] = Q[:, :j].T @ A[:, j]
        v = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.sqrt(v @ v)
        if R[j, j] == 0.0:
            raise np.linalg.LinAlgError("columns are linearly dependent")
        Q[:, j] = v / R[j, j]
    return Q, R


def qr_modified_gram_schmidt(A):
    """
    Modified Gram-Schmidt. Same projections, but each one is subtracted
    from the *running* vector as soon as it's computed, so later
    coefficients are taken against a vector that already has the
    earlier (slightly wrong) components removed. Mathematically
    identical to the classical version, numerically much better:
    loss of orthogonality ~ cond(A) * eps.
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    V = A.copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.sqrt(V[:, j] @ V[:, j])
        if R[j, j] == 0.0:
            raise np.linalg.LinAlgError("columns are linearly dependent")
        Q[:, j] = V[:, j] / R[j, j]
        R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
        V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    return Q, R


def householder_vector(x):
    """
    The Householder vector v (normalized, ||v|| = 1) such that
    (I - 2 v v^T) x = -sign(x0) ||x|| e1. The sign is chosen so that
    x0 and the added ||x|| have the same sign, which avoids the
    cancellation x0 - ||x|| ~ 0 when x is already nearly along e1.
    """
    x = np.array(x, dtype=np.float64)
    norm_x = np.sqrt(x @ x)
    v = x.copy()
    if norm_x == 0.0:
        return v  # nothing to reflect, caller skips
    v[0] += np.copysign(norm_x, x[0]) if x[0] != 0 else norm_x
    v /= np.sqrt(v @ v)
    return v


def qr_householder(A, mode="reduced"):
    """
    QR via Householder reflections. Column by column, a reflector
    H_k = I - 2 v v^T zeroes everything below the diagonal; R is what's
    left, Q = H_1 H_2 ... H_n. Every H_k is exactly orthogonal in exact
    arithmetic and is applied as a rank-1 update (never formed as a
    matrix), so Q is orthogonal to working precision regardless of how
    ill-conditioned A is. Cost ~ 2mn^2 - 2n^3/3 flops.

    mode="reduced" -> Q (m x n), R (n x n); "complete" -> Q (m x m), R (m x n).
    """
    A = _as_float_matrix(A)
    m, n = A.shape
    R = A.copy()
    vs = []
    for k in range(min(m - 1, n)):  # the last row of a square matrix needs no reflector
        v = householder_vector(R[k:, k])
        vs.append(v)
        if not np.any(v):
            continue
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])

    Q = np.eye(m)
    for k in range(len(vs) - 1, -1, -1):  # apply H_k in reverse to build Q
        v = vs[k]
        if not np.any(v):
            continue
        Q[k:, :] -= 2.0 * np.outer(v, v @ Q[k:, :])

    R = np.triu(R)
    if mode == "complete":
        return Q, R
    if mode == "reduced":
        return Q[:, :n], R[:n, :]
    raise ValueError("mode must be 'reduced' or 'complete'")


def qr_decompose(A, method="householder"):
    """Dispatch: method in {'householder', 'mgs', 'cgs'}."""
    if method == "householder":
        return qr_householder(A)
    if method == "mgs":
        return qr_modified_gram_schmidt(A)
    if method == "cgs":
        return qr_classical_gram_schmidt(A)
    raise ValueError(f"unknown QR method {method!r}")


def least_squares(A, b):
    """
    Minimize ||A x - b||_2 via QR: A = Q R  =>  R x = Q^T b. Avoids the
    normal equations A^T A x = A^T b, which square the condition number.
    """
    Q, R = qr_householder(A)
    return back_substitution(R, Q.T @ np.asarray(b, dtype=np.float64))


# ---------------------------------------------------------------------------
# eigenvalues
# ---------------------------------------------------------------------------

def power_method(A, x0=None, tol=1e-10, max_iter=10_000, rng=None):
    """
    Dominant eigenpair by repeated multiplication: x <- A x / ||A x||.
    Writing x0 in the eigenbasis, A^k x0 = sum c_i lambda_i^k v_i, and
    every term dies relative to the largest |lambda_1|, at rate
    |lambda_2 / lambda_1| per step. So convergence is linear and slow
    when the top two eigenvalues are close in magnitude, and it fails
    outright when they're equal in magnitude (e.g. lambda and -lambda).

    The eigenvalue estimate is the Rayleigh quotient x^T A x (x has unit
    norm), which for symmetric A converges twice as fast as x itself.

    Returns (eigenvalue, eigenvector, iterations, converged).
    """
    A = _as_float_matrix(A)
    _check_square(A)
    n = A.shape[0]
    if x0 is None:
        rng = rng or np.random.default_rng(0)
        x = rng.standard_normal(n)
    else:
        x = np.array(x0, dtype=np.float64)
    x /= np.linalg.norm(x)
    lam = x @ A @ x
    for it in range(1, max_iter + 1):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny == 0.0:
            return 0.0, x, it, True
        x_new = y / ny
        # fix the sign so a negative eigenvalue doesn't make x flip every step
        if x_new @ x < 0:
            x_new = -x_new
        lam_new = x_new @ A @ x_new
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)) and \
                np.linalg.norm(x_new - x) <= np.sqrt(tol):
            return float(lam_new), x_new, it, True
        x, lam = x_new, lam_new
    return float(lam), x, max_iter, False


def inverse_iteration(A, shift=0.0, x0=None, tol=1e-12, max_iter=1000, rng=None):
    """
    Power method on (A - shift I)^{-1}, whose dominant eigenvalue is
    1 / (lambda - shift) for the eigenvalue lambda closest to `shift`.
    One LU of the shifted matrix is reused for every iteration, so each
    step is just an O(n^2) solve. With a good shift this converges
    very fast; shift = 0 gives the smallest-magnitude eigenvalue.

    Returns (eigenvalue, eigenvector, iterations, converged).
    """
    A = _as_float_matrix(A)
    _check_square(A)
    n = A.shape[0]
    P, L, U = lu_decompose(A - shift * np.eye(n))
    if np.any(np.abs(np.diag(U)) < 1e-300):
        # shift is (numerically) an eigenvalue: nudge it
        P, L, U = lu_decompose(A - (shift + 1e-10) * np.eye(n))
    if x0 is None:
        rng = rng or np.random.default_rng(1)
        x = rng.standard_normal(n)
    else:
        x = np.array(x0, dtype=np.float64)
    x /= np.linalg.norm(x)
    lam = x @ A @ x
    for it in range(1, max_iter + 1):
        y = lu_solve(P, L, U, x)
        x_new = y / np.linalg.norm(y)
        if x_new @ x < 0:
            x_new = -x_new
        lam_new = x_new @ A @ x_new
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            return float(lam_new), x_new, it, True
        x, lam = x_new, lam_new
    return float(lam), x, max_iter, False


def qr_algorithm(A, tol=1e-10, max_iter=5000):
    """
    Unshifted QR iteration: A_{k+1} = R_k Q_k where A_k = Q_k R_k. Each
    step is a similarity transform (R Q = Q^T A Q), so eigenvalues are
    preserved, and for a matrix with eigenvalues of distinct magnitude
    A_k converges to upper triangular with the eigenvalues on the
    diagonal (diagonal for symmetric A). This is the "basic" version:
    no Hessenberg reduction, no shifts, so it's O(n^3) per step and
    converges at the same linear rates as the power method. Good enough
    to check the power method and inverse iteration against.

    Returns (eigenvalues sorted by decreasing magnitude, iterations, converged).
    """
    Ak = _as_float_matrix(A)
    _check_square(Ak)
    n = Ak.shape[0]
    scale = max(1.0, np.abs(Ak).max())
    for it in range(1, max_iter + 1):
        Q, R = qr_householder(Ak)
        Ak = R @ Q
        if np.abs(np.tril(Ak, -1)).max(initial=0.0) <= tol * scale:
            vals = np.diag(Ak)
            return vals[np.argsort(-np.abs(vals))], it, True
    vals = np.diag(Ak)
    return vals[np.argsort(-np.abs(vals))], max_iter, False
