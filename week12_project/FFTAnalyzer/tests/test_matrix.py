"""
test_matrix.py: matrix multiplication, LU, QR and eigenvalue methods,
checked against np.linalg (LAPACK), plus the stability behaviour each
algorithm is supposed to show (pivoting rescues tiny pivots, Householder
stays orthogonal where Gram-Schmidt doesn't).
"""

import numpy as np
import pytest

from conftest import rel_err
from fft_analyzer.core import matrix as M
from fft_analyzer.core.numerical import hilbert_matrix


# --------------------------------------------------------------------------
# multiplication
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [((1, 1), (1, 1)), ((3, 4), (4, 2)), ((10, 10), (10, 10)), ((7, 1), (1, 5))])
def test_matmul_variants_match_numpy(shape, rng):
    A, B = rng.standard_normal(shape[0]), rng.standard_normal(shape[1])
    ref = A @ B
    assert rel_err(M.matmul_naive(A, B), ref) < 1e-14
    assert rel_err(M.matmul(A, B), ref) < 1e-14
    assert rel_err(M.strassen(A, B, cutoff=1), ref) < 1e-12


@pytest.mark.parametrize("n", [16, 33, 64, 100])
def test_strassen_matches_numpy(n, rng):
    A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    assert rel_err(M.strassen(A, B, cutoff=8), A @ B) < 1e-12


def test_strassen_nonsquare(rng):
    A, B = rng.standard_normal((20, 13)), rng.standard_normal((13, 7))
    assert rel_err(M.strassen(A, B, cutoff=4), A @ B) < 1e-12


def test_matmul_dimension_mismatch():
    with pytest.raises(ValueError):
        M.matmul(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        M.matmul_naive(np.ones((2, 3)), np.ones((2, 3)))


def test_identity_is_neutral(rng):
    A = rng.standard_normal((6, 6))
    assert np.allclose(M.matmul(A, np.eye(6)), A)
    assert np.allclose(M.strassen(np.eye(6), A, cutoff=1), A)


# --------------------------------------------------------------------------
# LU
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
def test_lu_reconstructs(n, rng):
    A = rng.standard_normal((n, n))
    P, L, U = M.lu_decompose(A)
    assert rel_err(P @ A, L @ U) < 1e-14
    assert np.allclose(np.triu(U), U)
    assert np.allclose(np.tril(L), L) and np.allclose(np.diag(L), 1)
    assert np.abs(L).max() <= 1.0 + 1e-15, "partial pivoting bounds every multiplier by 1"
    assert np.allclose(P @ P.T, np.eye(n))


def test_lu_needs_pivot_for_zero_diagonal():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ZeroDivisionError):
        M.lu_no_pivot(A)
    P, L, U = M.lu_decompose(A)
    assert np.allclose(P @ A, L @ U)


def test_pivoting_rescues_tiny_pivot():
    # classic example: without pivoting, the 1e-20 pivot makes the
    # multiplier 1e20 and the answer x = [1, 1] comes out as [0, 1]
    A = np.array([[1e-20, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    L, U = M.lu_no_pivot(A)
    x_bad = M.back_substitution(U, M.forward_substitution(L, b, unit_diagonal=True))
    x_good = M.solve(A, b)
    assert abs(x_bad[0] - 1.0) > 0.5
    assert np.allclose(x_good, [1.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("n", [3, 10, 40])
def test_solve_matches_numpy(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    assert rel_err(M.solve(A, b), np.linalg.solve(A, b)) < 1e-12


def test_solve_multiple_rhs(rng):
    A = rng.standard_normal((8, 8))
    B = rng.standard_normal((8, 3))
    P, L, U = M.lu_decompose(A)
    X = M.lu_solve(P, L, U, B)
    assert rel_err(A @ X, B) < 1e-12


def test_singular_solve_raises():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(np.linalg.LinAlgError):
        M.solve(A, [1.0, 2.0])


@pytest.mark.parametrize("n", [1, 2, 3, 6, 12])
def test_determinant_matches_numpy(n, rng):
    A = rng.standard_normal((n, n))
    assert np.isclose(M.determinant(A), np.linalg.det(A), rtol=1e-11)


def test_determinant_known_values():
    assert M.determinant([[2.0]]) == 2.0
    assert np.isclose(M.determinant([[0, 1], [1, 0]]), -1.0)  # one swap
    assert M.determinant([[1, 2], [2, 4]]) == 0.0
    assert np.isclose(M.determinant(np.diag([1.0, 2.0, 3.0, 4.0])), 24.0)


def test_log_determinant_avoids_underflow():
    A = 0.1 * np.eye(400)  # det = 1e-400, below the smallest double
    assert M.determinant(A) == 0.0
    sign, logdet = M.log_abs_determinant(A)
    assert sign == 1.0 and np.isclose(logdet, 400 * np.log(0.1))


def test_permutation_sign():
    assert M.permutation_sign(np.eye(3)) == 1
    assert M.permutation_sign(np.eye(3)[[1, 0, 2]]) == -1
    assert M.permutation_sign(np.eye(3)[[1, 2, 0]]) == 1


def test_inverse(rng):
    A = rng.standard_normal((7, 7))
    assert rel_err(M.inverse(A), np.linalg.inv(A)) < 1e-12
    assert np.allclose(A @ M.inverse(A), np.eye(7), atol=1e-12)


# --------------------------------------------------------------------------
# QR
# --------------------------------------------------------------------------

QR_METHODS = [M.qr_classical_gram_schmidt, M.qr_modified_gram_schmidt, M.qr_householder]


@pytest.mark.parametrize("method", QR_METHODS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("shape", [(5, 5), (12, 4), (30, 30)])
def test_qr_reconstructs_and_is_orthogonal(method, shape, rng):
    A = rng.standard_normal(shape)
    Q, R = method(A)
    n = shape[1]
    assert Q.shape == shape and R.shape == (n, n)
    assert rel_err(Q @ R, A) < 1e-13
    assert np.abs(Q.T @ Q - np.eye(n)).max() < 1e-12
    assert np.allclose(np.triu(R), R)


def test_householder_complete_mode(rng):
    A = rng.standard_normal((7, 3))
    Q, R = M.qr_householder(A, mode="complete")
    assert Q.shape == (7, 7) and R.shape == (7, 3)
    assert np.allclose(Q @ Q.T, np.eye(7), atol=1e-13)
    assert rel_err(Q @ R, A) < 1e-13


def test_qr_r_matches_numpy_up_to_signs(rng):
    A = rng.standard_normal((6, 6))
    _, R = M.qr_householder(A)
    _, R_np = np.linalg.qr(A)
    assert np.allclose(np.abs(R), np.abs(R_np), atol=1e-12)


def test_orthogonality_loss_ordering_on_hilbert():
    # The expected numerical behaviour: CGS ~ cond^2 eps, MGS ~ cond eps,
    # Householder ~ eps, independent of conditioning
    H = hilbert_matrix(10)  # cond ~ 1.6e13
    loss = {}
    for f in QR_METHODS:
        Q, _ = f(H)
        loss[f.__name__] = np.linalg.norm(Q.T @ Q - np.eye(10))
    assert loss["qr_householder"] < 1e-13
    assert loss["qr_modified_gram_schmidt"] > 1e3 * loss["qr_householder"]
    assert loss["qr_classical_gram_schmidt"] > 10 * loss["qr_modified_gram_schmidt"]


def test_least_squares_matches_numpy(rng):
    A = rng.standard_normal((50, 4))
    b = rng.standard_normal(50)
    assert rel_err(M.least_squares(A, b), np.linalg.lstsq(A, b, rcond=None)[0]) < 1e-12


def test_qr_dispatch(rng):
    A = rng.standard_normal((4, 4))
    for m in ("householder", "mgs", "cgs"):
        Q, R = M.qr_decompose(A, m)
        assert rel_err(Q @ R, A) < 1e-13
    with pytest.raises(ValueError):
        M.qr_decompose(A, "givens")


# --------------------------------------------------------------------------
# eigenvalues
# --------------------------------------------------------------------------

def _sym_with_gap(rng, n=8):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    vals = np.r_[10.0, np.linspace(-4, 4, n - 1)]
    return Q @ np.diag(vals) @ Q.T, vals


def test_power_method_dominant_eigenpair(rng):
    A, vals = _sym_with_gap(rng)
    lam, v, it, ok = M.power_method(A, tol=1e-13)
    assert ok
    assert np.isclose(lam, 10.0, rtol=1e-10)
    assert np.linalg.norm(A @ v - lam * v) < 1e-5


def test_power_method_negative_dominant(rng):
    A, _ = _sym_with_gap(rng)
    lam, v, _, ok = M.power_method(-A, tol=1e-13)
    assert ok and np.isclose(lam, -10.0, rtol=1e-10)


def test_power_method_convergence_rate_depends_on_gap(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    fast = Q @ np.diag([10, 1, 1, 1, 1, 1.0]) @ Q.T
    slow = Q @ np.diag([10, 9.5, 1, 1, 1, 1.0]) @ Q.T
    x0 = np.ones(6)
    _, _, it_fast, _ = M.power_method(fast, x0=x0, tol=1e-12)
    _, _, it_slow, _ = M.power_method(slow, x0=x0, tol=1e-12)
    assert it_slow > 5 * it_fast


def test_inverse_iteration_finds_eigenvalue_near_shift(rng):
    A, vals = _sym_with_gap(rng)
    for target in vals:
        lam, v, _, ok = M.inverse_iteration(A, shift=target + 0.1)
        assert ok and np.isclose(lam, target, atol=1e-9)


def test_qr_algorithm_symmetric(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    vals = np.array([10.0, -6.0, 4.0, -2.5, 1.0, 0.3])  # distinct magnitudes
    A = Q @ np.diag(vals) @ Q.T
    est, _, ok = M.qr_algorithm(A, tol=1e-12)
    assert ok
    assert np.allclose(np.sort(est), np.sort(vals), atol=1e-8)


def test_qr_algorithm_stalls_on_equal_magnitudes(rng):
    # documented limitation of the *unshifted* QR iteration: eigenvalues
    # +4 and -4 have the same magnitude, the ratio |l2/l1| = 1 means that
    # part of the matrix never becomes triangular (shifts fix this)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    A = Q @ np.diag([10.0, 4.0, -4.0, 1.0]) @ Q.T
    _, _, ok = M.qr_algorithm(A, tol=1e-12, max_iter=500)
    assert not ok


def test_qr_algorithm_matches_numpy_eigvals(rng):
    S = rng.standard_normal((5, 5))
    S = S + S.T
    est, _, ok = M.qr_algorithm(S, tol=1e-12, max_iter=20000)
    assert np.allclose(np.sort(est), np.sort(np.linalg.eigvalsh(S)), atol=1e-6)
