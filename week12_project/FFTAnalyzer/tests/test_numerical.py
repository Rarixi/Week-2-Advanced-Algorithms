"""
test_numerical.py: floating-point and stability utilities. Machine
epsilon, error measures, norms and condition numbers are checked against
numpy; every catastrophic-cancellation example must show the naive
formula failing and the stable one succeeding.
"""

import math

import numpy as np
import pytest

from fft_analyzer.core import numerical as N
from fft_analyzer.core.fft import fft, ifft


def test_machine_epsilon():
    assert N.machine_epsilon(np.float64) == np.finfo(np.float64).eps
    assert N.machine_epsilon(np.float32) == np.finfo(np.float32).eps
    assert N.unit_roundoff() == np.finfo(np.float64).eps / 2


def test_one_plus_eps_behaviour():
    eps = N.machine_epsilon()
    assert 1.0 + eps != 1.0
    assert 1.0 + eps / 2 == 1.0


def test_ulp():
    assert N.ulp(1.0) == np.finfo(float).eps
    assert N.ulp(2.0) == 2 * np.finfo(float).eps
    assert N.ulp_error(1.0 + 3 * np.finfo(float).eps, 1.0) == pytest.approx(3.0)


def test_errors():
    assert N.absolute_error(1.1, 1.0) == pytest.approx(0.1)
    assert N.relative_error(110, 100) == pytest.approx(0.1)
    assert N.relative_error([3, 4.0], [3, 3.0]) == pytest.approx(1 / np.sqrt(18))
    assert N.relative_error(0.5, 0.0) == 0.5  # falls back to absolute
    assert N.correct_digits(1.0001, 1.0) == pytest.approx(4.0, abs=0.01)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_vector_norms(p, rng):
    x = rng.standard_normal(50)
    assert N.vector_norm(x, p) == pytest.approx(np.linalg.norm(x, p))


def test_l2_norm_does_not_overflow_or_underflow():
    assert N.vector_norm([3e200, 4e200]) == pytest.approx(5e200)
    assert N.vector_norm([3e-200, 4e-200]) == pytest.approx(5e-200)
    with np.errstate(over="ignore"):
        assert math.isinf(np.sqrt(np.sum(np.array([3e200, 4e200]) ** 2)))  # the naive way does


@pytest.mark.parametrize("p", [1, 2, np.inf, "fro"])
def test_matrix_norms(p, rng):
    A = rng.standard_normal((6, 4))
    assert N.matrix_norm(A, p) == pytest.approx(np.linalg.norm(A, p), rel=1e-9)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_condition_number_random(p, rng):
    A = rng.standard_normal((8, 8))
    assert N.condition_number(A, p) == pytest.approx(np.linalg.cond(A, p), rel=1e-6)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_condition_number_hilbert(n):
    # regression test: forming A^T A squares cond(A) and wrecks sigma_min;
    # the implementation must avoid that
    H = N.hilbert_matrix(n)
    assert N.condition_number(H) == pytest.approx(np.linalg.cond(H), rel=1e-3)


def test_condition_of_singular_is_inf():
    assert math.isinf(N.condition_number(np.array([[1.0, 2.0], [2.0, 4.0]]), p=1))


def test_hager_estimate_is_close_lower_bound(rng):
    for _ in range(5):
        A = rng.standard_normal((10, 10))
        est, true = N.condition_estimate_1norm(A), np.linalg.cond(A, 1)
        assert est <= true * (1 + 1e-10)
        assert est >= true / 3


def test_is_ill_conditioned():
    well, k, lost = N.is_ill_conditioned(np.eye(5))
    assert not well and k == pytest.approx(1.0) and lost == pytest.approx(0.0, abs=1e-9)
    bad, k, lost = N.is_ill_conditioned(N.hilbert_matrix(9))
    assert bad and k > 1e11 and lost > 11


def test_hilbert_and_vandermonde():
    H = N.hilbert_matrix(3)
    assert np.allclose(H, [[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4], [1 / 3, 1 / 4, 1 / 5]])
    assert np.allclose(N.vandermonde_matrix([1, 2, 3]), np.vander([1, 2, 3], increasing=True))


def test_stability_report_backward_stable_but_forward_inaccurate():
    # LU with pivoting keeps the residual at ~eps (backward stable) even
    # when cond(A) ~ 1e13 makes the forward error large
    r = N.stability_report(N.hilbert_matrix(10))
    assert r["relative_residual"] < 1e-14
    assert r["forward_relative_error"] > 1e-8
    assert r["forward_relative_error"] < 1e-15 * r["condition_number"] * 100


def test_error_grows_with_condition_number():
    errs = []
    for n in (4, 6, 8, 10):
        errs.append(N.stability_report(N.hilbert_matrix(n))["forward_relative_error"])
    assert errs == sorted(errs)


# --------------------------------------------------------------------------
# catastrophic cancellation
# --------------------------------------------------------------------------

def test_quadratic_roots():
    naive = N.quadratic_roots_naive(1.0, 1e8, 1.0)
    stable = N.quadratic_roots_stable(1.0, 1e8, 1.0)
    assert abs(naive[0] - (-1e-8)) / 1e-8 > 0.1           # >10% wrong
    assert abs(stable[0] - (-1e-8)) / 1e-8 < 1e-15
    # well-conditioned case: both agree with numpy
    assert np.allclose(sorted(N.quadratic_roots_stable(1, -3, 2)), [1, 2])
    assert np.allclose(sorted(N.quadratic_roots_naive(1, -3, 2)), [1, 2])


def test_one_minus_cos():
    assert N.one_minus_cos_naive(1e-8) == 0.0
    assert N.one_minus_cos_stable(1e-8) == pytest.approx(0.5, rel=1e-15)


def test_expm1():
    x = 1e-12
    assert abs(N.exp_minus_one_naive(x) - x) / x > 1e-5
    assert N.exp_minus_one_stable(x) == pytest.approx(x, rel=1e-15)


def test_variance():
    data = [1e9 + v for v in (4.0, 7.0, 13.0, 16.0)]
    assert N.variance_welford(data) == pytest.approx(22.5, rel=1e-12)
    assert abs(N.variance_naive(data) - 22.5) > 1.0


def test_kahan_summation():
    vals = [1.0] + [1e-16] * 100_000
    assert N.naive_sum(vals) == 1.0
    assert N.kahan_sum(vals) == pytest.approx(1.0 + 1e-11, rel=1e-15)


def test_cancellation_table():
    rows = N.cancellation_examples()
    assert len(rows) == 5
    for r in rows:
        assert r["stable_rel_err"] < 1e-12, r["name"]
        assert r["naive_rel_err"] > 1e3 * max(r["stable_rel_err"], 1e-16), r["name"]


def test_fft_roundtrip_error_helper(rng):
    assert N.fft_roundtrip_error(rng.standard_normal(1024), fft, ifft) < 1e-14
