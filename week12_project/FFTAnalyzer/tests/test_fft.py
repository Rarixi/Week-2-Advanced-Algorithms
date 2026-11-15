"""
test_fft.py: FFT correctness. Every transform is checked against numpy's
FFT and against the O(n^2) DFT, for power-of-two and arbitrary lengths,
real and complex inputs, plus the algebraic properties a correct DFT must
have (linearity, Parseval, conjugate symmetry, shift theorem), the
convolution theorem, and the 2^20-point performance goal.
"""

import time

import numpy as np
import pytest

from conftest import rel_err
from fft_analyzer.core import fft as F


POW2 = [1, 2, 4, 8, 16, 64, 256, 1024]
ODD_SIZES = [3, 5, 6, 7, 12, 15, 100, 127, 1000]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_is_power_of_two():
    assert [n for n in range(-2, 20) if F.is_power_of_two(n)] == [1, 2, 4, 8, 16]


def test_next_power_of_two():
    assert [F.next_power_of_two(n) for n in (0, 1, 2, 3, 5, 8, 9, 1000)] == [1, 1, 2, 4, 8, 8, 16, 1024]


def test_bit_reverse_n8():
    assert F.bit_reverse_indices(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


def test_bit_reverse_is_involution():
    r = F.bit_reverse_indices(1024)
    assert np.array_equal(r[r], np.arange(1024))
    assert sorted(r.tolist()) == list(range(1024))


def test_bit_reverse_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        F.bit_reverse_indices(12)


# --------------------------------------------------------------------------
# agreement with numpy and the naive DFT
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POW2)
def test_recursive_matches_numpy(n, rng):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel_err(F.fft_recursive(x), np.fft.fft(x)) < 1e-13


@pytest.mark.parametrize("n", POW2)
def test_iterative_matches_numpy(n, rng):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel_err(F.fft_iterative(x), np.fft.fft(x)) < 1e-13


@pytest.mark.parametrize("n", [1, 2, 8, 64, 100, 127])
def test_dft_reference_matches_numpy(n, rng):
    x = rng.standard_normal(n)
    assert rel_err(F.dft(x), np.fft.fft(x)) < 1e-12


@pytest.mark.parametrize("n", [8, 64, 512])
def test_recursive_and_iterative_agree(n, rng):
    x = rng.standard_normal(n)
    assert rel_err(F.fft_recursive(x), F.fft_iterative(x)) < 1e-14


@pytest.mark.parametrize("n", ODD_SIZES)
def test_bluestein_arbitrary_length(n, rng):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel_err(F.fft(x), np.fft.fft(x)) < 1e-12
    assert rel_err(F.fft(x), F.dft(x)) < 1e-12


def test_radix2_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        F.fft_iterative(np.ones(6))
    with pytest.raises(ValueError):
        F.fft_recursive(np.ones(10))


def test_batched_last_axis(rng):
    X = rng.standard_normal((5, 64))
    assert rel_err(F.fft(X), np.fft.fft(X, axis=-1)) < 1e-13
    Y = rng.standard_normal((4, 30))  # Bluestein, batched
    assert rel_err(F.fft(Y), np.fft.fft(Y, axis=-1)) < 1e-12


def test_empty_and_single():
    assert F.fft(np.array([])).size == 0
    assert F.fft([3.0])[0] == 3.0
    assert F.ifft([3.0])[0] == 3.0


def test_known_transforms():
    # delta -> all ones; constant -> n at DC; e^{2 pi i k0 j/n} -> n at bin k0
    n = 16
    d = np.zeros(n)
    d[0] = 1
    assert np.allclose(F.fft(d), np.ones(n))
    assert np.allclose(F.fft(np.ones(n)), np.r_[n, np.zeros(n - 1)])
    k0 = 3
    tone = np.exp(2j * np.pi * k0 * np.arange(n) / n)
    expect = np.zeros(n, complex)
    expect[k0] = n
    assert np.allclose(F.fft(tone), expect, atol=1e-12)


# --------------------------------------------------------------------------
# inverse
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", POW2 + ODD_SIZES)
def test_inverse_round_trip(n, rng):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel_err(F.ifft(F.fft(x)), x) < 1e-13


@pytest.mark.parametrize("n", [16, 256, 100])
def test_ifft_matches_numpy(n, rng):
    X = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel_err(F.ifft(X), np.fft.ifft(X)) < 1e-13


def test_ifft_recursive_round_trip(rng):
    x = rng.standard_normal(128)
    assert rel_err(F.ifft_recursive(F.fft_recursive(x)), x) < 1e-14


# --------------------------------------------------------------------------
# DFT properties
# --------------------------------------------------------------------------

def test_linearity(rng):
    a, b = rng.standard_normal(256), rng.standard_normal(256)
    assert rel_err(F.fft(2.5 * a - 1.5j * b), 2.5 * F.fft(a) - 1.5j * F.fft(b)) < 1e-13


@pytest.mark.parametrize("n", [64, 100, 4096])
def test_parseval(n, rng):
    x = rng.standard_normal(n)
    X = F.fft(x)
    assert np.isclose(np.sum(np.abs(x) ** 2), np.sum(np.abs(X) ** 2) / n, rtol=1e-12)


def test_real_input_conjugate_symmetry(rng):
    n = 128
    X = F.fft(rng.standard_normal(n))
    k = np.arange(1, n)
    assert np.allclose(X[k], np.conj(X[n - k]), atol=1e-12)
    assert abs(X[0].imag) < 1e-12 and abs(X[n // 2].imag) < 1e-12


def test_shift_theorem(rng):
    # circular shift by m multiplies bin k by exp(-2 pi i k m / n)
    n, m = 64, 5
    x = rng.standard_normal(n)
    k = np.arange(n)
    assert rel_err(F.fft(np.roll(x, m)), F.fft(x) * np.exp(-2j * np.pi * k * m / n)) < 1e-13


@pytest.mark.parametrize("n", [2 ** 10, 2 ** 14, 2 ** 18])
def test_round_off_growth_is_small(n, rng):
    # FFT error should grow ~ log n, not ~ n (the naive DFT's error does)
    x = rng.standard_normal(n)
    err = rel_err(F.ifft(F.fft(x)), x)
    assert err < 10 * np.log2(n) * np.finfo(float).eps


# --------------------------------------------------------------------------
# helpers: frequencies and shifts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 8, 101])
def test_fftfreq_matches_numpy(n):
    assert np.allclose(F.fftfreq(n, 0.01), np.fft.fftfreq(n, 0.01))
    assert np.allclose(F.rfftfreq(n, 0.01), np.fft.rfftfreq(n, 0.01))


def test_fftshift_matches_numpy(rng):
    for shape in [(8,), (7,), (6, 9)]:
        a = rng.standard_normal(shape)
        assert np.array_equal(F.fftshift(a), np.fft.fftshift(a))
        assert np.array_equal(F.ifftshift(F.fftshift(a)), a)


# --------------------------------------------------------------------------
# convolution
# --------------------------------------------------------------------------

def test_circular_convolution_definition(rng):
    n = 32
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    direct = np.array([sum(a[j] * b[(k - j) % n] for j in range(n)) for k in range(n)])
    assert rel_err(F.circular_convolve(a, b), direct) < 1e-13


def test_circular_convolution_real_output(rng):
    out = F.circular_convolve(rng.standard_normal(16), rng.standard_normal(16))
    assert not np.iscomplexobj(out)


def test_circular_convolution_length_check():
    with pytest.raises(ValueError):
        F.circular_convolve(np.ones(4), np.ones(5))


@pytest.mark.parametrize("na,nb", [(1, 1), (5, 3), (100, 37), (1000, 1), (64, 64), (300, 1000)])
def test_linear_convolution_matches_numpy(na, nb, rng):
    a, b = rng.standard_normal(na), rng.standard_normal(nb)
    out = F.linear_convolve(a, b)
    assert out.size == na + nb - 1
    assert rel_err(out, np.convolve(a, b)) < 1e-12
    assert rel_err(F.direct_convolve(a, b), np.convolve(a, b)) < 1e-13


def test_linear_convolution_complex(rng):
    a = rng.standard_normal(20) + 1j * rng.standard_normal(20)
    b = rng.standard_normal(7)
    assert rel_err(F.linear_convolve(a, b), np.convolve(a, b)) < 1e-12


def test_zero_padding_prevents_wraparound():
    # unpadded circular convolution of [1,1,1,1] with itself wraps; linear doesn't
    a = np.ones(4)
    assert np.allclose(F.circular_convolve(a, a), [4, 4, 4, 4])
    assert np.allclose(F.linear_convolve(a, a), [1, 2, 3, 4, 3, 2, 1])


def test_polynomial_multiplication():
    # (1 + 2x + 3x^2)(4 + 5x) = 4 + 13x + 22x^2 + 15x^3
    assert np.allclose(F.linear_convolve([1, 2, 3], [4, 5]), [4, 13, 22, 15])


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(8, 8), (16, 32), (12, 20)])
def test_fft2_matches_numpy(shape, rng):
    a = rng.standard_normal(shape)
    assert rel_err(F.fft2(a), np.fft.fft2(a)) < 1e-12
    assert rel_err(F.ifft2(F.fft2(a)), a) < 1e-13


def test_convolve2d_full_matches_direct(rng):
    img = rng.standard_normal((10, 12))
    k = rng.standard_normal((3, 4))
    direct = np.zeros((12, 15))
    for i in range(3):
        for j in range(4):
            direct[i:i + 10, j:j + 12] += k[i, j] * img
    assert rel_err(F.convolve2d(img, k, "full"), direct) < 1e-12
    assert F.convolve2d(img, k, "same").shape == img.shape


# --------------------------------------------------------------------------
# performance goal
# --------------------------------------------------------------------------

def test_two_to_the_twenty_points_is_fast(rng):
    n = 2 ** 20
    x = rng.standard_normal(n)
    t0 = time.perf_counter()
    X = F.fft(x)
    elapsed = time.perf_counter() - t0
    assert rel_err(X, np.fft.fft(x)) < 1e-12
    assert elapsed < 5.0, f"2^20-point FFT took {elapsed:.2f}s"


def test_fft_scales_n_log_n(rng):
    # doubling n should roughly double the time (allowing generous slack),
    # nowhere near the 4x an O(n^2) algorithm would show
    def best(n, reps=3):
        x = rng.standard_normal(n)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            F.fft_iterative(x)
            ts.append(time.perf_counter() - t0)
        return min(ts)
    ratio = best(2 ** 18) / best(2 ** 16)  # 4x the data
    assert ratio < 9.0, f"4x data took {ratio:.1f}x the time"
