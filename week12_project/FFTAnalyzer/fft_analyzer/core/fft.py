"""
fft.py
---------
Part 2.1 of the FFTAnalyzer project. The Cooley-Tukey Fast Fourier
Transform, written from scratch, plus everything the rest of the toolkit
builds on top of it: the inverse transform, circular and linear
convolution, 2D transforms for images, and a few small helpers
(frequency bins, shifting the zero frequency to the middle).

The discrete Fourier transform of a length-n sequence x is

    X[k] = sum_{j=0}^{n-1} x[j] * exp(-2*pi*i*j*k / n),   k = 0..n-1

Computed directly that's n outputs times n terms, O(n^2). Cooley-Tukey
splits the sum into the even-indexed and odd-indexed terms:

    X[k]       = E[k] + w^k * O[k]
    X[k + n/2] = E[k] - w^k * O[k],        w = exp(-2*pi*i / n)

where E and O are the length-n/2 DFTs of x[0::2] and x[1::2]. Two
half-size problems plus O(n) combining work gives T(n) = 2T(n/2) + O(n),
which is O(n log n). The pair of lines above is the "butterfly", one
multiply by the twiddle factor w^k feeds two outputs.

Two implementations of the same recurrence are given:

  * fft_recursive   -- the recurrence written exactly as above. Easy to
                       read and to check, but it makes 2n - 1 Python
                       function calls, which dominates its running time.
  * fft_iterative   -- the bottom-up version. The recursion always ends
                       up reading the input in bit-reversed index order,
                       so the iterative version does that permutation
                       once up front, then runs log2(n) passes of
                       butterflies, each pass vectorized with numpy over
                       every butterfly at that size at once. This is the
                       one that handles 2^20 points comfortably.

Both need n to be a power of two. The public `fft` / `ifft` entry points
accept any length: powers of two go straight to the iterative FFT, any
other length goes through Bluestein's algorithm, which rewrites a
length-n DFT as a convolution and evaluates that convolution with
power-of-two FFTs, so it's still O(n log n).

A note on accuracy: the twiddle factors for each pass are computed
directly as exp(-2*pi*i*k/size), never by repeatedly multiplying
w * w * w ..., since that recurrence accumulates rounding error linearly
in k. With direct twiddles the FFT's round-off error grows like
O(log n) * machine epsilon, the test suite checks this against numpy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def is_power_of_two(n):
    """True for 1, 2, 4, 8, ...; False for 0, negatives and everything else."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n):
    """Smallest power of two >= n (and >= 1)."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def bit_reverse_indices(n):
    """
    The bit-reversal permutation for a power-of-two n: index i maps to
    the integer whose log2(n)-bit binary representation is i's read
    backwards. For n = 8: [0, 4, 2, 6, 1, 5, 3, 7]. Built one bit at a
    time over the whole index array, so it costs O(n log n) numpy work
    rather than a Python loop over n indices.
    """
    if not is_power_of_two(n):
        raise ValueError(f"n must be a power of two, got {n}")
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def _as_complex(x):
    return np.asarray(x, dtype=np.complex128)


# ---------------------------------------------------------------------------
# reference DFT (O(n^2)), used by the tests and for tiny inputs
# ---------------------------------------------------------------------------

def dft(x):
    """
    The DFT straight from its definition, as a matrix-vector product
    with the n x n Fourier matrix F[k, j] = exp(-2*pi*i*j*k/n). O(n^2)
    time and memory, only meant as an independent reference to check
    the fast versions against. The exponent j*k is reduced mod n before
    it goes into exp(), which keeps the angle small and accurate.
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    jk = np.outer(np.arange(n), np.arange(n)) % n
    F = np.exp(-2j * np.pi * jk / n)
    return x @ F.T


# ---------------------------------------------------------------------------
# recursive radix-2 FFT
# ---------------------------------------------------------------------------

def fft_recursive(x):
    """
    Textbook recursive radix-2 Cooley-Tukey. Splits into even and odd
    samples, transforms each half recursively, then combines with one
    vectorized butterfly over all k at once. n must be a power of two.
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"fft_recursive needs a power-of-two length, got {n}")
    return _fft_rec(x)


def _fft_rec(x):
    n = x.shape[-1]
    if n == 1:
        return x.copy()
    even = _fft_rec(x[..., 0::2])
    odd = _fft_rec(x[..., 1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    t = twiddle * odd
    return np.concatenate([even + t, even - t], axis=-1)


# ---------------------------------------------------------------------------
# iterative radix-2 FFT
# ---------------------------------------------------------------------------

def fft_iterative(x):
    """
    Bottom-up radix-2 Cooley-Tukey on the last axis of `x` (so a 2D array
    is transformed row by row in one call, which is how fft2 uses it).

    Step 1: reorder the input into bit-reversed order. After that, the
    length-2 DFTs the recursion would have bottomed out in are sitting
    in adjacent pairs, the length-4 DFTs in adjacent blocks of four,
    and so on.
    Step 2: for size = 2, 4, 8, ..., n, view the array as blocks of
    `size`; in each block the first half holds an even-part DFT and the
    second half an odd-part DFT of length size/2. One butterfly pass
    combines them into a length-`size` DFT. Every block at a given size
    is processed in one vectorized numpy expression.

    log2(n) passes, O(n) work each, O(n log n) total.
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"fft_iterative needs a power-of-two length, got {n}")
    if n == 1:
        return x.copy()

    lead = x.shape[:-1]
    a = x[..., bit_reverse_indices(n)]

    size = 2
    while size <= n:
        half = size // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half) / size)
        blocks = a.reshape(lead + (n // size, size))
        even = blocks[..., :half]
        odd = blocks[..., half:] * twiddle
        a = np.concatenate([even + odd, even - odd], axis=-1).reshape(lead + (n,))
        size *= 2

    return a


# ---------------------------------------------------------------------------
# Bluestein (chirp-z) for arbitrary lengths
# ---------------------------------------------------------------------------

def fft_bluestein(x):
    """
    DFT of any length n in O(n log n), via Bluestein's identity
    j*k = (j^2 + k^2 - (k - j)^2) / 2, which turns the DFT into

        X[k] = w[k] * sum_j (x[j] * w[j]) * conj(w[k - j]),
        w[j] = exp(-pi*i*j^2 / n)

    i.e. a convolution of (x * w) with conj(w). That convolution is
    evaluated with power-of-two FFTs of length m >= 2n - 1.

    j^2 gets big quickly, and exp() of a huge angle loses accuracy, so
    j^2 is reduced mod 2n first (the chirp has period 2n in j^2).
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    if n == 1:
        return x.copy()

    j = np.arange(n, dtype=np.int64)
    chirp = np.exp(-1j * np.pi * ((j * j) % (2 * n)) / n)

    m = next_power_of_two(2 * n - 1)
    lead = x.shape[:-1]

    a = np.zeros(lead + (m,), dtype=np.complex128)
    a[..., :n] = x * chirp

    b = np.zeros(m, dtype=np.complex128)
    b[:n] = np.conj(chirp)
    b[m - n + 1:] = np.conj(chirp[1:][::-1])  # negative indices wrap around

    conv = _ifft_pow2(fft_iterative(a) * fft_iterative(b))
    return chirp * conv[..., :n]


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def fft(x):
    """
    Forward DFT along the last axis, any length. Powers of two use the
    iterative radix-2 FFT, everything else uses Bluestein.
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    if is_power_of_two(n):
        return fft_iterative(x)
    return fft_bluestein(x)


def _ifft_pow2(X):
    n = X.shape[-1]
    return np.conj(fft_iterative(np.conj(X))) / n


def ifft(X):
    """
    Inverse DFT along the last axis:  x[j] = (1/n) sum_k X[k] exp(+2*pi*i*j*k/n).

    Implemented with the conjugation trick, ifft(X) = conj(fft(conj(X))) / n,
    so the inverse reuses exactly the same forward code path (and the
    same accuracy) instead of duplicating it with the sign flipped.
    """
    X = _as_complex(X)
    n = X.shape[-1]
    if n == 0:
        return X.copy()
    return np.conj(fft(np.conj(X))) / n


def ifft_recursive(X):
    """Inverse of fft_recursive, same conjugation trick."""
    X = _as_complex(X)
    return np.conj(fft_recursive(np.conj(X))) / X.shape[-1]


def fft2(x):
    """2D DFT: a 1D FFT over every row, then over every column."""
    x = _as_complex(x)
    rows = fft(x)
    return fft(rows.swapaxes(-1, -2)).swapaxes(-1, -2)


def ifft2(X):
    """Inverse 2D DFT."""
    X = _as_complex(X)
    rows = ifft(X)
    return ifft(rows.swapaxes(-1, -2)).swapaxes(-1, -2)


def fftfreq(n, d=1.0):
    """
    Frequency (in cycles per unit of d) for each FFT bin of a length-n
    transform with sample spacing d: bins 0..ceil(n/2)-1 are the
    non-negative frequencies, the rest are negative, same ordering as
    the output of fft().
    """
    k = np.arange(n)
    k = np.where(k < (n + 1) // 2, k, k - n)
    return k / (n * d)


def rfftfreq(n, d=1.0):
    """Non-negative frequencies only, bins 0..n//2, for one-sided spectra."""
    return np.arange(n // 2 + 1) / (n * d)


def fftshift(X, axes=None):
    """Move the zero-frequency bin to the centre (for plotting)."""
    X = np.asarray(X)
    if axes is None:
        axes = tuple(range(X.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    for ax in axes:
        X = np.roll(X, X.shape[ax] // 2, axis=ax)
    return X


def ifftshift(X, axes=None):
    """Undo fftshift."""
    X = np.asarray(X)
    if axes is None:
        axes = tuple(range(X.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    for ax in axes:
        X = np.roll(X, -(X.shape[ax] // 2), axis=ax)
    return X


# ---------------------------------------------------------------------------
# convolution
# ---------------------------------------------------------------------------

def _maybe_real(result, *inputs):
    """If every input was real, drop the (round-off sized) imaginary part."""
    if all(not np.iscomplexobj(np.asarray(v)) for v in inputs):
        return result.real
    return result


def circular_convolve(a, b):
    """
    Circular (cyclic) convolution of two equal-length sequences:

        (a (*) b)[k] = sum_j a[j] * b[(k - j) mod n]

    By the convolution theorem this is ifft(fft(a) * fft(b)), O(n log n)
    instead of O(n^2). Indices wrap around, which is exactly what the
    DFT assumes: it treats every finite sequence as one period of a
    periodic signal.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("circular convolution needs equal-length inputs")
    out = ifft(fft(a) * fft(b))
    return _maybe_real(out, a, b)


def linear_convolve(a, b):
    """
    Ordinary (linear, non-wrapping) convolution, output length
    len(a) + len(b) - 1, same result as np.convolve(a, b, "full").

    Circular convolution wraps the tail of the result back onto the
    start. Zero-padding both inputs to length L >= len(a) + len(b) - 1
    leaves enough empty room that nothing wraps, so the circular result
    of the padded inputs, truncated to the first len(a)+len(b)-1
    samples, is the linear convolution. L is rounded up to a power of
    two so the fast radix-2 path is used.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("linear_convolve expects 1D inputs")
    if a.size == 0 or b.size == 0:
        return np.zeros(0)
    out_len = a.size + b.size - 1
    L = next_power_of_two(out_len)
    A = fft(np.concatenate([a, np.zeros(L - a.size)]))
    B = fft(np.concatenate([b, np.zeros(L - b.size)]))
    out = ifft(A * B)[:out_len]
    return _maybe_real(out, a, b)


def direct_convolve(a, b):
    """O(n*m) linear convolution from the definition, the reference the
    FFT version is tested and benchmarked against."""
    a = np.asarray(a, dtype=np.result_type(a, b, np.float64))
    b = np.asarray(b, dtype=a.dtype)
    n, m = a.size, b.size
    out = np.zeros(n + m - 1, dtype=a.dtype)
    for j in range(m):  # vectorized over a, loop over the shorter kernel
        out[j:j + n] += a * b[j]
    return out


def convolve2d(image, kernel, mode="same"):
    """
    2D linear convolution through the 2D FFT. Both arrays are zero-padded
    to (power-of-two) sizes at least H + kh - 1 by W + kw - 1 so nothing
    wraps around. mode="full" returns the whole result, mode="same"
    returns the centre part the same size as `image`.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    H, W = image.shape
    kh, kw = kernel.shape
    fh, fw = H + kh - 1, W + kw - 1
    Ph, Pw = next_power_of_two(fh), next_power_of_two(fw)

    a = np.zeros((Ph, Pw), dtype=np.complex128)
    a[:H, :W] = image
    k = np.zeros((Ph, Pw), dtype=np.complex128)
    k[:kh, :kw] = kernel

    full = ifft2(fft2(a) * fft2(k))[:fh, :fw]
    full = _maybe_real(full, image, kernel)
    if mode == "full":
        return full
    if mode == "same":
        top, left = (kh - 1) // 2, (kw - 1) // 2
        return full[top:top + H, left:left + W]
    raise ValueError("mode must be 'full' or 'same'")
