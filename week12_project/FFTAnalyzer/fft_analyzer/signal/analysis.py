"""
analysis.py
--------------
Part 3.3 of the FFTAnalyzer project. Spectral analysis on top of the
project's FFT: amplitude spectra, the periodogram, Welch's averaged
power spectral density, the short-time Fourier transform / spectrogram,
peak frequency detection (with sub-bin accuracy), and autocorrelation
computed in the frequency domain.

Scaling conventions matter here and are easy to get wrong, so they're
stated once:

  * amplitude_spectrum: one-sided, scaled so a sine of amplitude A shows
    a peak of height A (after correcting for the window's coherent gain).
  * periodogram / welch_psd: one-sided power spectral density in
    units^2 / Hz, scaled so that integrating the PSD over frequency
    (sum(psd) * df) gives the signal's mean power, Parseval's theorem.
    The tests check that identity directly.

The periodogram is an inconsistent estimator: its variance doesn't go
down as the signal gets longer, you just get more, equally noisy bins.
Welch's method fixes that by averaging the periodograms of overlapping,
windowed segments, trading frequency resolution for variance.
"""

import numpy as np

from ..core.fft import fft, ifft, next_power_of_two, rfftfreq
from .filters import get_window


def _one_sided(X, n):
    """Keep bins 0..n//2 of a length-n FFT."""
    return X[..., : n // 2 + 1]


def _double_interior(P, n):
    """
    One-sided power: every bin except DC (and Nyquist, for even n) has a
    mirror-image negative-frequency twin holding equal power, so it's
    counted twice.
    """
    P = P.copy()
    if n % 2 == 0:
        P[..., 1:-1] *= 2
    else:
        P[..., 1:] *= 2
    return P


def amplitude_spectrum(x, sample_rate, window="hann"):
    """
    (freqs, amplitudes): one-sided amplitude spectrum. A pure tone of
    amplitude A at a bin-centred frequency comes out with a peak of A.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    w = get_window(window, n, symmetric=False)
    X = _one_sided(fft(x * w), n)
    amp = np.abs(X) / np.sum(w)
    amp = _double_interior(amp, n)
    return rfftfreq(n, 1.0 / sample_rate), amp


def periodogram(x, sample_rate, window="rectangular", detrend=True):
    """
    (freqs, psd) with psd[k] = |FFT(x w)[k]|^2 / (fs * sum(w^2)),
    one-sided. Dividing by sum(w^2) rather than n makes the PSD's
    integral equal mean power for any window, not just the rectangular.
    """
    x = np.asarray(x, dtype=np.float64)
    if detrend:
        x = x - x.mean()
    n = x.size
    w = get_window(window, n, symmetric=False)
    X = _one_sided(fft(x * w), n)
    psd = (np.abs(X) ** 2) / (sample_rate * np.sum(w * w))
    return rfftfreq(n, 1.0 / sample_rate), _double_interior(psd, n)


def welch_psd(x, sample_rate, segment_length=256, overlap=0.5, window="hann", detrend=True):
    """
    Welch's method: split into segments of `segment_length` overlapping by
    `overlap` (fraction), window each, average their periodograms. With
    50% overlap and a Hann window, K segments cut the variance by a
    factor of roughly K compared with one periodogram.
    """
    x = np.asarray(x, dtype=np.float64)
    L = int(min(segment_length, x.size))
    step = max(1, int(round(L * (1 - overlap))))
    w = get_window(window, L, symmetric=False)
    starts = range(0, x.size - L + 1, step)
    segs = np.array([x[s:s + L] for s in starts])
    if detrend:
        segs = segs - segs.mean(axis=1, keepdims=True)
    X = _one_sided(fft(segs * w), L)
    psd = (np.abs(X) ** 2) / (sample_rate * np.sum(w * w))
    psd = _double_interior(psd, L).mean(axis=0)
    return rfftfreq(L, 1.0 / sample_rate), psd


def power_spectral_density(x, sample_rate, method="welch", **kwargs):
    """Dispatch: method='welch' (default) or 'periodogram'."""
    if method == "welch":
        return welch_psd(x, sample_rate, **kwargs)
    if method == "periodogram":
        return periodogram(x, sample_rate, **kwargs)
    raise ValueError("method must be 'welch' or 'periodogram'")


def stft(x, sample_rate, window_length=256, hop=None, window="hann", n_fft=None):
    """
    Short-time Fourier transform: slide a window along the signal, FFT
    each frame. Returns (freqs, times, Z) with Z of shape
    (n_freqs, n_frames), complex, one-sided. `times` are frame centres.

    All frames go through one batched call to the iterative FFT, which
    operates on the last axis of a 2D array, rather than a Python loop
    of separate FFTs.
    """
    x = np.asarray(x, dtype=np.float64)
    L = int(window_length)
    hop = L // 4 if hop is None else int(hop)
    n_fft = next_power_of_two(L) if n_fft is None else int(n_fft)
    if x.size < L:
        x = np.concatenate([x, np.zeros(L - x.size)])
    w = get_window(window, L, symmetric=False)
    starts = np.arange(0, x.size - L + 1, hop)
    frames = np.stack([x[s:s + L] * w for s in starts])
    if n_fft > L:
        frames = np.concatenate([frames, np.zeros((frames.shape[0], n_fft - L))], axis=1)
    Z = _one_sided(fft(frames), n_fft).T
    times = (starts + L / 2.0) / sample_rate
    return rfftfreq(n_fft, 1.0 / sample_rate), times, Z


def spectrogram(x, sample_rate, window_length=256, hop=None, window="hann", n_fft=None, db=True):
    """
    (freqs, times, S): power per time-frequency cell, |STFT|^2, in dB by
    default (10 log10, floored at -200 dB to avoid log(0)).

    The window length is the time/frequency resolution trade-off
    (the uncertainty principle for the DFT): frequency resolution is
    fs / window_length Hz, time resolution is window_length / fs seconds,
    and you can't make both small at once.
    """
    f, t, Z = stft(x, sample_rate, window_length, hop, window, n_fft)
    S = np.abs(Z) ** 2
    if db:
        S = 10 * np.log10(np.maximum(S, 1e-20))
    return f, t, S


def istft(Z, sample_rate, window_length=256, hop=None, window="hann", n_fft=None, length=None):
    """
    Inverse STFT by weighted overlap-add: inverse FFT each frame, multiply
    by the window again, add into place, and divide by the summed squared
    window. Exact reconstruction (up to round-off) wherever at least one
    window is non-zero. Used for time-varying filtering.
    """
    L = int(window_length)
    hop = L // 4 if hop is None else int(hop)
    n_fft = next_power_of_two(L) if n_fft is None else int(n_fft)
    w = get_window(window, L, symmetric=False)
    n_frames = Z.shape[1]
    # rebuild the full conjugate-symmetric spectrum from the one-sided one
    half = Z.T
    if n_fft % 2 == 0:
        full = np.concatenate([half, np.conj(half[:, -2:0:-1])], axis=1)
    else:
        full = np.concatenate([half, np.conj(half[:, -1:0:-1])], axis=1)
    frames = ifft(full).real[:, :L]
    total = (n_frames - 1) * hop + L
    out = np.zeros(total)
    norm = np.zeros(total)
    for i in range(n_frames):
        s = i * hop
        out[s:s + L] += frames[i] * w
        norm[s:s + L] += w * w
    # where the summed window is tiny (the first/last few samples, under
    # a single window's tapered tail) dividing would blow up round-off,
    # so those samples are left at zero; callers that need the edges
    # should pad by window_length before the STFT (see AudioProcessor)
    nz = norm > 1e-3 * norm.max()
    out[nz] /= norm[nz]
    out[~nz] = 0.0
    if length is not None:
        out = out[:length] if out.size >= length else np.concatenate([out, np.zeros(length - out.size)])
    return out


def _parabolic_offset(y_left, y_mid, y_right):
    """Vertex offset (in bins, within +/-0.5) of the parabola through three points."""
    denom = y_left - 2 * y_mid + y_right
    if denom == 0:
        return 0.0
    return 0.5 * (y_left - y_right) / denom


def find_peaks(freqs, spectrum, num_peaks=5, min_height=None, min_distance_hz=0.0, refine=True):
    """
    Local maxima of a spectrum, highest first. A bin is a peak if it's
    strictly greater than its left neighbour and >= its right one.
    Peaks closer than `min_distance_hz` to a higher peak are dropped.

    With refine=True the frequency is interpolated between bins by
    fitting a parabola through the log-magnitude of the peak and its two
    neighbours. For a Gaussian-ish window main lobe that's nearly exact,
    and it gets the frequency to a small fraction of a bin instead of
    being stuck at bin resolution (fs / n).

    Returns a list of (frequency, height).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(spectrum, dtype=np.float64)
    if s.size < 3:
        i = int(np.argmax(s))
        return [(float(freqs[i]), float(s[i]))]
    is_peak = np.zeros(s.size, dtype=bool)
    is_peak[1:-1] = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    idx = np.flatnonzero(is_peak)
    if min_height is not None:
        idx = idx[s[idx] >= min_height]
    idx = idx[np.argsort(-s[idx])]

    df = freqs[1] - freqs[0]
    chosen = []
    for i in idx:
        f = freqs[i]
        h = s[i]
        if refine and s[i - 1] > 0 and s[i] > 0 and s[i + 1] > 0:
            a, b, c = np.log(s[i - 1]), np.log(s[i]), np.log(s[i + 1])
            p = _parabolic_offset(a, b, c)
            f = freqs[i] + p * df
            h = float(np.exp(b - 0.25 * (a - c) * p))
        if all(abs(f - g) >= min_distance_hz for g, _ in chosen):
            chosen.append((float(f), float(h)))
        if len(chosen) >= num_peaks:
            break
    return chosen


def peak_frequency(x, sample_rate, window="hann", refine=True, zero_pad=1):
    """
    The single dominant frequency of x, ignoring DC. zero_pad > 1 pads the
    FFT to zero_pad * n samples, which interpolates the spectrum (it does
    *not* add resolution: two tones closer than ~fs/n still merge).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    w = get_window(window, n, symmetric=False)
    m = next_power_of_two(n * max(1, int(zero_pad)))
    buf = np.zeros(m)
    buf[:n] = x * w
    mag = np.abs(_one_sided(fft(buf), m))
    freqs = rfftfreq(m, 1.0 / sample_rate)
    mag[0] = 0.0
    peaks = find_peaks(freqs, mag, num_peaks=1, refine=refine)
    return peaks[0][0]


def autocorrelation(x, normalize=True, max_lag=None):
    """
    Autocorrelation r[k] = sum_t (x[t] - mean)(x[t+k] - mean) for lags
    k = 0..n-1, via the Wiener-Khinchin theorem: autocorrelation is the
    inverse FFT of the power spectrum |X|^2. Zero-padding to >= 2n - 1
    prevents circular wrap-around (same reason as linear convolution).
    O(n log n) instead of O(n^2). normalize=True divides by r[0], so
    r[0] = 1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    x = x - x.mean()
    m = next_power_of_two(2 * n - 1)
    buf = np.zeros(m)
    buf[:n] = x
    X = fft(buf)
    r = ifft(X * np.conj(X)).real[:n]
    if normalize:
        r = r / r[0] if r[0] != 0 else r
    if max_lag is not None:
        r = r[: max_lag + 1]
    return r


def autocorrelation_direct(x, normalize=True):
    """O(n^2) reference autocorrelation for the tests."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    r = np.array([np.dot(x[: n - k], x[k:]) for k in range(n)])
    if normalize and r[0] != 0:
        r = r / r[0]
    return r


def spectral_centroid(freqs, spectrum):
    """Power-weighted mean frequency: a "brightness" measure."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    total = np.sum(spectrum)
    return float(np.sum(freqs * spectrum) / total) if total > 0 else 0.0


def band_power(freqs, psd, low, high):
    """Integrated PSD between low and high Hz (rectangle rule)."""
    freqs = np.asarray(freqs)
    df = freqs[1] - freqs[0]
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(np.asarray(psd)[mask]) * df)
