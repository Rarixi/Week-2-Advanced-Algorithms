"""
filters.py
-------------
Part 3.1 of the FFTAnalyzer project. Frequency-domain filters (low-pass,
high-pass, band-pass, band-stop) and the window functions used both for
spectral analysis and for designing FIR filters (Hann, Hamming,
Blackman, rectangular).

Two ways of filtering live here, and the difference is worth knowing:

1. fft_filter: FFT the whole signal, multiply each bin by a frequency
   response H(f), inverse FFT. With an ideal "brick-wall" H (1 in the
   passband, 0 outside) this removes the stop band exactly *for
   frequencies that land exactly on a bin*. But the DFT treats the
   signal as periodic, so anything that doesn't complete a whole number
   of cycles leaks across bins, and a hard cutoff then rings (Gibbs
   phenomenon) around transients and at the signal's edges. The
   `transition` parameter replaces the hard edge with a raised-cosine
   taper, which trades a little selectivity for much less ringing.

2. fir_filter: build a finite impulse response by windowing an ideal
   sinc ("windowed-sinc" design), then apply it by linear convolution
   (via FFT, from core/fft.py). The window is what controls the trade
   off: a rectangular window gives the narrowest transition but only
   ~21 dB of minimum stop-band attenuation (textbook figure); Hamming
   ~53 dB; Blackman ~74 dB with a transition about three times wider.
   Measured on a 101-tap, 100 Hz low-pass at fs = 1 kHz, the worst
   stop-band leakage beyond 160 Hz came out at -37, -61 and -83 dB
   respectively (further from the edge than the textbook minimum, so
   lower); the test suite checks that ordering.
"""

import numpy as np

from ..core.fft import fft, fftfreq, ifft, linear_convolve


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def _window_axis(n, symmetric):
    if n <= 0:
        raise ValueError("window length must be positive")
    if n == 1:
        return np.zeros(1), 1
    # symmetric windows (filter design) span 0..n-1 over the period n-1;
    # periodic windows (spectral analysis, STFT) use period n so that
    # overlapping copies tile cleanly
    return np.arange(n), (n - 1 if symmetric else n)


def rectangular(n, symmetric=True):
    """All ones: no tapering at all. Narrowest main lobe, worst leakage."""
    return np.ones(n)


def hanning(n, symmetric=True):
    """Hann window, 0.5 - 0.5 cos(2 pi k / N). Zero at both ends;
    side lobes ~-31 dB and falling off at 18 dB/octave."""
    k, N = _window_axis(n, symmetric)
    if n == 1:
        return np.ones(1)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / N)


hann = hanning


def hamming(n, symmetric=True):
    """Hamming, 0.54 - 0.46 cos(2 pi k / N). Doesn't reach zero at the ends,
    which lets it cancel the first side lobe: ~-43 dB, but slow roll-off."""
    k, N = _window_axis(n, symmetric)
    if n == 1:
        return np.ones(1)
    return 0.54 - 0.46 * np.cos(2 * np.pi * k / N)


def blackman(n, symmetric=True):
    """Blackman, 0.42 - 0.5 cos(2 pi k/N) + 0.08 cos(4 pi k/N). Side lobes
    ~-58 dB at the cost of a main lobe ~1.5x wider than Hann's."""
    k, N = _window_axis(n, symmetric)
    if n == 1:
        return np.ones(1)
    w = 0.42 - 0.5 * np.cos(2 * np.pi * k / N) + 0.08 * np.cos(4 * np.pi * k / N)
    w[np.abs(w) < 1e-15] = 0.0  # endpoints are exactly zero, clean up -1e-17
    return w


WINDOWS = {
    "rectangular": rectangular,
    "boxcar": rectangular,
    "hann": hanning,
    "hanning": hanning,
    "hamming": hamming,
    "blackman": blackman,
}


def get_window(name, n, symmetric=True):
    """Look up a window by name ('hann', 'hamming', 'blackman', 'rectangular')."""
    try:
        return WINDOWS[name.lower()](n, symmetric=symmetric)
    except KeyError:
        raise ValueError(f"unknown window {name!r}, choose from {sorted(WINDOWS)}") from None


def coherent_gain(w):
    """Mean of the window: the factor a windowed sinusoid's peak is scaled by."""
    return float(np.mean(w))


def equivalent_noise_bandwidth(w):
    """ENBW in bins: n * sum(w^2) / sum(w)^2 (1.0 rectangular, 1.5 Hann, ~1.36 Hamming, ~1.73 Blackman)."""
    w = np.asarray(w, dtype=np.float64)
    return float(w.size * np.sum(w * w) / np.sum(w) ** 2)


# ---------------------------------------------------------------------------
# frequency responses
# ---------------------------------------------------------------------------

def _taper(f, edge, transition, rising):
    """Raised-cosine step from 0 to 1 centred on `edge`, `transition` Hz wide."""
    if transition <= 0:
        return (f >= edge).astype(float) if rising else (f <= edge).astype(float)
    lo, hi = edge - transition / 2.0, edge + transition / 2.0
    ramp = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    up = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return up if rising else 1.0 - up


def frequency_response(freqs, kind, cutoff, transition=0.0):
    """
    Real, zero-phase response H(|f|) for kind in
    {'lowpass', 'highpass', 'bandpass', 'bandstop'}. `cutoff` is one
    frequency for low/high-pass and a (low, high) pair for band filters.
    Zero phase (real H) means the filter delays nothing, which is one
    advantage of filtering in the frequency domain over a causal filter.
    """
    f = np.abs(np.asarray(freqs, dtype=np.float64))
    if kind == "lowpass":
        return _taper(f, float(cutoff), transition, rising=False)
    if kind == "highpass":
        return _taper(f, float(cutoff), transition, rising=True)
    lo, hi = cutoff
    if lo >= hi:
        raise ValueError("band edges must satisfy low < high")
    band = _taper(f, lo, transition, rising=True) * _taper(f, hi, transition, rising=False)
    if kind == "bandpass":
        return band
    if kind == "bandstop":
        return 1.0 - band
    raise ValueError("kind must be lowpass, highpass, bandpass or bandstop")


def fft_filter(x, sample_rate, kind, cutoff, transition=0.0):
    """
    Filter a real signal by multiplying its spectrum by H(f). Because H
    depends only on |f|, positive and negative frequency bins get the same
    gain, the spectrum stays conjugate-symmetric, and the output is real.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    f = fftfreq(n, 1.0 / sample_rate)
    H = frequency_response(f, kind, cutoff, transition)
    return ifft(fft(x) * H).real


def low_pass(x, sample_rate, cutoff, transition=0.0):
    """Keep |f| <= cutoff."""
    return fft_filter(x, sample_rate, "lowpass", cutoff, transition)


def high_pass(x, sample_rate, cutoff, transition=0.0):
    """Keep |f| >= cutoff."""
    return fft_filter(x, sample_rate, "highpass", cutoff, transition)


def band_pass(x, sample_rate, low, high, transition=0.0):
    """Keep low <= |f| <= high."""
    return fft_filter(x, sample_rate, "bandpass", (low, high), transition)


def band_stop(x, sample_rate, low, high, transition=0.0):
    """Remove low <= |f| <= high (a notch, e.g. for 50/60 Hz mains hum)."""
    return fft_filter(x, sample_rate, "bandstop", (low, high), transition)


# ---------------------------------------------------------------------------
# windowed-sinc FIR design
# ---------------------------------------------------------------------------

def _sinc_lowpass(num_taps, fc):
    """Ideal low-pass impulse response 2 fc sinc(2 fc (k - M/2)), fc in cycles/sample."""
    m = np.arange(num_taps) - (num_taps - 1) / 2.0
    return 2 * fc * np.sinc(2 * fc * m)


def design_fir(num_taps, cutoff, sample_rate, kind="lowpass", window="hamming"):
    """
    Windowed-sinc FIR design. num_taps should be odd (a type-I linear
    phase filter, which can realize all four kinds). The ideal sinc is
    infinitely long; truncating it is the same as multiplying by a
    rectangular window, which is where the ripple comes from, so it's
    multiplied by a smoother window instead.

    High-pass, band-pass and band-stop are built from low-passes by
    spectral inversion (delta - lowpass) and differences of low-passes.
    Returns the taps, normalized to unit gain in the passband.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd")
    nyq = sample_rate / 2.0
    w = get_window(window, num_taps, symmetric=True)
    centre = num_taps // 2
    delta = np.zeros(num_taps)
    delta[centre] = 1.0

    def lp(fc_hz):
        if not 0 < fc_hz < nyq:
            raise ValueError("cutoff must be between 0 and the Nyquist frequency")
        h = _sinc_lowpass(num_taps, fc_hz / sample_rate) * w
        return h / np.sum(h)  # unit DC gain

    if kind == "lowpass":
        return lp(float(cutoff))
    if kind == "highpass":
        return delta - lp(float(cutoff))
    lo, hi = cutoff
    if kind == "bandpass":
        return lp(hi) - lp(lo)
    if kind == "bandstop":
        return delta - (lp(hi) - lp(lo))
    raise ValueError("kind must be lowpass, highpass, bandpass or bandstop")


def fir_filter(x, taps, mode="same"):
    """
    Apply an FIR filter by FFT linear convolution. mode="same" trims the
    (len(taps) - 1) / 2 sample group delay so the output lines up with
    the input; mode="full" returns the whole convolution.
    """
    x = np.asarray(x, dtype=np.float64)
    y = linear_convolve(x, taps)
    if mode == "full":
        return y
    d = (len(taps) - 1) // 2
    return y[d:d + x.size]


def fir_frequency_response(taps, n_fft=8192, sample_rate=1.0):
    """(frequencies, |H|) of an FIR filter, by zero-padded FFT of its taps."""
    h = np.zeros(n_fft)
    h[:len(taps)] = taps
    H = fft(h)[: n_fft // 2 + 1]
    return np.arange(n_fft // 2 + 1) * sample_rate / n_fft, np.abs(H)


def moving_average(x, width):
    """Box-car smoothing via FFT convolution, edges handled by 'same' trimming."""
    if width < 1:
        raise ValueError("width must be >= 1")
    taps = np.ones(int(width)) / int(width)
    x = np.asarray(x, dtype=np.float64)
    y = linear_convolve(x, taps)
    d = (int(width) - 1) // 2
    return y[d:d + x.size]
