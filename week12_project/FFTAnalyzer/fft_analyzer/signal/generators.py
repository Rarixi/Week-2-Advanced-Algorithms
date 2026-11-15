"""
generators.py
----------------
Part 3.2 of the FFTAnalyzer project. Test signal generators: pure sine
waves, multi-tone mixtures, linear and exponential chirps, and white
and pink noise. Every generator returns (t, x), the sample times and
the samples, so callers never have to recompute the time axis, and
every random generator takes a seed so results are reproducible.

Pink noise is the interesting one. White noise has a flat power
spectrum, equal power per hertz. Pink noise has power proportional to
1/f, equal power per *octave*, which is how a lot of natural signals
(and human hearing) behave. It's generated here the frequency-domain
way: take white noise, FFT it, scale bin k by 1/sqrt(f_k) (amplitude,
so power scales by 1/f), and inverse FFT. That's this project's own FFT
doing the work, and the test suite checks the slope of the result's
log-log spectrum comes out near -1.
"""

import numpy as np

from ..core.fft import fft, fftfreq, ifft


def time_axis(duration, sample_rate):
    """Sample times 0, 1/fs, 2/fs, ... for `duration` seconds."""
    n = int(round(duration * sample_rate))
    if n <= 0:
        raise ValueError("duration * sample_rate must be at least 1 sample")
    return np.arange(n) / float(sample_rate)


def sine_wave(freq, duration, sample_rate, amplitude=1.0, phase=0.0):
    """A * sin(2*pi*f*t + phase)."""
    t = time_axis(duration, sample_rate)
    return t, amplitude * np.sin(2 * np.pi * freq * t + phase)


def multi_tone(freqs, duration, sample_rate, amplitudes=None, phases=None):
    """Sum of sines, one per frequency, with optional per-tone amplitude and phase."""
    freqs = list(freqs)
    amplitudes = [1.0] * len(freqs) if amplitudes is None else list(amplitudes)
    phases = [0.0] * len(freqs) if phases is None else list(phases)
    if not (len(freqs) == len(amplitudes) == len(phases)):
        raise ValueError("freqs, amplitudes and phases must have the same length")
    t = time_axis(duration, sample_rate)
    x = np.zeros_like(t)
    for f, a, p in zip(freqs, amplitudes, phases):
        x += a * np.sin(2 * np.pi * f * t + p)
    return t, x


def chirp(f0, f1, duration, sample_rate, method="linear", amplitude=1.0):
    """
    A sweep from f0 Hz at t=0 to f1 Hz at t=duration.

    The phase is the integral of the instantaneous frequency, not
    f(t) * t: writing sin(2*pi*f(t)*t) is a common bug that makes the
    sweep run at twice the intended rate.
      linear:       f(t) = f0 + (f1 - f0) t / T
                    phase = 2*pi*(f0 t + (f1 - f0) t^2 / (2T))
      exponential:  f(t) = f0 * (f1/f0)^(t/T)
                    phase = 2*pi*f0*T*((f1/f0)^(t/T) - 1) / ln(f1/f0)
    """
    t = time_axis(duration, sample_rate)
    T = float(duration)
    if method == "linear":
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * T))
    elif method in ("exponential", "logarithmic"):
        if f0 <= 0 or f1 <= 0:
            raise ValueError("exponential chirp needs positive f0 and f1")
        if f0 == f1:
            phase = 2 * np.pi * f0 * t
        else:
            k = f1 / f0
            phase = 2 * np.pi * f0 * T * (k ** (t / T) - 1) / np.log(k)
    else:
        raise ValueError("method must be 'linear' or 'exponential'")
    return t, amplitude * np.sin(phase)


def chirp_instantaneous_frequency(f0, f1, t, duration, method="linear"):
    """The frequency the chirp is at, at time(s) t (used by the tests)."""
    t = np.asarray(t, dtype=np.float64)
    if method == "linear":
        return f0 + (f1 - f0) * t / duration
    return f0 * (f1 / f0) ** (t / duration)


def white_noise(duration, sample_rate, std=1.0, seed=None):
    """Independent Gaussian samples, flat expected power spectrum."""
    rng = np.random.default_rng(seed)
    t = time_axis(duration, sample_rate)
    return t, rng.normal(0.0, std, size=t.size)


def pink_noise(duration, sample_rate, std=1.0, seed=None):
    """
    1/f noise by spectral shaping: FFT of white noise, divide bin k's
    amplitude by sqrt(|f_k|), zero the DC bin (1/f blows up at 0), inverse
    FFT, then rescale to the requested standard deviation. Because each
    negative-frequency bin gets the same real scale factor as its
    positive mirror, the spectrum stays conjugate-symmetric and the
    result is real up to round-off.
    """
    rng = np.random.default_rng(seed)
    t = time_axis(duration, sample_rate)
    n = t.size
    W = fft(rng.standard_normal(n))
    f = np.abs(fftfreq(n, 1.0 / sample_rate))
    scale = np.zeros(n)
    scale[f > 0] = 1.0 / np.sqrt(f[f > 0])
    x = ifft(W * scale).real
    x -= x.mean()
    sd = x.std()
    if sd > 0:
        x *= std / sd
    return t, x


def square_wave(freq, duration, sample_rate, amplitude=1.0):
    """sign(sin(2 pi f t)): odd harmonics at f, 3f, 5f, ... with amplitude 4/(pi k)."""
    t = time_axis(duration, sample_rate)
    return t, amplitude * np.sign(np.sin(2 * np.pi * freq * t))


def add_noise(x, snr_db, seed=None):
    """Add white Gaussian noise to x at the given signal-to-noise ratio (dB)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    p_signal = np.mean(x * x)
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)


def snr_db(clean, noisy):
    """10 log10(signal power / error power)."""
    clean = np.asarray(clean, dtype=np.float64)
    err = np.asarray(noisy, dtype=np.float64) - clean
    pe = np.mean(err * err)
    if pe == 0:
        return np.inf
    return 10 * np.log10(np.mean(clean * clean) / pe)
