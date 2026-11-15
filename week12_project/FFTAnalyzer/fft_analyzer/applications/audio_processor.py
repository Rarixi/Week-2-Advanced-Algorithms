"""
audio_processor.py
---------------------
Part 5.1 of the FFTAnalyzer project. An audio workflow built on the
toolkit: load a WAV file, filter it in the frequency domain (low / high
/ band-pass, band-stop for hum), reduce broadband noise by spectral
subtraction on the STFT, visualize the spectrogram before and after,
and write the result back out as a WAV.

WAV I/O uses only the standard library's `wave` module (8, 16, 24 and
32-bit integer PCM), so no audio library is needed. Samples are held
internally as float64 in [-1, 1], one row per channel.

Typical use:

    ap = AudioProcessor.load("speech.wav")
    ap.band_stop(55, 65).high_pass(80).spectral_denoise()
    ap.plot_spectrogram("after.png")
    ap.save("cleaned.wav")
"""

import wave

import numpy as np

from ..signal import analysis, filters
from ..signal.generators import add_noise


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def read_wav(path):
    """
    Returns (samples, sample_rate) with samples float64 in [-1, 1] and
    shape (channels, n). Handles 8-bit (unsigned), 16, 24 and 32-bit
    (signed) PCM.
    """
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        fs = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float64) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")
    return data.reshape(-1, nch).T.copy(), fs


def write_wav(path, samples, sample_rate, sample_width=2):
    """
    Write float samples in [-1, 1] (shape (n,) or (channels, n)) as integer
    PCM. Values outside [-1, 1] are clipped rather than allowed to wrap
    around, which would sound like loud clicks.
    """
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    x = np.clip(x, -1.0, 1.0)
    inter = x.T.ravel()
    if sample_width == 2:
        raw = np.round(inter * 32767.0).astype("<i2").tobytes()
    elif sample_width == 1:
        raw = np.round(inter * 127.0 + 128.0).astype(np.uint8).tobytes()
    elif sample_width == 4:
        raw = np.round(inter * 2147483647.0).astype("<i4").tobytes()
    else:
        raise ValueError("sample_width must be 1, 2 or 4")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(sample_width)
        w.setframerate(int(sample_rate))
        w.writeframes(raw)


# ---------------------------------------------------------------------------
# processor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    Holds one clip (float samples, shape (channels, n)) and applies
    processing steps in place; every step returns self so calls chain.
    `history` records each step for the plots and the report.
    """

    def __init__(self, samples, sample_rate):
        x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
        self.samples = x
        self.original = x.copy()
        self.sample_rate = int(sample_rate)
        self.history = []

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, path):
        x, fs = read_wav(path)
        return cls(x, fs)

    @classmethod
    def from_array(cls, x, sample_rate):
        return cls(x, sample_rate)

    # -- properties ------------------------------------------------------

    @property
    def channels(self):
        return self.samples.shape[0]

    @property
    def duration(self):
        return self.samples.shape[1] / self.sample_rate

    def mono(self):
        """Channel average as a 1D array (for analysis and plots)."""
        return self.samples.mean(axis=0)

    # -- processing ------------------------------------------------------

    def _apply(self, func, label):
        self.samples = np.stack([func(ch) for ch in self.samples])
        self.history.append(label)
        return self

    def low_pass(self, cutoff, transition=0.0):
        fs = self.sample_rate
        return self._apply(lambda c: filters.low_pass(c, fs, cutoff, transition),
                           f"low-pass {cutoff} Hz")

    def high_pass(self, cutoff, transition=0.0):
        fs = self.sample_rate
        return self._apply(lambda c: filters.high_pass(c, fs, cutoff, transition),
                           f"high-pass {cutoff} Hz")

    def band_pass(self, low, high, transition=0.0):
        fs = self.sample_rate
        return self._apply(lambda c: filters.band_pass(c, fs, low, high, transition),
                           f"band-pass {low}-{high} Hz")

    def band_stop(self, low, high, transition=0.0):
        fs = self.sample_rate
        return self._apply(lambda c: filters.band_stop(c, fs, low, high, transition),
                           f"band-stop {low}-{high} Hz")

    def remove_hum(self, mains=60.0, harmonics=4, width=4.0):
        """Notch out mains hum and its first few harmonics (60 Hz US, 50 Hz EU)."""
        for k in range(1, harmonics + 1):
            f0 = mains * k
            if f0 + width < self.sample_rate / 2:
                self.band_stop(f0 - width / 2, f0 + width / 2)
        return self

    def fir(self, num_taps, cutoff, kind="lowpass", window="hamming"):
        """Windowed-sinc FIR filter applied by FFT convolution."""
        taps = filters.design_fir(num_taps, cutoff, self.sample_rate, kind, window)
        return self._apply(lambda c: filters.fir_filter(c, taps), f"FIR {kind} {cutoff} ({window})")

    def spectral_denoise(self, window_length=1024, noise_percentile=30, over_subtract=1.5,
                         floor=0.1):
        """
        Spectral subtraction. Compute the STFT; estimate the noise power in
        each frequency bin as a low percentile of that bin's power over
        time (speech and music come and go, steady background noise
        doesn't, so the quiet frames are mostly noise); subtract
        `over_subtract` times that estimate from every frame's power,
        keeping at least `floor` of the original magnitude (avoids the
        "musical noise" a hard zero creates); keep the original phase;
        resynthesize by overlap-add (analysis.istft).
        """
        fs = self.sample_rate
        L = window_length
        hop = L // 4

        def den(ch):
            # pad by a full window each side so every real sample is
            # covered by several overlapping windows (exact overlap-add)
            padded = np.concatenate([np.zeros(L), ch, np.zeros(L)])
            f, t, Z = analysis.stft(padded, fs, L, hop)
            P = np.abs(Z) ** 2
            noise = np.percentile(P, noise_percentile, axis=1, keepdims=True)
            gain = np.sqrt(np.maximum(1.0 - over_subtract * noise / np.maximum(P, 1e-20), floor ** 2))
            out = analysis.istft(Z * gain, fs, L, hop, length=padded.size)
            return out[L:L + ch.size]

        return self._apply(den, "spectral denoise")

    def normalize(self, peak=0.95):
        m = np.abs(self.samples).max()
        if m > 0:
            self.samples = self.samples * (peak / m)
        self.history.append(f"normalize to {peak}")
        return self

    def reset(self):
        self.samples = self.original.copy()
        self.history = []
        return self

    # -- analysis --------------------------------------------------------

    def spectrum(self, which="current"):
        x = self.mono() if which == "current" else self.original.mean(axis=0)
        return analysis.amplitude_spectrum(x, self.sample_rate)

    def dominant_frequencies(self, k=5, which="current"):
        f, A = self.spectrum(which)
        return analysis.find_peaks(f, A, num_peaks=k, min_distance_hz=5.0)

    def band_energy_change(self, low, high):
        """dB change in power inside [low, high] Hz, original -> current."""
        f0, p0 = analysis.welch_psd(self.original.mean(axis=0), self.sample_rate, 2048)
        f1, p1 = analysis.welch_psd(self.mono(), self.sample_rate, 2048)
        e0 = analysis.band_power(f0, p0, low, high)
        e1 = analysis.band_power(f1, p1, low, high)
        return 10 * np.log10(max(e1, 1e-30) / max(e0, 1e-30))

    # -- output ----------------------------------------------------------

    def save(self, path, sample_width=2):
        write_wav(path, self.samples, self.sample_rate, sample_width)
        return path

    def plot_spectrogram(self, save_path=None, window_length=1024, fmax=None, compare=True):
        """Spectrogram of the current audio, with the original above it if compare=True."""
        import matplotlib.pyplot as plt
        from ..visualization.plots import plot_spectrogram
        if compare:
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            plot_spectrogram(self.original.mean(axis=0), self.sample_rate, window_length,
                             title="Original", ax=a1, fmax=fmax)
            plot_spectrogram(self.mono(), self.sample_rate, window_length,
                             title=_wrap("Processed: " + (", ".join(self.history) or "none")), ax=a2, fmax=fmax)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=130, bbox_inches="tight")
            return fig
        return plot_spectrogram(self.mono(), self.sample_rate, window_length, save_path=save_path, fmax=fmax)

    def plot_spectra(self, save_path=None, xlim=None):
        """Original vs processed amplitude spectrum in dB."""
        import matplotlib.pyplot as plt
        from ..visualization.plots import COLORS, plot_spectrum
        fig, ax = plt.subplots(figsize=(9, 4))
        f0, A0 = self.spectrum("original")
        f1, A1 = self.spectrum("current")
        plot_spectrum(f0, A0, ax=ax, db=True, label="original", color=COLORS[5])
        plot_spectrum(f1, A1, ax=ax, db=True, label="processed", color=COLORS[0],
                      title="Spectrum before and after", xlim=xlim)
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig


def _wrap(text, width=95):
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def make_demo_clip(sample_rate=16_000, duration=3.0, seed=0):
    """
    A synthetic "recording" for the demos and tests, since no audio file
    ships with the project: a melody of three harmonic notes (220, 330,
    440 Hz fundamentals with decaying overtones, each fading in and out),
    plus 60 Hz mains hum with two harmonics, plus white noise at 15 dB
    SNR. Returns (clean, noisy, sample_rate).
    """
    t = np.arange(int(sample_rate * duration)) / sample_rate
    clean = np.zeros_like(t)
    notes = [(0.0, 1.0, 220.0), (1.0, 2.0, 330.0), (2.0, 3.0, 440.0)]
    for start, end, f0 in notes:
        seg = (t >= start) & (t < end)
        tt = t[seg] - start
        env = np.sin(np.pi * tt / (end - start)) ** 2
        tone = sum((0.6 / k) * np.sin(2 * np.pi * k * f0 * tt) for k in range(1, 5))
        clean[seg] += env * tone
    clean *= 0.5 / np.abs(clean).max()
    hum = 0.15 * np.sin(2 * np.pi * 60 * t) + 0.08 * np.sin(2 * np.pi * 120 * t) + 0.04 * np.sin(2 * np.pi * 180 * t)
    noisy = add_noise(clean + hum, 15, seed=seed)
    return clean, noisy, sample_rate
