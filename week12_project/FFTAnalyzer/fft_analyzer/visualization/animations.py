"""
animations.py
----------------
Part 6.2 of the FFTAnalyzer project. Two animations, both saved as GIFs
with matplotlib's PillowWriter (falls back to saving a strip of frames as
a PNG if Pillow isn't installed):

  * animate_fft_process: the iterative FFT stage by stage. Frame 0 is
    the bit-reversed input; after stage s, every block of 2^s samples
    holds a finished DFT of length 2^s. The animation shows the
    magnitude of the working array after each butterfly pass, with the
    block boundaries drawn in, ending at the full spectrum.

  * animate_filtering: a low-pass filter whose cutoff sweeps downward.
    Top panel: the spectrum, with removed bins greyed out. Bottom panel:
    the filtered signal against the original, so you can watch the high
    harmonics disappear and the waveform smooth out.

The per-stage arrays come from fft_stages(), which replays the same
butterfly passes as core.fft.fft_iterative but records every
intermediate state; the tests check its last stage equals fft().
"""

import os

import matplotlib

if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from ..core.fft import bit_reverse_indices, fft, fftfreq, ifft, is_power_of_two
from .plots import COLORS


def fft_stages(x):
    """
    The working array after the bit-reversal permutation and after each
    of the log2(n) butterfly passes. stages[0] is the permuted input,
    stages[-1] equals fft(x).
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    if not is_power_of_two(n):
        raise ValueError("fft_stages needs a power-of-two length")
    a = x[bit_reverse_indices(n)]
    stages = [a.copy()]
    size = 2
    while size <= n:
        half = size // 2
        tw = np.exp(-2j * np.pi * np.arange(half) / size)
        blocks = a.reshape(n // size, size)
        even, odd = blocks[:, :half], blocks[:, half:] * tw
        a = np.concatenate([even + odd, even - odd], axis=1).ravel()
        stages.append(a.copy())
        size *= 2
    return stages


def _save_animation(anim, fig, save_path, fps, frames_fallback):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    try:
        import PIL  # noqa: F401
        anim.save(save_path, writer=animation.PillowWriter(fps=fps))
        return save_path
    except ImportError:
        # no Pillow: save a contact sheet of the frames instead
        png = os.path.splitext(save_path)[0] + "_frames.png"
        frames_fallback(png)
        return png
    finally:
        plt.close(fig)


def animate_fft_process(x=None, save_path="fft_process.gif", fps=1.5):
    """Animate the butterfly passes of a radix-2 FFT. Default input: a
    32-sample two-tone signal, small enough to see every sample."""
    if x is None:
        n = 32
        k = np.arange(n)
        x = np.sin(2 * np.pi * 3 * k / n) + 0.5 * np.sin(2 * np.pi * 9 * k / n)
    stages = fft_stages(x)
    n = len(stages[0])
    ymax = max(np.abs(s).max() for s in stages) * 1.1

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(np.arange(n), np.abs(stages[0]), color=COLORS[0], width=0.8)
    lines = []
    title = ax.set_title("", loc="left", fontsize=11)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-1, n)
    ax.set_xlabel("array index")
    ax.set_ylabel("|value|")
    ax.spines[["top", "right"]].set_visible(False)

    def draw(i):
        for b, h in zip(bars, np.abs(stages[i])):
            b.set_height(h)
            b.set_color(COLORS[0] if i < len(stages) - 1 else COLORS[2])
        for ln in lines:
            ln.remove()
        lines.clear()
        size = 2 ** i
        if 1 < size < n:
            for edge in range(size, n, size):
                lines.append(ax.axvline(edge - 0.5, color=COLORS[5], lw=0.8, ls="--"))
        if i == 0:
            title.set_text("Stage 0: input in bit-reversed order")
        elif i == len(stages) - 1:
            title.set_text(f"Stage {i}: one length-{n} DFT, the finished spectrum |X[k]|")
        else:
            title.set_text(f"Stage {i}: {n // size} independent DFTs of length {size}")
        return list(bars) + lines

    anim = animation.FuncAnimation(fig, draw, frames=len(stages), interval=1000 / fps, blit=False)

    def fallback(png):
        fig2, axes = plt.subplots(len(stages), 1, figsize=(8, 1.6 * len(stages)), sharex=True)
        for i, a in enumerate(axes):
            a.bar(np.arange(n), np.abs(stages[i]), color=COLORS[0])
            a.set_ylabel(f"stage {i}", fontsize=8)
        fig2.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig2)

    return _save_animation(anim, fig, save_path, fps, fallback)


def animate_filtering(x=None, sample_rate=1000, cutoffs=None, save_path="filtering.gif",
                      fps=4, show_seconds=0.05):
    """
    Animate a brick-wall low-pass as its cutoff sweeps down from Nyquist.
    Default input: a 50 Hz square wave (odd harmonics 50, 150, 250, ...
    at amplitudes 4/(pi k)), whose shape visibly rounds off toward a
    sine as harmonics are removed.
    """
    if x is None:
        t = np.arange(int(sample_rate * 0.512)) / sample_rate
        x = np.sign(np.sin(2 * np.pi * 50 * t))
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if cutoffs is None:
        cutoffs = np.linspace(sample_rate / 2, 40, 24)
    X = fft(x)
    f = fftfreq(n, 1 / sample_rate)
    pos = f >= 0
    mag = np.abs(X[pos]) / n * 2
    t = np.arange(n) / sample_rate
    show = t <= show_seconds

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6))
    kept = a1.vlines(f[pos], 0, mag, color=COLORS[0], lw=1.2)
    cut_line = a1.axvline(cutoffs[0], color=COLORS[4], lw=1.2)
    a1.set_xlim(0, sample_rate / 2)
    a1.set_ylim(0, mag.max() * 1.1)
    a1.set_xlabel("Frequency (Hz)")
    a1.set_ylabel("Amplitude")
    a1.spines[["top", "right"]].set_visible(False)
    title = a1.set_title("", loc="left", fontsize=11)

    a2.plot(t[show], x[show], color=COLORS[5], lw=0.9, label="original")
    (filt_line,) = a2.plot(t[show], x[show], color=COLORS[1], lw=1.4, label="filtered")
    a2.set_xlabel("Time (s)")
    a2.legend(frameon=False, loc="upper right")
    a2.spines[["top", "right"]].set_visible(False)

    def draw(i):
        fc = cutoffs[i]
        H = (np.abs(f) <= fc).astype(float)
        y = ifft(X * H).real
        colors = [COLORS[0] if fv <= fc else "#d0d0d0" for fv in f[pos]]
        kept.set_color(colors)
        cut_line.set_xdata([fc, fc])
        filt_line.set_ydata(y[show])
        title.set_text(f"Low-pass cutoff {fc:6.1f} Hz: {int(np.sum(H))} of {n} bins kept")
        return kept, cut_line, filt_line

    anim = animation.FuncAnimation(fig, draw, frames=len(cutoffs), interval=1000 / fps, blit=False)

    def fallback(png):
        idx = np.linspace(0, len(cutoffs) - 1, 4).astype(int)
        fig2, axes = plt.subplots(len(idx), 1, figsize=(8, 7), sharex=True)
        for a, i in zip(axes, idx):
            y = ifft(X * (np.abs(f) <= cutoffs[i])).real
            a.plot(t[show], x[show], color=COLORS[5], lw=0.8)
            a.plot(t[show], y[show], color=COLORS[1], lw=1.2)
            a.set_ylabel(f"{cutoffs[i]:.0f} Hz", fontsize=8)
        fig2.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig2)

    return _save_animation(anim, fig, save_path, fps, fallback)
