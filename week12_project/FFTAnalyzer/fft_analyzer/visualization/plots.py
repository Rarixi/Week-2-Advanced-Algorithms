"""
plots.py
-----------
Part 6.1 of the FFTAnalyzer project. Static plots: signals and their
spectra, spectrograms, 2D FFT heatmaps, filter responses, image
comparison grids, and condition-number / stability plots for the matrix
code.

Every function takes an optional `ax` (draw into an existing axes) and
returns the matplotlib Figure, and every function that writes a file
takes `save_path`. The Agg backend is selected when no display is
available, so everything works headless (the tests and benchmarks run
that way).
"""

import os

import matplotlib

if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..core.fft import fft2, fftshift
from ..signal.analysis import amplitude_spectrum, spectrogram

# one consistent palette for every figure in the project
COLORS = ["#2a6fdb", "#e0703a", "#2f9e6e", "#8e5bd0", "#c23b5a", "#7a7a7a"]


def _fig_ax(ax, figsize=(8, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_signal(t, x, title="Signal", ax=None, save_path=None, max_points=20_000, label=None, color=None):
    """Time-domain plot (decimated for display if very long)."""
    fig, ax = _fig_ax(ax)
    step = max(1, len(t) // max_points)
    ax.plot(np.asarray(t)[::step], np.asarray(x)[::step], lw=0.9, color=color or COLORS[0], label=label)
    _style(ax, title, "Time (s)", "Amplitude")
    if label:
        ax.legend(frameon=False)
    return _save(fig, save_path)


def plot_spectrum(freqs, values, title="Frequency domain", ax=None, save_path=None,
                  db=False, xlim=None, label=None, color=None, peaks=None):
    """One-sided amplitude or power spectrum, optionally in dB, with peak markers."""
    fig, ax = _fig_ax(ax)
    v = np.asarray(values, dtype=np.float64)
    y = 20 * np.log10(np.maximum(v, 1e-12)) if db else v
    ax.plot(freqs, y, lw=1.0, color=color or COLORS[0], label=label)
    if peaks:
        for f, h in peaks:
            hy = 20 * np.log10(max(h, 1e-12)) if db else h
            ax.plot([f], [hy], "o", ms=4, color=COLORS[1])
            ax.annotate(f"{f:.1f} Hz", (f, hy), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if xlim:
        ax.set_xlim(*xlim)
    _style(ax, title, "Frequency (Hz)", "Magnitude (dB)" if db else "Amplitude")
    if label:
        ax.legend(frameon=False)
    return _save(fig, save_path)


def plot_frequency_domain(x, sample_rate, title="Signal and its spectrum", save_path=None, xlim=None, db=False):
    """Two-panel figure: the signal on top, its amplitude spectrum below."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    t = np.arange(len(x)) / sample_rate
    plot_signal(t, x, "Time domain", ax=a1)
    f, A = amplitude_spectrum(x, sample_rate)
    plot_spectrum(f, A, "Amplitude spectrum (Hann window)", ax=a2, db=db, xlim=xlim)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_spectrogram(x, sample_rate, window_length=512, hop=None, title="Spectrogram",
                     ax=None, save_path=None, fmax=None, vmin_db=-100):
    """Spectrogram (dB) as an image, time on x, frequency on y."""
    fig, ax = _fig_ax(ax, figsize=(9, 4))
    f, t, S = spectrogram(x, sample_rate, window_length, hop)
    vmax = S.max()
    im = ax.pcolormesh(t, f, S, shading="auto", cmap="magma",
                       vmin=max(vmin_db, vmax - 90), vmax=vmax)
    if fmax:
        ax.set_ylim(0, fmax)
    _style(ax, title, "Time (s)", "Frequency (Hz)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return _save(fig, save_path)


def plot_fft2_heatmap(image, title="2D FFT (log magnitude, centred)", ax=None, save_path=None):
    """Heatmap of log(1 + |FFT2|) with zero frequency at the centre."""
    fig, ax = _fig_ax(ax, figsize=(5, 5))
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    mag = np.log1p(np.abs(fftshift(fft2(img - img.mean()))))
    H, W = mag.shape
    im = ax.imshow(mag, cmap="viridis", extent=[-0.5, 0.5, 0.5, -0.5])
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("v (cycles/pixel)")
    ax.set_ylabel("u (cycles/pixel)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, save_path)


def plot_image_grid(images, titles, save_path=None, cmap="gray", ncols=None, suptitle=None):
    """Side-by-side image comparison."""
    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, title in zip(axes.ravel(), images, titles):
        img = np.asarray(img)
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0 if img.dtype == np.uint8 or img.min() >= 0 else None,
                      vmax=255 if img.dtype == np.uint8 else None)
        else:
            ax.imshow(np.clip(img, 0, 1) if img.dtype != np.uint8 else img)
        ax.set_title(title, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_filter_response(freqs, responses, labels, title="Filter frequency response",
                         save_path=None, db=True, ax=None):
    """Overlay several |H(f)| curves."""
    fig, ax = _fig_ax(ax)
    for i, (H, lab) in enumerate(zip(responses, labels)):
        H = np.abs(np.asarray(H))
        y = 20 * np.log10(np.maximum(H, 1e-10)) if db else H
        ax.plot(freqs, y, lw=1.2, color=COLORS[i % len(COLORS)], label=lab)
    if db:
        ax.set_ylim(-120, 5)
    _style(ax, title, "Frequency (Hz)", "Gain (dB)" if db else "Gain")
    ax.legend(frameon=False)
    return _save(fig, save_path)


def plot_windows(windows, n=64, save_path=None):
    """Window shapes (left) and their spectra / leakage (right)."""
    from ..core.fft import fft
    from ..signal.filters import get_window
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    m = 4096
    for i, name in enumerate(windows):
        w = get_window(name, n)
        a1.plot(w, lw=1.3, color=COLORS[i % len(COLORS)], label=name)
        W = np.abs(fft(np.concatenate([w, np.zeros(m - n)])))
        W = fftshift(W) / W.max()
        a2.plot(np.linspace(-n / 2, n / 2, m, endpoint=False), 20 * np.log10(np.maximum(W, 1e-12)),
                lw=1.0, color=COLORS[i % len(COLORS)], label=name)
    _style(a1, "Window functions", "Sample", "w[n]")
    _style(a2, "Spectral leakage", "Frequency (bins)", "dB")
    a2.set_xlim(-12, 12)
    a2.set_ylim(-120, 5)
    a1.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_condition_numbers(sizes, conds, labels, title="Condition number vs matrix size",
                           save_path=None, ax=None):
    """Log-scale condition numbers for several matrix families, with the
    1/sqrt(eps) and 1/eps reference lines."""
    fig, ax = _fig_ax(ax)
    for i, (c, lab) in enumerate(zip(conds, labels)):
        ax.semilogy(sizes, c, "o-", ms=3.5, lw=1.3, color=COLORS[i % len(COLORS)], label=lab)
    eps = np.finfo(np.float64).eps
    ax.axhline(1 / np.sqrt(eps), color=COLORS[5], ls="--", lw=0.9)
    ax.axhline(1 / eps, color=COLORS[4], ls="--", lw=0.9)
    ax.text(sizes[0], 1 / np.sqrt(eps) * 2, "1/sqrt(eps): half the digits gone", fontsize=8, color=COLORS[5])
    ax.text(sizes[0], 1 / eps * 2, "1/eps: numerically singular", fontsize=8, color=COLORS[4])
    _style(ax, title, "n", "cond_2(A)")
    ax.legend(frameon=False)
    return _save(fig, save_path)


def plot_error_vs_condition(conds, errors, labels, title="Error vs condition number",
                            save_path=None, ax=None):
    """Log-log scatter/line of measured error against cond(A), with the
    eps * cond(A) reference line the rule of thumb predicts."""
    fig, ax = _fig_ax(ax)
    for i, (e, lab) in enumerate(zip(errors, labels)):
        ax.loglog(conds, np.maximum(e, 1e-18), "o-", ms=3.5, lw=1.2, color=COLORS[i % len(COLORS)], label=lab)
    c = np.asarray(conds)
    eps = np.finfo(np.float64).eps
    ax.loglog(c, eps * c, "--", color=COLORS[5], lw=0.9, label="eps * cond")
    ax.loglog(c, eps * c ** 2, ":", color=COLORS[5], lw=0.9, label="eps * cond^2")
    _style(ax, title, "cond_2(A)", "error")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, save_path)


def plot_runtime(sizes, times, labels, title="Runtime", save_path=None, ax=None,
                 reference=None, xlabel="n"):
    """Log-log runtime curves. `reference` = list of (label, func(n)) guide lines,
    scaled to meet the first curve at its last point."""
    fig, ax = _fig_ax(ax)
    for i, (t, lab) in enumerate(zip(times, labels)):
        s = np.asarray(sizes[: len(t)])
        ax.loglog(s, t, "o-", ms=3.5, lw=1.3, color=COLORS[i % len(COLORS)], label=lab)
    if reference:
        s = np.asarray(sizes, dtype=float)
        anchor_n, anchor_t = s[len(times[0]) - 1], times[0][-1]
        for j, (lab, f) in enumerate(reference):
            ref = f(s) / f(anchor_n) * anchor_t
            ax.loglog(s, ref, "--", lw=0.8, color=COLORS[5], alpha=0.8)
            ax.text(s[-1], ref[-1], " " + lab, fontsize=8, color=COLORS[5], va="center")
    _style(ax, title, xlabel, "seconds")
    ax.legend(frameon=False)
    return _save(fig, save_path)
