"""
signal_demo.py
-----------------
End-to-end signal processing demo. Run from the FFTAnalyzer/ folder:

    python3 examples/signal_demo.py

Writes figures to examples/output/:
  signal_spectrum.png     a three-tone signal + noise and its spectrum, peaks labelled
  signal_filters.png      low / high / band-pass applied to the same signal
  signal_windows.png      Hann / Hamming / Blackman / rectangular shapes and leakage
  signal_fir.png          windowed-sinc FIR responses for three windows
  signal_chirp_spectrogram.png
  signal_psd.png          periodogram vs Welch on white and pink noise
  timeseries_report.png   periodicity / seasonality report on synthetic sales data
  fft_process.gif         the radix-2 FFT, stage by stage
  filtering.gif           a low-pass sweeping down over a square wave
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fft_analyzer.applications.data_analyzer import TimeSeriesAnalyzer, synthetic_sales
from fft_analyzer.signal import analysis, filters, generators
from fft_analyzer.visualization import animations, plots

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = 2000

    # --- spectrum + peak detection ---------------------------------------
    t, x = generators.multi_tone([50, 120, 437.5], 1.0, fs, [1.0, 0.6, 0.3])
    x_noisy = generators.add_noise(x, 10, seed=0)
    f, A = analysis.amplitude_spectrum(x_noisy, fs)
    peaks = analysis.find_peaks(f, A, num_peaks=3, min_distance_hz=10)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    plots.plot_signal(t[:400], x_noisy[:400], "Three tones + noise (10 dB SNR), first 0.2 s", ax=a1)
    plots.plot_spectrum(f, A, "Amplitude spectrum with detected peaks", ax=a2, xlim=(0, 600), peaks=peaks)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "signal_spectrum.png"), dpi=130)
    plt.close(fig)
    print("detected peaks (Hz, amplitude):", [(round(p, 2), round(h, 3)) for p, h in peaks])

    # --- the three filter types -----------------------------------------
    lp = filters.low_pass(x_noisy, fs, 80, transition=10)
    hp = filters.high_pass(x_noisy, fs, 300, transition=10)
    bp = filters.band_pass(x_noisy, fs, 100, 140, transition=10)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7))
    for row, (y, name) in enumerate([(lp, "low-pass 80 Hz"), (hp, "high-pass 300 Hz"),
                                     (bp, "band-pass 100-140 Hz")]):
        plots.plot_signal(t[:400], y[:400], name, ax=axes[row, 0], color=plots.COLORS[row])
        fy, Ay = analysis.amplitude_spectrum(y, fs)
        plots.plot_spectrum(fy, Ay, name + " spectrum", ax=axes[row, 1], xlim=(0, 600),
                            color=plots.COLORS[row])
        print(f"{name:22s} peak -> {analysis.peak_frequency(y, fs):.2f} Hz")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "signal_filters.png"), dpi=130)
    plt.close(fig)

    # --- windows ------------------------------------------------------------
    plt.close(plots.plot_windows(["rectangular", "hann", "hamming", "blackman"],
                                 save_path=os.path.join(OUT, "signal_windows.png")))

    # --- FIR design -----------------------------------------------------------
    # same configuration as the report and tests: 101 taps, 100 Hz cutoff, fs = 1 kHz
    fir_fs = 1000
    resp = []
    for w in ["rectangular", "hamming", "blackman"]:
        taps = filters.design_fir(101, 100, fir_fs, "lowpass", w)
        ff, H = filters.fir_frequency_response(taps, 8192, fir_fs)
        resp.append(H)
        print(f"FIR 101 taps, {w:11s}: worst stop-band gain beyond 160 Hz = "
              f"{20 * np.log10(H[ff > 160].max()):6.1f} dB")
    plt.close(plots.plot_filter_response(ff, resp, ["rectangular", "hamming", "blackman"],
                                         "Windowed-sinc FIR low-pass (101 taps, 100 Hz, fs = 1 kHz)",
                                         save_path=os.path.join(OUT, "signal_fir.png")))

    # --- chirp spectrogram ----------------------------------------------------
    t, c = generators.chirp(20, 900, 3.0, fs)
    plt.close(plots.plot_spectrogram(c, fs, 256, 64, "Linear chirp 20 -> 900 Hz",
                                     save_path=os.path.join(OUT, "signal_chirp_spectrogram.png")))

    # --- PSD: periodogram vs Welch -------------------------------------------
    _, wn = generators.white_noise(8, fs, seed=1)
    _, pn = generators.pink_noise(8, fs, seed=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (sig, name) in zip(axes, [(wn, "white"), (pn, "pink")]):
        fpg, ppg = analysis.periodogram(sig, fs)
        fw, pw = analysis.welch_psd(sig, fs, 1024)
        ax.loglog(fpg[1:], ppg[1:], color=plots.COLORS[5], lw=0.4, alpha=0.6, label="periodogram")
        ax.loglog(fw[1:], pw[1:], color=plots.COLORS[0], lw=1.5, label="Welch (1024, 50%)")
        ax.set_title(f"{name} noise PSD", loc="left")
        ax.set_xlabel("Hz")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        m = (fw > 5) & (fw < 800)
        slope = np.polyfit(np.log(fw[m]), np.log(pw[m]), 1)[0]
        print(f"{name} noise: Welch log-log PSD slope = {slope:+.2f} (expected {0 if name == 'white' else -1})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "signal_psd.png"), dpi=130)
    plt.close(fig)

    # --- time series ------------------------------------------------------------
    ts = TimeSeriesAnalyzer(synthetic_sales(), name="synthetic daily sales (2 years)")
    for p in ts.periodicities(3):
        print(f"period {p['period']!s:>8}  spectral {p['period_spectral']:7.2f}  "
              f"ACF {p['acf']:+.2f}  confirmed={p['confirmed']}")
    plt.close(ts.plot_report(os.path.join(OUT, "timeseries_report.png")))

    # --- animations -------------------------------------------------------------
    print("wrote", animations.animate_fft_process(save_path=os.path.join(OUT, "fft_process.gif")))
    print("wrote", animations.animate_filtering(save_path=os.path.join(OUT, "filtering.gif")))
    print("figures in", OUT)


if __name__ == "__main__":
    main()
