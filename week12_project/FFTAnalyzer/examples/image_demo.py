"""
image_demo.py
----------------
Image processing demo. Run from the FFTAnalyzer/ folder:

    python3 examples/image_demo.py                 # synthetic test card
    python3 examples/image_demo.py photo.png       # or your own image

Writes to examples/output/:
  image_filters.png        blur / sharpen / Sobel / Laplacian of Gaussian
  image_denoise.png        white noise removal (ideal vs Butterworth vs Gaussian low-pass)
  image_notch.png          periodic stripe noise removed with a notch filter
  image_frequency.png      image and spectrum before/after blurring
  image_compression.png    JPEG-style reconstructions at several qualities
  image_rate_distortion.png
  image_q30.png / image_q30.npz   a saved compressed image and its coefficients
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fft_analyzer.applications.image_processor import ImageProcessor
from fft_analyzer.image import compression
from fft_analyzer.image import filters as imf
from fft_analyzer.visualization import plots

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def main(path=None):
    os.makedirs(OUT, exist_ok=True)
    ip = ImageProcessor.load(path, grayscale=True) if path else ImageProcessor.test_card(256)
    img = ip.image

    # --- filters --------------------------------------------------------------
    plt.close(plots.plot_image_grid(
        [img, imf.gaussian_blur(img, 3), imf.sharpen(img, 1.5, 1.5),
         imf.edge_detect(img, "sobel"), imf.edge_detect(img, "laplacian", sigma=1.5)],
        ["original", "Gaussian blur (s=3)", "unsharp mask", "Sobel magnitude", "|LoG| (s=1.5)"],
        save_path=os.path.join(OUT, "image_filters.png")))

    # --- white noise ------------------------------------------------------------
    # the cutoff is a trade-off: too low blurs real edges and fine texture
    # (the stripe patch has a 6-pixel period, 0.17 cycles/pixel) more than
    # it removes noise, so sweep it before picking one
    noisy = imf.add_gaussian_noise(img, 0.15, seed=0)
    print(f"white noise (s=0.15): RMSE noisy {rmse(noisy, img):.4f}")
    for k in ("ideal", "butterworth", "gaussian"):
        sweep = [(c, rmse(imf.denoise_lowpass(noisy, c, k), img)) for c in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35)]
        print(f"  {k:11s} " + "  ".join(f"{c:.2f}:{e:.4f}" for c, e in sweep))
    results = {k: imf.denoise_lowpass(noisy, 0.25, k) for k in ("ideal", "butterworth", "gaussian")}
    plt.close(plots.plot_image_grid(
        [noisy] + list(results.values()),
        ["noisy (s=0.15)"] + [f"{k} low-pass 0.25" for k in results],
        save_path=os.path.join(OUT, "image_denoise.png")))

    # --- periodic noise -----------------------------------------------------------
    striped = imf.add_periodic_noise(img, (0.0, 0.2), 0.25)
    peaks = imf.find_periodic_noise_peaks(striped, 1)
    fixed = imf.notch_filter(striped, peaks)
    print(f"periodic noise at {peaks}: RMSE {rmse(striped, img):.4f} -> {rmse(fixed, img):.4f}")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(np.clip(striped, 0, 1), cmap="gray")
    axes[0].set_title("stripe noise")
    plots.plot_fft2_heatmap(striped, "its spectrum (note the two dots)", ax=axes[1])
    axes[2].imshow(np.clip(fixed, 0, 1), cmap="gray")
    axes[2].set_title("after notch filter")
    plots.plot_fft2_heatmap(fixed, "spectrum after notch", ax=axes[3])
    for a in (axes[0], axes[2]):
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "image_notch.png"), dpi=120)
    plt.close(fig)

    # --- frequency-domain view ------------------------------------------------------
    plt.close(ImageProcessor(img).blur(3).plot_frequency_domain(os.path.join(OUT, "image_frequency.png")))

    # --- compression ------------------------------------------------------------------
    qualities = [5, 15, 30, 60, 90]
    recons, titles = [], []
    print(f"{'quality':>7} {'PSNR dB':>8} {'bits/px':>8} {'ratio':>7} {'zeros':>7}")
    for q in qualities:
        recon, s = ip.compress(q)
        recons.append(recon)
        titles.append(f"q={q}: {s['psnr_db']:.1f} dB, {s['bits_per_pixel']:.2f} bpp")
        print(f"{q:7d} {s['psnr_db']:8.2f} {s['bits_per_pixel']:8.3f} {s['estimated_ratio']:7.1f} "
              f"{100 * s['zero_fraction']:6.1f}%")
    plt.close(plots.plot_image_grid([img] + recons, ["original"] + titles, ncols=3,
                                    save_path=os.path.join(OUT, "image_compression.png")))
    plt.close(ip.plot_quality_sweep(os.path.join(OUT, "image_rate_distortion.png")))
    _, s = ip.compress(30, save_path=os.path.join(OUT, "image_q30.png"),
                       save_coefficients=os.path.join(OUT, "image_q30.npz"))
    raw = img.size
    print(f"saved q30: coefficient file {s['npz_bytes']} bytes vs {raw} bytes raw 8-bit "
          f"({raw / s['npz_bytes']:.1f}x, zlib on int16 coefficients, no Huffman stage)")

    dct_frac, dft_frac = compression.energy_compaction(
        np.round(img[96:104, 40:48] * 255), keep=6)
    print(f"energy in top 6 of 64 coefficients for an 8x8 block: DCT {dct_frac:.3f}, DFT {dft_frac:.3f}")
    print("figures in", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
