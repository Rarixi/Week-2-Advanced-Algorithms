"""
image_processor.py
---------------------
Part 5.2 of the FFTAnalyzer project. An image workflow on top of
image/filters.py and image/compression.py: load an image, apply FFT
filters (blur, sharpen, edges, de-noise, notch out periodic noise),
JPEG-style DCT compression at any quality, a quality sweep that charts
PSNR against estimated size, frequency-domain visualizations, and
saving both the reconstructed image and the quantized coefficients.

Image I/O goes through Pillow when it's installed (any format) and
falls back to matplotlib's PNG reader/writer otherwise.

    ip = ImageProcessor.load("photo.png")
    ip.denoise(cutoff=0.12).sharpen(amount=0.8)
    recon, stats = ip.compress(quality=40, save_path="photo_q40.png")
"""

import os

import numpy as np

from ..image import compression
from ..image import filters as imf


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_image(path, grayscale=False):
    """Returns a float image in [0, 1], H x W (gray) or H x W x 3 (RGB)."""
    try:
        from PIL import Image
        img = Image.open(path)
        img = img.convert("L" if grayscale else "RGB")
        arr = np.asarray(img, dtype=np.float64) / 255.0
    except ImportError:
        import matplotlib.image as mpimg
        arr = np.asarray(mpimg.imread(path), dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3:
            arr = arr[..., :3]
        if grayscale and arr.ndim == 3:
            arr = imf.to_grayscale(arr)
    return arr


def save_image(path, image):
    """Save a float [0, 1] or uint8 image (format from the extension)."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import matplotlib.image as mpimg
        mpimg.imsave(path, img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
    return path


def make_test_image(size=256, color=False):
    """
    A synthetic test card, so the demos and tests don't depend on any
    image file: a smooth gradient (low frequencies), a disc and a square
    (sharp edges), a fine stripe patch (high frequencies) and a
    checkerboard. Returns float in [0, 1].
    """
    y, x = np.mgrid[0:size, 0:size] / size
    img = 0.25 + 0.35 * x + 0.15 * np.sin(2 * np.pi * 1.5 * y)
    disc = (x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.15 ** 2
    img[disc] = 0.9
    sq = (x > 0.55) & (x < 0.85) & (y > 0.15) & (y < 0.45)
    img[sq] = 0.1
    stripes = (x > 0.1) & (x < 0.45) & (y > 0.6) & (y < 0.9)
    img[stripes] = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * x[stripes] * size / 6))
    checker = (x > 0.6) & (x < 0.9) & (y > 0.6) & (y < 0.9)
    cb = ((np.floor(x * size / 16) + np.floor(y * size / 16)) % 2).astype(float)
    img[checker] = 0.2 + 0.6 * cb[checker]
    img = np.clip(img, 0.0, 1.0)
    if not color:
        return img
    r = img
    g = np.clip(img * 0.8 + 0.2 * y, 0, 1)
    b = np.clip(1.0 - img * 0.7 + 0.1 * np.sin(2 * np.pi * 3 * x), 0, 1)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# processor
# ---------------------------------------------------------------------------

class ImageProcessor:
    """
    Holds one image (float [0, 1]) and applies filters in place; each
    filter returns self so calls chain. `history` records the steps.
    """

    def __init__(self, image):
        self.image = imf.to_float(image)
        self.original = self.image.copy()
        self.history = []

    @classmethod
    def load(cls, path, grayscale=False):
        return cls(load_image(path, grayscale))

    @classmethod
    def test_card(cls, size=256, color=False):
        return cls(make_test_image(size, color))

    def _set(self, img, label):
        self.image = np.clip(img, 0.0, 1.0)
        self.history.append(label)
        return self

    # -- filters --------------------------------------------------------

    def blur(self, sigma=2.0):
        return self._set(imf.gaussian_blur(self.image, sigma), f"gaussian blur s={sigma}")

    def sharpen(self, sigma=1.5, amount=1.0):
        return self._set(imf.sharpen(self.image, sigma, amount), f"sharpen a={amount}")

    def denoise(self, cutoff=0.25, kind="butterworth"):
        return self._set(imf.denoise_lowpass(self.image, cutoff, kind), f"{kind} low-pass {cutoff}")

    def remove_periodic_noise(self, num_peaks=1, radius=0.02):
        """Find the strongest off-centre spectral peaks and notch them out."""
        peaks = imf.find_periodic_noise_peaks(self.image, num_peaks)
        return self._set(imf.notch_filter(self.image, peaks, radius), f"notch {peaks}")

    def edges(self, method="sobel", sigma=1.0):
        """Replaces the image with its (normalized) edge map."""
        return self._set(imf.edge_detect(self.image, method, sigma=sigma), f"{method} edges")

    def reset(self):
        self.image = self.original.copy()
        self.history = []
        return self

    # -- compression ----------------------------------------------------

    def compress(self, quality=50, subsample=False, save_path=None, save_coefficients=None):
        """
        JPEG-style round trip at `quality`. Returns (reconstruction uint8,
        stats). save_path writes the reconstruction; save_coefficients
        writes the quantized coefficients and tables as a compressed
        .npz (the actual "compressed file" of this toolkit).
        """
        data = compression.compress(self.image, quality, subsample)
        recon = compression.decompress(data)
        orig8 = np.clip(np.round(self.image * 255), 0, 255).astype(np.uint8)
        stats = compression.compression_stats(orig8, recon, data)
        if save_path:
            save_image(save_path, recon)
        if save_coefficients:
            arrays = {f"q{i}": c.astype(np.int16) for i, c in enumerate(data["channels"])}
            arrays.update({f"table{i}": t for i, t in enumerate(data["tables"])})
            np.savez_compressed(save_coefficients, **arrays,
                                shapes=np.array(data["shapes"]),
                                image_shape=np.array(data["image_shape"]),
                                quality=data["quality"], subsample=data["subsample"],
                                color=data["color"])
            stats["npz_bytes"] = os.path.getsize(save_coefficients)
        return recon, stats

    def quality_sweep(self, qualities=(5, 10, 20, 30, 50, 70, 85, 95), subsample=False):
        """PSNR / bits per pixel / zero fraction at each quality."""
        rows = []
        for q in qualities:
            _, s = self.compress(q, subsample)
            rows.append(s)
        return rows

    # -- visualization --------------------------------------------------

    def plot_frequency_domain(self, save_path=None):
        """Original and current image above their centred log-magnitude spectra."""
        import matplotlib.pyplot as plt
        from ..visualization.plots import plot_fft2_heatmap
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for col, (img, name) in enumerate([(self.original, "original"), (self.image, "processed")]):
            ax = axes[0, col]
            ax.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=1)
            ax.set_title(name if name == "original" else "processed: " + (", ".join(self.history) or "none"),
                         fontsize=9)
            ax.axis("off")
            plot_fft2_heatmap(img, title=f"spectrum ({name})", ax=axes[1, col])
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        return fig

    def plot_quality_sweep(self, save_path=None, rows=None):
        import matplotlib.pyplot as plt
        from ..visualization.plots import COLORS
        rows = rows or self.quality_sweep()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        q = [r["quality"] for r in rows]
        a1.plot([r["bits_per_pixel"] for r in rows], [r["psnr_db"] for r in rows], "o-", color=COLORS[0])
        for r in rows:
            a1.annotate(f"q{r['quality']}", (r["bits_per_pixel"], r["psnr_db"]), fontsize=7,
                        textcoords="offset points", xytext=(3, -8))
        a1.set_xlabel("estimated bits per pixel")
        a1.set_ylabel("PSNR (dB)")
        a1.set_title("Rate-distortion", loc="left", fontsize=11)
        a2.plot(q, [100 * r["zero_fraction"] for r in rows], "o-", color=COLORS[1])
        a2.set_xlabel("quality")
        a2.set_ylabel("% coefficients quantized to 0")
        a2.set_title("Sparsity after quantization", loc="left", fontsize=11)
        for a in (a1, a2):
            a.spines[["top", "right"]].set_visible(False)
            a.grid(alpha=0.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig

    def save(self, path):
        return save_image(path, self.image)
