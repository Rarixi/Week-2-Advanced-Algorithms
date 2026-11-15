"""
compression.py
-----------------
Part 4.2 of the FFTAnalyzer project. JPEG-style lossy compression: the 2D
Discrete Cosine Transform, quantization with the standard JPEG tables
scaled by a quality setting, and reconstruction, plus the metrics that
say how well it worked (MSE, PSNR, compression ratio, an entropy-based
estimate of the coded size).

Why the DCT and not the FFT? The DFT of a block implicitly repeats the
block periodically, so unless the left and right edges happen to match
there's a jump at the seam, and a jump has energy at every frequency.
The DCT-II implicitly *mirrors* the block instead (even extension), so
there's no seam, and the energy of a smooth block piles up in a few
low-frequency coefficients. That energy compaction is the whole basis
of JPEG: after quantization, most high-frequency coefficients become
exactly zero and cost almost nothing to store.

The DCT here is implemented two ways:
  * dct / idct (1D, any length) via the project's own FFT, using
    Makhoul's reordering trick so a length-n DCT is a length-n FFT plus
    O(n) twiddling, O(n log n);
  * dct_matrix(8), the orthonormal 8 x 8 DCT matrix C, so a block's 2D
    DCT is C B C^T. For JPEG's fixed 8 x 8 blocks the matrix form,
    applied to every block of the image in one batched product, is the
    faster of the two; the tests check both agree.

The pipeline (per 8 x 8 block of each channel):

    level shift (-128) -> 2D DCT -> divide by Q, round -> [stored]
    -> multiply by Q -> inverse DCT -> +128, clip to [0, 255]

Colour images go through RGB -> YCbCr first, with the (coarser) chroma
table on Cb and Cr, and optional 4:2:0 chroma subsampling, the same
choices a real JPEG encoder makes, since the eye is far less sensitive to
colour detail than to brightness detail. Entropy coding (zig-zag + run
length + Huffman) is not implemented; instead the coded size is
*estimated* from the Shannon entropy of the quantized coefficients,
which is a lower bound on what any symbol-by-symbol coder could achieve.
"""

import numpy as np

from ..core.fft import fft, ifft

# ---------------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------------


def dct(x, norm="ortho"):
    """
    DCT-II along the last axis:  X[k] = sum_j x[j] cos(pi k (2j + 1) / (2n)),
    scaled to be orthonormal when norm='ortho' (so idct is its transpose).

    Makhoul (1980): reorder x into v = [x0, x2, x4, ..., x5, x3, x1] (evens
    ascending, then odds descending). Then
    X[k] = Re( exp(-i pi k / (2n)) * FFT(v)[k] ), one complex FFT.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    v = np.concatenate([x[..., ::2], x[..., 1::2][..., ::-1]], axis=-1)
    V = fft(v)
    k = np.arange(n)
    X = (np.exp(-1j * np.pi * k / (2 * n)) * V).real
    if norm == "ortho":
        X = X * np.sqrt(2.0 / n)
        X[..., 0] /= np.sqrt(2.0)
    elif norm is not None:
        raise ValueError("norm must be 'ortho' or None")
    return X


def idct(X, norm="ortho"):
    """
    Inverse of dct (a scaled DCT-III), by running Makhoul's steps
    backwards: undo the scaling, rebuild the complex spectrum V from X
    using V[k] = exp(i pi k / 2n) (X[k] - i X[n-k]), inverse FFT, then
    un-interleave evens and odds.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    n = X.shape[-1]
    if norm == "ortho":
        X[..., 0] *= np.sqrt(2.0)
        X = X / np.sqrt(2.0 / n)
    elif norm is not None:
        raise ValueError("norm must be 'ortho' or None")
    k = np.arange(n)
    X_rev = np.zeros_like(X)
    X_rev[..., 1:] = X[..., :0:-1]  # X[n - k], with X[n] = 0
    V = np.exp(1j * np.pi * k / (2 * n)) * (X - 1j * X_rev)
    v = ifft(V).real
    x = np.empty_like(v)
    half = (n + 1) // 2
    x[..., ::2] = v[..., :half]
    x[..., 1::2] = v[..., half:][..., ::-1]
    return x


def dct2(block):
    """2D orthonormal DCT-II: rows then columns."""
    return dct(dct(block).swapaxes(-1, -2)).swapaxes(-1, -2)


def idct2(coeffs):
    """Inverse 2D DCT."""
    return idct(idct(coeffs).swapaxes(-1, -2)).swapaxes(-1, -2)


def dct_matrix(n=8):
    """
    Orthonormal DCT-II matrix: C[k, j] = a_k cos(pi k (2j+1) / (2n)),
    a_0 = sqrt(1/n), a_k = sqrt(2/n). C C^T = I, so the 2D DCT of a block
    B is C B C^T and its inverse is C^T Y C.
    """
    j = np.arange(n)
    k = j[:, None]
    C = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2 * j + 1) / (2 * n))
    C[0, :] = np.sqrt(1.0 / n)
    return C


# ---------------------------------------------------------------------------
# JPEG tables
# ---------------------------------------------------------------------------

# ITU-T T.81 Annex K, tables K.1 and K.2
JPEG_LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

JPEG_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def quality_scaled_table(base, quality):
    """
    The IJG (libjpeg) quality scaling: quality 50 uses the table as is,
    higher quality shrinks every step, lower quality grows it.
        scale = 5000 / q        for q < 50
        scale = 200 - 2q        for q >= 50
        Q' = clip(floor((Q * scale + 50) / 100), 1, 255)
    Quality 100 gives a table of all 1s (still lossy: coefficients are
    rounded to integers).
    """
    q = float(np.clip(quality, 1, 100))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    table = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(table, 1, 255)


def zigzag_indices(n=8):
    """
    JPEG zig-zag scan order for an n x n block: walk the anti-diagonals,
    alternating direction, so coefficients come out roughly from lowest
    to highest frequency and the trailing zeros bunch together.
    Returns a list of (row, col).
    """
    order = []
    for s in range(2 * n - 1):
        diag = [(i, s - i) for i in range(n) if 0 <= s - i < n]
        order.extend(diag if s % 2 else diag[::-1])
    return order


# ---------------------------------------------------------------------------
# colour conversion
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(rgb):
    """JFIF RGB -> YCbCr, inputs and outputs in [0, 255]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
    ycc = rgb @ M.T
    ycc[..., 1:] += 128.0
    return ycc


def ycbcr_to_rgb(ycc):
    """JFIF YCbCr -> RGB."""
    ycc = np.asarray(ycc, dtype=np.float64).copy()
    ycc[..., 1:] -= 128.0
    M = np.array([[1.0, 0.0, 1.402],
                  [1.0, -0.344136, -0.714136],
                  [1.0, 1.772, 0.0]])
    return ycc @ M.T


# ---------------------------------------------------------------------------
# block processing
# ---------------------------------------------------------------------------

def _to_uint8_range(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float64)
    image = image.astype(np.float64)
    return image * 255.0 if image.max() <= 1.0 else image


def blockify(channel, block=8):
    """
    Split an H x W channel into (H/b) x (W/b) blocks of b x b, padding the
    bottom/right by edge replication if needed. Returns
    (blocks of shape (nh, nw, b, b), original shape).
    """
    H, W = channel.shape
    ph = (-H) % block
    pw = (-W) % block
    padded = np.pad(channel, ((0, ph), (0, pw)), mode="edge")
    nh, nw = padded.shape[0] // block, padded.shape[1] // block
    blocks = padded.reshape(nh, block, nw, block).swapaxes(1, 2)
    return blocks, (H, W)


def unblockify(blocks, shape):
    """Inverse of blockify (crops the padding)."""
    nh, nw, b, _ = blocks.shape
    full = blocks.swapaxes(1, 2).reshape(nh * b, nw * b)
    return full[: shape[0], : shape[1]]


def encode_channel(channel, table, block=8):
    """
    Level shift, 8 x 8 DCT of every block at once (C B C^T broadcast over
    the block grid), quantize. Returns integer coefficients
    (nh, nw, 8, 8) and the original shape.
    """
    blocks, shape = blockify(channel - 128.0, block)
    C = dct_matrix(block)
    coeffs = C @ blocks @ C.T
    return np.round(coeffs / table).astype(np.int32), shape


def decode_channel(q, table, shape, block=8):
    """Dequantize, inverse DCT (C^T Y C), unshift."""
    C = dct_matrix(block)
    blocks = C.T @ (q.astype(np.float64) * table) @ C
    return unblockify(blocks, shape) + 128.0


def _subsample(ch):
    H, W = ch.shape
    padded = np.pad(ch, ((0, H % 2), (0, W % 2)), mode="edge")
    return padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).mean(axis=(1, 3))


def _upsample(ch, shape):
    return np.repeat(np.repeat(ch, 2, axis=0), 2, axis=1)[: shape[0], : shape[1]]


def compress(image, quality=50, subsample=False):
    """
    JPEG-style compression of a grayscale (H x W) or RGB (H x W x 3) image
    (uint8, or float in [0, 1] / [0, 255]).

    Returns a dict:
      'channels'   list of quantized coefficient arrays (nh, nw, 8, 8)
      'tables'     the quantization table used for each channel
      'shapes'     each channel's pre-blocking shape
      'image_shape', 'quality', 'subsample', 'color'
    """
    img = _to_uint8_range(image)
    if img.ndim == 2:
        table = quality_scaled_table(JPEG_LUMA_Q, quality)
        q, shape = encode_channel(img, table)
        return {"channels": [q], "tables": [table], "shapes": [shape],
                "image_shape": img.shape, "quality": quality,
                "subsample": False, "color": False}
    ycc = rgb_to_ycbcr(img[..., :3])
    lt = quality_scaled_table(JPEG_LUMA_Q, quality)
    ct = quality_scaled_table(JPEG_CHROMA_Q, quality)
    chans, tables, shapes = [], [], []
    for c in range(3):
        ch = ycc[..., c]
        if c > 0 and subsample:
            ch = _subsample(ch)
        q, shape = encode_channel(ch, lt if c == 0 else ct)
        chans.append(q)
        tables.append(lt if c == 0 else ct)
        shapes.append(shape)
    return {"channels": chans, "tables": tables, "shapes": shapes,
            "image_shape": img.shape, "quality": quality,
            "subsample": subsample, "color": True}


def decompress(data):
    """Reconstruct a uint8 image from the dict returned by compress."""
    chans = [decode_channel(q, t, s) for q, t, s in
             zip(data["channels"], data["tables"], data["shapes"])]
    if not data["color"]:
        return np.clip(np.round(chans[0]), 0, 255).astype(np.uint8)
    H, W = data["image_shape"][:2]
    if data["subsample"]:
        chans = [chans[0]] + [_upsample(c, (H, W)) for c in chans[1:]]
    rgb = ycbcr_to_rgb(np.stack(chans, axis=-1))
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def compress_decompress(image, quality=50, subsample=False):
    """Round trip, returning (reconstructed uint8 image, stats dict)."""
    data = compress(image, quality, subsample)
    recon = decompress(data)
    original = np.clip(np.round(_to_uint8_range(image)), 0, 255).astype(np.uint8)
    if original.ndim == 3:
        original = original[..., :3]
    stats = compression_stats(original, recon, data)
    return recon, stats


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b, peak=255.0):
    """Peak signal-to-noise ratio in dB, 10 log10(peak^2 / MSE). ~30-40 dB is "good" for 8-bit images."""
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(peak * peak / m))


def coefficient_entropy_bits(channels):
    """
    Shannon entropy (bits per coefficient) of all quantized coefficients,
    times the coefficient count: an estimate of the entropy-coded size.
    """
    allq = np.concatenate([c.ravel() for c in channels])
    _, counts = np.unique(allq, return_counts=True)
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log2(p)))
    return h * allq.size, h


def compression_stats(original, recon, data):
    """
    MSE, PSNR, fraction of zero coefficients, and an estimated compression
    ratio: (8 bits x original samples) / (entropy estimate of the
    quantized coefficients).
    """
    total_bits, bits_per_coeff = coefficient_entropy_bits(data["channels"])
    coeff_count = sum(c.size for c in data["channels"])
    zeros = sum(int(np.sum(c == 0)) for c in data["channels"])
    original_bits = 8 * np.asarray(original).size
    return {
        "quality": data["quality"],
        "mse": mse(original, recon),
        "psnr_db": psnr(original, recon),
        "zero_fraction": zeros / coeff_count,
        "bits_per_coefficient": bits_per_coeff,
        "estimated_bits": total_bits,
        "estimated_ratio": original_bits / max(total_bits, 1.0),
        "bits_per_pixel": total_bits / (original.shape[0] * original.shape[1]),
    }


def energy_compaction(block, keep):
    """
    Fraction of a block's energy captured by its `keep` largest DCT
    coefficients vs its `keep` largest DFT coefficients, the experiment
    that justifies using the DCT. Returns (dct_fraction, dft_fraction).
    """
    from ..core.fft import fft2
    block = np.asarray(block, dtype=np.float64)
    block = block - block.mean()
    total = np.sum(block ** 2)
    if total == 0:
        return 1.0, 1.0
    d = np.sort((dct2(block) ** 2).ravel())[::-1]
    f = np.sort((np.abs(fft2(block)) ** 2).ravel())[::-1] / block.size  # Parseval scaling
    return float(d[:keep].sum() / total), float(f[:keep].sum() / total)
