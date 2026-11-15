"""
filters.py (image)
---------------------
Part 4.1 of the FFTAnalyzer project. Image filtering through the 2D FFT:
Gaussian blur, sharpening (unsharp masking), edge detection (Sobel and
Laplacian), and de-noising by frequency-domain low-pass and notch
filtering.

The 2D DFT works exactly like the 1D one, applied to every row and then
every column (fft2 in core/fft.py). Low spatial frequencies near the
centre of the (shifted) spectrum are the smooth, slowly varying parts of
the image; high frequencies are edges, texture and noise. So:

  * blur        = keep the low frequencies (multiply by a Gaussian)
  * edges       = keep the high frequencies (a derivative multiplies
                  the spectrum by i*2*pi*u, which grows with frequency)
  * de-noise    = blur, but chosen to remove where the noise lives

Boundary handling: the DFT treats the image as periodic, so filtering
straight in the frequency domain makes the right edge bleed into the
left one. Every filter here reflect-pads the image first (mirror the
border, which continues it smoothly), filters, then crops back, unless
the caller asks for padding="wrap" explicitly. Colour images (H x W x C)
are filtered channel by channel.
"""

import numpy as np

from ..core.fft import convolve2d, fft2, fftfreq, ifft2, next_power_of_two


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def to_float(image):
    """uint8 [0, 255] -> float [0, 1]; floats are passed through."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    return image.astype(np.float64)


def to_grayscale(image):
    """ITU-R BT.601 luma weights, the same ones JPEG's Y channel uses."""
    image = to_float(image)
    if image.ndim == 2:
        return image
    rgb = image[..., :3]
    return rgb @ np.array([0.299, 0.587, 0.114])


def _per_channel(func):
    """Decorator: apply a 2D filter to each channel of an H x W x C image."""
    def wrapper(image, *args, **kwargs):
        image = to_float(image)
        if image.ndim == 2:
            return func(image, *args, **kwargs)
        return np.stack([func(image[..., c], *args, **kwargs)
                         for c in range(image.shape[2])], axis=-1)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def _pad(image, pad, mode):
    if pad == 0 or mode == "wrap":
        return image
    np_mode = {"reflect": "symmetric", "constant": "constant", "edge": "edge"}[mode]
    return np.pad(image, pad, mode=np_mode)


def _freq_grid(shape):
    """Spatial frequency (cycles per pixel) at every cell of a 2D FFT."""
    u = fftfreq(shape[0])[:, None]
    v = fftfreq(shape[1])[None, :]
    return u, v


def apply_transfer_function(image, H_func, pad=None, padding="reflect"):
    """
    Generic frequency-domain filter: pad, FFT2, multiply by H(u, v), IFFT2,
    crop. `H_func(u, v)` receives the frequency grid (cycles/pixel) of the
    padded image. The padded size is rounded up to powers of two so the
    radix-2 FFT path is used.
    """
    image = np.asarray(image, dtype=np.float64)
    H0, W0 = image.shape
    if pad is None:
        pad = max(8, min(H0, W0) // 8)
    padded = _pad(image, pad, padding)
    if padding == "wrap":
        pad = 0
    ph, pw = padded.shape
    Ph, Pw = next_power_of_two(ph), next_power_of_two(pw)
    buf = np.zeros((Ph, Pw))
    buf[:ph, :pw] = padded
    if padding != "wrap" and (Ph, Pw) != (ph, pw):
        # fill the power-of-two slack by continuing the reflection, not with zeros
        buf = np.pad(padded, ((0, Ph - ph), (0, Pw - pw)), mode="symmetric")
    u, v = _freq_grid((Ph, Pw))
    out = ifft2(fft2(buf) * H_func(u, v)).real
    return out[pad:pad + H0, pad:pad + W0]


# ---------------------------------------------------------------------------
# smoothing / sharpening
# ---------------------------------------------------------------------------

def gaussian_kernel(sigma, radius=None):
    """Normalized 2D Gaussian kernel, radius defaults to ceil(3 sigma)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(np.ceil(3 * sigma)) if radius is None else int(radius)
    ax = np.arange(-radius, radius + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def gaussian_transfer(sigma):
    """
    The Fourier transform of a Gaussian of std sigma (pixels) is another
    Gaussian, exp(-2 pi^2 sigma^2 (u^2 + v^2)), so a Gaussian blur needs
    no kernel at all in the frequency domain, just this multiplier.
    """
    def H(u, v):
        return np.exp(-2 * np.pi ** 2 * sigma ** 2 * (u * u + v * v))
    return H


@_per_channel
def gaussian_blur(image, sigma=2.0, padding="reflect"):
    """Gaussian blur by multiplying the spectrum by the Gaussian's transform."""
    return apply_transfer_function(image, gaussian_transfer(sigma),
                                   pad=int(np.ceil(4 * sigma)) + 1, padding=padding)


@_per_channel
def gaussian_blur_spatial(image, sigma=2.0):
    """Same blur via an explicit kernel and FFT convolution (convolve2d),
    reflect-padded. Used to cross-check gaussian_blur."""
    k = gaussian_kernel(sigma)
    r = k.shape[0] // 2
    padded = np.pad(image, r, mode="symmetric")
    out = convolve2d(padded, k, mode="same")
    return out[r:-r, r:-r]


@_per_channel
def sharpen(image, sigma=1.5, amount=1.0):
    """
    Unsharp masking: sharpened = image + amount * (image - blur(image)).
    (image - blur) is a high-pass of the image, so this boosts the high
    frequencies, i.e. the transfer function is 1 + amount * (1 - G).
    Output is clipped to [0, 1].
    """
    G = gaussian_transfer(sigma)

    def H(u, v):
        return 1.0 + amount * (1.0 - G(u, v))
    return np.clip(apply_transfer_function(image, H, pad=int(np.ceil(4 * sigma)) + 1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# edge detection
# ---------------------------------------------------------------------------

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T
LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float64)


def _conv_same_reflect(image, kernel):
    r = max(kernel.shape) // 2
    padded = np.pad(image, r, mode="symmetric")
    out = convolve2d(padded, kernel, mode="same")
    return out[r:-r, r:-r] if r else out


def sobel(image):
    """
    Sobel gradients via FFT convolution. Returns (magnitude, gx, gy).
    The Sobel kernel is a central difference in one direction times a
    [1, 2, 1] smoothing in the other, so it's a derivative that's a bit
    less sensitive to noise than a bare difference.

    Note convolution flips the kernel, so with SOBEL_X as written gx is
    positive where intensity *decreases* to the right; the magnitude
    doesn't care, and the sign is flipped back here so that gx > 0 means
    "gets brighter to the right", the usual convention.
    """
    img = to_grayscale(image)
    gx = -_conv_same_reflect(img, SOBEL_X)
    gy = -_conv_same_reflect(img, SOBEL_Y)
    return np.hypot(gx, gy), gx, gy


def laplacian(image, method="kernel", sigma=None, padding="reflect"):
    """
    Second-derivative edge response, which crosses zero at edges.

      method="kernel":   convolve with the 5-point stencil (FFT convolution)
      method="spectral": multiply the spectrum by -4 pi^2 (u^2 + v^2), the
                         exact Laplacian of the band-limited image

    sigma, if given, Gaussian-smooths first (Laplacian of Gaussian), since
    a second derivative amplifies noise even more than a first.
    padding applies to the spectral method: "reflect" (default) for
    ordinary images, "wrap" for images that really are periodic, where
    the spectral result is then exact.
    """
    img = to_grayscale(image)
    if sigma:
        img = gaussian_blur(img, sigma)
    if method == "kernel":
        return _conv_same_reflect(img, LAPLACIAN_KERNEL)
    if method == "spectral":
        return apply_transfer_function(img, lambda u, v: -4 * np.pi ** 2 * (u * u + v * v),
                                       padding=padding)
    raise ValueError("method must be 'kernel' or 'spectral'")


def edge_detect(image, method="sobel", threshold=None, sigma=1.0):
    """
    Edge map in [0, 1]. method='sobel' (gradient magnitude) or 'laplacian'
    (|LoG|). With a threshold, returns a boolean mask instead.
    """
    if method == "sobel":
        mag, _, _ = sobel(gaussian_blur(to_grayscale(image), sigma) if sigma else image)
    elif method == "laplacian":
        mag = np.abs(laplacian(image, sigma=sigma))
    else:
        raise ValueError("method must be 'sobel' or 'laplacian'")
    peak = mag.max()
    mag = mag / peak if peak > 0 else mag
    return mag > threshold if threshold is not None else mag


# ---------------------------------------------------------------------------
# frequency-domain de-noising
# ---------------------------------------------------------------------------

def _radius(u, v):
    return np.sqrt(u * u + v * v)


def ideal_lowpass(cutoff):
    """1 inside radius `cutoff` (cycles/pixel, max 0.5), 0 outside. Rings badly."""
    return lambda u, v: (_radius(u, v) <= cutoff).astype(np.float64)


def butterworth_lowpass(cutoff, order=2):
    """1 / (1 + (r / cutoff)^(2 order)): smooth roll-off, little ringing."""
    return lambda u, v: 1.0 / (1.0 + (_radius(u, v) / cutoff) ** (2 * order))


def gaussian_lowpass(cutoff):
    """exp(-r^2 / (2 cutoff^2)): no ringing at all (a Gaussian stays a Gaussian)."""
    return lambda u, v: np.exp(-(_radius(u, v) ** 2) / (2 * cutoff ** 2))


@_per_channel
def denoise_lowpass(image, cutoff=0.25, kind="butterworth", order=2):
    """Suppress broadband (white) noise by low-passing. cutoff in cycles/pixel."""
    if kind == "ideal":
        H = ideal_lowpass(cutoff)
    elif kind == "butterworth":
        H = butterworth_lowpass(cutoff, order)
    elif kind == "gaussian":
        H = gaussian_lowpass(cutoff)
    else:
        raise ValueError("kind must be ideal, butterworth or gaussian")
    return apply_transfer_function(image, H)


def find_periodic_noise_peaks(image, num_peaks=2, exclude_radius=0.05):
    """
    Locate strong isolated peaks in the magnitude spectrum away from the
    centre: the signature of periodic noise (scan lines, moire, an
    electrical interference pattern). Returns a list of (u, v) in
    cycles/pixel, one per conjugate pair.
    """
    img = to_grayscale(image)
    F = np.abs(fft2(img - img.mean()))
    u, v = _freq_grid(F.shape)
    F[_radius(u, v) < exclude_radius] = 0.0
    found = []
    flat = np.argsort(F.ravel())[::-1]
    for idx in flat:
        i, j = np.unravel_index(idx, F.shape)
        cand = (float(u[i, 0]), float(v[0, j]))
        if any(np.hypot(cand[0] - a, cand[1] - b) < 0.02 or np.hypot(cand[0] + a, cand[1] + b) < 0.02
               for a, b in found):
            continue
        found.append(cand)
        if len(found) >= num_peaks:
            break
    return found


@_per_channel
def notch_filter(image, centres, radius=0.02):
    """
    Remove periodic noise: zero (Gaussian notch) small discs around each
    (u, v) in `centres` and their mirror images (-u, -v). A notch only
    removes those few frequencies, so the rest of the image is untouched,
    unlike a low-pass.
    """
    def H(u, v):
        h = np.ones_like(u * v)
        for cu, cv in centres:
            for su in (1, -1):
                d2 = (u - su * cu) ** 2 + (v - su * cv) ** 2
                h = h * (1.0 - np.exp(-d2 / (2 * radius ** 2)))
        return h
    return apply_transfer_function(image, H, padding="wrap")


def add_gaussian_noise(image, sigma=0.05, seed=None):
    """Additive white Gaussian noise, clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    image = to_float(image)
    return np.clip(image + rng.normal(0, sigma, image.shape), 0.0, 1.0)


def add_periodic_noise(image, freq=(0.0, 0.25), amplitude=0.2):
    """Add a sinusoidal grating amplitude * sin(2 pi (fu y + fv x))."""
    image = to_float(image)
    H, W = image.shape[:2]
    y, x = np.mgrid[0:H, 0:W]
    pattern = amplitude * np.sin(2 * np.pi * (freq[0] * y + freq[1] * x))
    if image.ndim == 3:
        pattern = pattern[..., None]
    return image + pattern


def log_magnitude_spectrum(image):
    """log(1 + |FFT2|), centred, for display."""
    from ..core.fft import fftshift
    img = to_grayscale(image)
    return np.log1p(np.abs(fftshift(fft2(img - img.mean()))))
