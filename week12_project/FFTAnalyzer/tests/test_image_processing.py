"""
test_image_processing.py: 2D filters, the DCT, JPEG-style compression,
the three application classes (image, audio, time series), and the
visualization layer (every plot and animation renders and writes a file).
"""

import numpy as np
import pytest

from conftest import rel_err
from fft_analyzer.applications import audio_processor as AP
from fft_analyzer.applications import data_analyzer as DA
from fft_analyzer.applications import image_processor as IP
from fft_analyzer.image import compression as C
from fft_analyzer.image import filters as IF


# --------------------------------------------------------------------------
# DCT
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 8, 16, 100])
def test_dct_inverse(n, rng):
    x = rng.standard_normal(n)
    assert rel_err(C.idct(C.dct(x)), x) < 1e-13


@pytest.mark.parametrize("n", [8, 13, 64])
def test_dct_matches_definition(n, rng):
    x = rng.standard_normal(n)
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    D = np.cos(np.pi * k * (2 * j + 1) / (2 * n)) @ x
    assert rel_err(C.dct(x, norm=None), D) < 1e-13


def test_dct_matches_scipy(rng):
    sf = pytest.importorskip("scipy.fft")
    x = rng.standard_normal((4, 32))
    assert np.allclose(C.dct(x), sf.dct(x, norm="ortho", axis=-1))
    assert np.allclose(C.dct2(x), sf.dctn(x, norm="ortho"))


def test_dct_matrix_orthonormal_and_consistent(rng):
    D = C.dct_matrix(8)
    assert np.allclose(D @ D.T, np.eye(8), atol=1e-14)
    B = rng.standard_normal((8, 8))
    assert np.allclose(D @ B @ D.T, C.dct2(B), atol=1e-12)
    assert np.allclose(C.idct2(C.dct2(B)), B, atol=1e-12)


def test_dct_energy_preserved(rng):
    B = rng.standard_normal((8, 8))
    assert np.sum(C.dct2(B) ** 2) == pytest.approx(np.sum(B ** 2))


def test_constant_block_has_only_dc():
    Y = C.dct2(np.full((8, 8), 5.0))
    assert Y[0, 0] == pytest.approx(40.0)
    Y[0, 0] = 0
    assert np.abs(Y).max() < 1e-12


def test_dct_compacts_energy_better_than_dft():
    y, x = np.mgrid[0:8, 0:8]
    block = 10 * x + 3 * y  # a smooth ramp, typical image content
    dct_frac, dft_frac = C.energy_compaction(block, keep=3)
    assert dct_frac > 0.97
    assert dct_frac > dft_frac


# --------------------------------------------------------------------------
# JPEG pieces
# --------------------------------------------------------------------------

def test_quality_table_scaling():
    assert np.array_equal(C.quality_scaled_table(C.JPEG_LUMA_Q, 50), C.JPEG_LUMA_Q)
    assert np.all(C.quality_scaled_table(C.JPEG_LUMA_Q, 100) == 1)
    q10 = C.quality_scaled_table(C.JPEG_LUMA_Q, 10)
    q90 = C.quality_scaled_table(C.JPEG_LUMA_Q, 90)
    assert np.all(q10 >= C.JPEG_LUMA_Q) and np.all(q90 <= C.JPEG_LUMA_Q)
    assert q10.max() <= 255


def test_zigzag_order():
    z = C.zigzag_indices(8)
    assert z[:6] == [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2)]
    assert z[-1] == (7, 7)
    assert len(set(z)) == 64


def test_ycbcr_round_trip(rng):
    rgb = rng.uniform(0, 255, (10, 10, 3))
    assert np.allclose(C.ycbcr_to_rgb(C.rgb_to_ycbcr(rgb)), rgb, atol=1e-3)
    gray = np.full((2, 2, 3), 100.0)
    ycc = C.rgb_to_ycbcr(gray)
    assert np.allclose(ycc[..., 0], 100) and np.allclose(ycc[..., 1:], 128)


def test_blockify_round_trip(rng):
    a = rng.standard_normal((21, 30))  # not multiples of 8
    blocks, shape = C.blockify(a)
    assert blocks.shape == (3, 4, 8, 8)
    assert np.array_equal(C.unblockify(blocks, shape), a)


# --------------------------------------------------------------------------
# compression behaviour
# --------------------------------------------------------------------------

def test_compression_error_decreases_with_quality(test_image):
    psnrs, bits = [], []
    for q in (5, 20, 50, 80, 95):
        _, s = C.compress_decompress(test_image, q)
        psnrs.append(s["psnr_db"])
        bits.append(s["estimated_bits"])
    assert psnrs == sorted(psnrs)
    assert bits == sorted(bits)
    assert psnrs[0] > 22 and psnrs[-1] > 40


def test_compression_q100_nearly_lossless(test_image):
    recon, s = C.compress_decompress(test_image, 100)
    orig = np.round(test_image * 255)
    assert np.abs(recon.astype(float) - orig).max() <= 2
    assert s["psnr_db"] > 50


def test_compression_creates_sparsity(test_image):
    _, s = C.compress_decompress(test_image, 30)
    assert s["zero_fraction"] > 0.85
    assert s["estimated_ratio"] > 5


def test_compression_color_and_subsampling(rng):
    img = IP.make_test_image(96, color=True)
    full, s_full = C.compress_decompress(img, 75)
    sub, s_sub = C.compress_decompress(img, 75, subsample=True)
    assert full.shape == sub.shape == (96, 96, 3) and full.dtype == np.uint8
    assert s_full["psnr_db"] > 30
    assert s_sub["estimated_bits"] < s_full["estimated_bits"]
    # 4:2:0 averages chroma over 2x2 pixels; the synthetic card has
    # full-contrast colour edges and 3-pixel stripes, a worst case, so
    # subsampling alone costs ~10 dB here (it's ~invisible on photos)
    assert 18 < s_sub["psnr_db"] < s_full["psnr_db"]


def test_compression_odd_size(rng):
    img = rng.uniform(0, 1, (37, 53))
    recon, _ = C.compress_decompress(img, 90)
    assert recon.shape == (37, 53)


def test_psnr_and_mse():
    a = np.zeros((4, 4))
    assert C.psnr(a, a) == float("inf")
    assert C.mse(a, a + 2) == 4.0
    assert C.psnr(a, a + 1) == pytest.approx(20 * np.log10(255))


# --------------------------------------------------------------------------
# image filters
# --------------------------------------------------------------------------

def test_gaussian_blur_matches_spatial_kernel(test_image):
    a = IF.gaussian_blur(test_image, 2.0)
    b = IF.gaussian_blur_spatial(test_image, 2.0)
    assert np.abs(a - b).max() < 1e-3  # continuous vs sampled Gaussian


def test_gaussian_blur_matches_scipy(test_image):
    nd = pytest.importorskip("scipy.ndimage")
    ref = nd.gaussian_filter(test_image, 2.0, mode="reflect")
    assert np.abs(IF.gaussian_blur(test_image, 2.0) - ref).max() < 1e-3


def test_blur_preserves_mean_and_constants(test_image):
    assert IF.gaussian_blur(test_image, 3).mean() == pytest.approx(test_image.mean(), abs=2e-3)
    flat = np.full((40, 50), 0.7)
    assert np.allclose(IF.gaussian_blur(flat, 3), 0.7, atol=1e-12)


def test_blur_reduces_high_frequency_energy(test_image):
    from fft_analyzer.core.fft import fft2, fftfreq
    def hf(img):
        F = np.abs(fft2(img)) ** 2
        u = fftfreq(img.shape[0])[:, None]
        v = fftfreq(img.shape[1])[None, :]
        return F[np.hypot(u, v) > 0.2].sum()
    # (not ~500x as the Gaussian alone predicts at r=0.2: the unpadded
    # FFT here also sees the image's left/right edge mismatch, which no
    # blur removes)
    assert hf(IF.gaussian_blur(test_image, 2)) < hf(test_image) / 10


def test_blur_color_per_channel():
    img = IP.make_test_image(64, color=True)
    out = IF.gaussian_blur(img, 1.5)
    assert out.shape == img.shape
    assert np.allclose(out[..., 1], IF.gaussian_blur(img[..., 1], 1.5))


def test_sharpen_increases_edge_contrast():
    img = np.zeros((64, 64))
    img[:, 32:] = 0.6
    img += 0.2
    s = IF.sharpen(img, 1.5, 1.0)
    assert s[:, 33].mean() > img[:, 33].mean()   # overshoot on the bright side
    assert s[:, 30].mean() < img[:, 30].mean()   # undershoot on the dark side


def test_sobel_matches_scipy(test_image):
    nd = pytest.importorskip("scipy.ndimage")
    _, gx, gy = IF.sobel(test_image)
    assert np.allclose(gx, nd.sobel(test_image, axis=1, mode="reflect"), atol=1e-12)
    assert np.allclose(gy, nd.sobel(test_image, axis=0, mode="reflect"), atol=1e-12)


def test_sobel_finds_vertical_edge():
    img = np.zeros((32, 32))
    img[:, 16:] = 1.0
    mag, gx, gy = IF.sobel(img)
    assert np.argmax(mag[16]) in (15, 16)
    assert np.all(gx[:, 15:17] > 0) and np.abs(gy[2:-2]).max() < 1e-12


def test_laplacian_kernel_vs_spectral_on_smooth_image():
    y, x = np.mgrid[0:64, 0:64]
    img = np.sin(2 * np.pi * x / 32) * np.cos(2 * np.pi * y / 32)
    # the image is exactly periodic, so wrap padding makes the spectral
    # Laplacian exact: -2 (2 pi / 32)^2 * img
    s = IF.laplacian(img, "spectral", padding="wrap")
    assert np.allclose(s, -2 * (2 * np.pi / 32) ** 2 * img, atol=1e-12)
    # the 5-point stencil approximates it with O(h^2) error ~ (kh)^2 / 12 = 0.3%
    k = IF.laplacian(img, "kernel")
    inner = (slice(4, -4), slice(4, -4))
    assert rel_err(k[inner], s[inner]) < 0.01


def test_laplacian_matches_scipy(test_image):
    nd = pytest.importorskip("scipy.ndimage")
    assert np.allclose(IF.laplacian(test_image), nd.laplace(test_image, mode="reflect"), atol=1e-12)


def test_edge_detect_threshold(test_image):
    edges = IF.edge_detect(test_image, "sobel", threshold=0.3)
    assert edges.dtype == bool and 0.01 < edges.mean() < 0.4


def test_lowpass_denoise_improves_rmse(test_image):
    noisy = IF.add_gaussian_noise(test_image, 0.15, seed=0)
    rmse = lambda a: np.sqrt(np.mean((a - test_image) ** 2))
    for kind in ("ideal", "butterworth", "gaussian"):
        assert rmse(IF.denoise_lowpass(noisy, 0.25, kind)) < rmse(noisy)


def test_notch_removes_periodic_noise(test_image):
    striped = IF.add_periodic_noise(test_image, (0.0, 0.25), 0.2)
    peaks = IF.find_periodic_noise_peaks(striped, 1)
    assert abs(abs(peaks[0][1]) - 0.25) < 0.01 and abs(peaks[0][0]) < 0.01
    cleaned = IF.notch_filter(striped, peaks)
    rmse = lambda a: np.sqrt(np.mean((a - test_image) ** 2))
    assert rmse(cleaned) < rmse(striped) / 3


# --------------------------------------------------------------------------
# applications
# --------------------------------------------------------------------------

def test_image_processor_chain_and_save(tmp_path):
    ip = IP.ImageProcessor.test_card(64)
    ip.blur(1.0).sharpen(amount=0.5)
    assert len(ip.history) == 2 and ip.image.shape == (64, 64)
    recon, s = ip.compress(40, save_path=str(tmp_path / "c.png"),
                           save_coefficients=str(tmp_path / "c.npz"))
    assert (tmp_path / "c.png").exists() and s["npz_bytes"] > 0
    back = IP.load_image(str(tmp_path / "c.png"), grayscale=True)
    assert np.abs(back * 255 - recon).max() <= 1
    ip.reset()
    assert np.array_equal(ip.image, ip.original)


def test_image_processor_quality_sweep():
    rows = IP.ImageProcessor.test_card(64).quality_sweep((10, 50, 90))
    assert [r["quality"] for r in rows] == [10, 50, 90]
    assert rows[0]["psnr_db"] < rows[-1]["psnr_db"]


def test_wav_round_trip(tmp_path, rng):
    x = 0.5 * rng.standard_normal((2, 1000)).clip(-1.9, 1.9) / 2
    for width in (1, 2, 4):
        p = tmp_path / f"t{width}.wav"
        AP.write_wav(p, x, 8000, width)
        y, fs = AP.read_wav(p)
        assert fs == 8000 and y.shape == x.shape
        assert np.abs(y - x).max() < {1: 1e-2, 2: 1e-4, 4: 1e-8}[width]


def test_wav_clips_instead_of_wrapping(tmp_path):
    AP.write_wav(tmp_path / "c.wav", np.array([2.0, -2.0, 0.5]), 8000)
    y, _ = AP.read_wav(tmp_path / "c.wav")
    assert y[0, 0] > 0.99 and y[0, 1] < -0.99


def test_audio_cleanup_improves_snr(tmp_path):
    from fft_analyzer.signal.generators import snr_db
    clean, noisy, fs = AP.make_demo_clip(duration=1.5)
    AP.write_wav(tmp_path / "n.wav", noisy, fs)
    ap = AP.AudioProcessor.load(tmp_path / "n.wav")
    before = snr_db(clean[: ap.samples.shape[1]], ap.mono())
    ap.remove_hum(60, 3).low_pass(2000, 100).spectral_denoise()
    after = snr_db(clean, ap.mono())
    assert after > before + 12
    assert ap.band_energy_change(55, 65) < -25
    ap.save(tmp_path / "out.wav")
    assert (tmp_path / "out.wav").exists()


def test_audio_dominant_frequencies():
    clean, noisy, fs = AP.make_demo_clip()
    found = [f for f, _ in AP.AudioProcessor(noisy, fs).dominant_frequencies(6)]
    for f in (60, 330, 120):
        assert min(abs(np.array(found) - f)) < 0.5


def test_audio_stereo_processing():
    fs = 8000
    t = np.arange(fs) / fs
    st = np.stack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 1000 * t)])
    ap = AP.AudioProcessor(st, fs).low_pass(500)
    assert ap.channels == 2
    assert np.abs(ap.samples[1]).max() < 1e-9 and np.abs(ap.samples[0]).max() > 0.99


def test_detect_periodicities_sales():
    found = DA.detect_periodicities(DA.synthetic_sales(), 3)
    confirmed = [p["period"] for p in found if p["confirmed"]]
    assert 7 in confirmed
    yearly = [p for p in confirmed if p > 300]
    assert yearly and abs(yearly[0] - 365) / 365 < 0.05


def test_detect_periodicities_hidden_cycle_in_returns():
    r = DA.log_returns(DA.synthetic_prices())
    found = DA.detect_periodicities(r, 2)
    assert found[0]["confirmed"] and found[0]["period"] == 21


def test_no_false_periodicity_in_white_noise():
    hits = sum(any(p["confirmed"] for p in DA.detect_periodicities(np.random.default_rng(s).standard_normal(500), 2))
               for s in range(30))
    assert hits <= 2


def test_decompose_reconstructs_and_measures_strength():
    x = DA.synthetic_sales()
    d = DA.decompose(x, 7)
    assert np.allclose(d["trend"] + d["seasonal"] + d["residual"], x)
    assert d["seasonal_strength"] > 0.7
    prof = DA.seasonal_profile(x, 7)
    true = 25 * np.sin(2 * np.pi * np.arange(7) / 7 + 0.6)
    assert np.corrcoef(prof, true)[0, 1] > 0.98
    noise = DA.decompose(np.random.default_rng(0).standard_normal(700), 7)
    assert noise["seasonal_strength"] < 0.1


def test_smoothers_preserve_lines_and_reduce_noise(rng):
    line = np.linspace(0, 10, 300)
    noisy = line + rng.normal(0, 1, 300)
    for f, kw in [(DA.moving_average, {"window": 9}), (DA.gaussian_smooth, {"sigma": 3}),
                  (DA.windowed_smooth, {"width": 15})]:
        assert np.allclose(f(line, **kw)[20:-20], line[20:-20], atol=1e-9)
        assert np.std(f(noisy, **kw) - line) < 0.5
    assert np.allclose(DA.moving_average(np.ones(10), 4), 1.0)
    ex = DA.exponential_smooth(np.ones(50), 0.3)
    assert np.allclose(ex, 1.0)


def test_time_series_analyzer_csv(tmp_path):
    x = DA.synthetic_sales(400)
    p = tmp_path / "s.csv"
    rows = ["date,sales"] + [f"d{i},{v:.3f}" if i != 10 else f"d{i}," for i, v in enumerate(x)]
    p.write_text("\n".join(rows))
    ts = DA.TimeSeriesAnalyzer.from_csv(str(p), "sales", "date")
    assert ts.values.size == 400 and np.isfinite(ts.values).all()
    assert ts.dominant_period() == 7
    assert ts.seasonality()["period"] == 7


# --------------------------------------------------------------------------
# visualization smoke tests
# --------------------------------------------------------------------------

def test_plots_render(tmp_path, test_image):
    import matplotlib.pyplot as plt
    from fft_analyzer.signal.generators import chirp
    from fft_analyzer.visualization import plots as P
    _, x = chirp(10, 200, 1, 1000)
    figs = [
        P.plot_frequency_domain(x, 1000, save_path=str(tmp_path / "a.png")),
        P.plot_spectrogram(x, 1000, 128, save_path=str(tmp_path / "b.png")),
        P.plot_fft2_heatmap(test_image, save_path=str(tmp_path / "c.png")),
        P.plot_windows(["hann", "blackman"], save_path=str(tmp_path / "d.png")),
        P.plot_condition_numbers([2, 4, 6], [[1, 10, 100]], ["x"], save_path=str(tmp_path / "e.png")),
        P.plot_image_grid([test_image, test_image], ["a", "b"], save_path=str(tmp_path / "f.png")),
        P.plot_runtime([8, 16, 32], [[1e-3, 2e-3, 4e-3]], ["fft"], reference=[("n", lambda n: n)],
                       save_path=str(tmp_path / "g.png")),
        P.plot_error_vs_condition([10, 100, 1000], [[1e-15, 1e-14, 1e-13]], ["lu"],
                                  save_path=str(tmp_path / "h.png")),
    ]
    for f in figs:
        plt.close(f)
    assert len(list(tmp_path.glob("*.png"))) == 8


def test_fft_stages_end_at_fft(rng):
    from fft_analyzer.core.fft import fft
    from fft_analyzer.visualization.animations import fft_stages
    x = rng.standard_normal(32)
    st = fft_stages(x)
    assert len(st) == 6
    assert np.allclose(st[-1], fft(x))
    with pytest.raises(ValueError):
        fft_stages(np.ones(12))


def test_animations_write_files(tmp_path):
    from fft_analyzer.visualization import animations as AN
    a = AN.animate_fft_process(save_path=str(tmp_path / "fft.gif"))
    b = AN.animate_filtering(save_path=str(tmp_path / "filt.gif"), cutoffs=[400, 200, 60])
    import os
    assert os.path.getsize(a) > 0 and os.path.getsize(b) > 0
