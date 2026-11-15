"""
test_filters.py: signal generators, FFT filters, windows, FIR design and
spectral analysis. Filters are tested by what they do to known tones
(passband kept, stop band removed), windows against their closed forms
(and scipy, if installed), PSD estimates against Parseval's theorem.
"""

import numpy as np
import pytest

from conftest import rel_err
from fft_analyzer.signal import analysis as A
from fft_analyzer.signal import filters as F
from fft_analyzer.signal import generators as G

FS = 1000


def tone_amplitude(x, f, fs=FS):
    """Amplitude of the component at f Hz (bin-centred tones only)."""
    freqs, amp = A.amplitude_spectrum(x, fs, window="rectangular")
    return amp[np.argmin(np.abs(freqs - f))]


@pytest.fixture
def three_tones():
    return G.multi_tone([50, 150, 300], 1.0, FS, [1.0, 1.0, 1.0])[1]


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------

def test_sine_wave_values():
    t, x = G.sine_wave(5, 1.0, 100, amplitude=2.0)
    assert t.size == 100 and np.isclose(x.max(), 2.0, atol=1e-2)
    assert np.allclose(x, 2 * np.sin(2 * np.pi * 5 * t))


def test_multi_tone_amplitudes(three_tones):
    for f in (50, 150, 300):
        assert tone_amplitude(three_tones, f) == pytest.approx(1.0, abs=1e-9)


def test_multi_tone_length_mismatch():
    with pytest.raises(ValueError):
        G.multi_tone([1, 2], 1, 100, amplitudes=[1])


@pytest.mark.parametrize("method", ["linear", "exponential"])
def test_chirp_instantaneous_frequency(method):
    # numerical derivative of the phase must equal the intended f(t);
    # catches the classic sin(2 pi f(t) t) mistake (which doubles the sweep rate)
    fs, T, f0, f1 = 20_000, 2.0, 50, 2000
    t, x = G.chirp(f0, f1, T, fs, method=method)
    # estimate via STFT peak at a few times
    f, times, S = A.spectrogram(x, fs, 1024, 256, db=False)
    for i in (5, len(times) // 2, len(times) - 5):
        est = f[np.argmax(S[:, i])]
        expected = G.chirp_instantaneous_frequency(f0, f1, times[i], T, method)
        assert abs(est - expected) < 2.5 * fs / 1024 + 0.02 * expected


def test_white_noise_reproducible_and_flat():
    _, a = G.white_noise(4, FS, seed=3)
    _, b = G.white_noise(4, FS, seed=3)
    assert np.array_equal(a, b)
    f, p = A.welch_psd(a, FS, 256)
    slope = np.polyfit(np.log(f[2:]), np.log(p[2:]), 1)[0]
    assert abs(slope) < 0.15


def test_pink_noise_has_one_over_f_spectrum():
    _, x = G.pink_noise(20, FS, std=1.0, seed=7)
    assert np.isclose(x.std(), 1.0) and abs(x.mean()) < 1e-12
    f, p = A.welch_psd(x, FS, 2048)
    m = (f > 2) & (f < 400)
    slope = np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0]
    assert -1.15 < slope < -0.85


def test_add_noise_hits_requested_snr():
    _, x = G.sine_wave(10, 10, FS)
    y = G.add_noise(x, 20, seed=1)
    assert G.snr_db(x, y) == pytest.approx(20, abs=0.3)


def test_square_wave_odd_harmonics():
    _, x = G.square_wave(10, 1.0, FS)
    for k in (1, 3, 5):
        assert tone_amplitude(x, 10 * k) == pytest.approx(4 / (np.pi * k), rel=0.02)
    assert tone_amplitude(x, 20) < 1e-2  # sign(sin) is 0 at the zero crossings, tiny asymmetry


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hann", "hamming", "blackman"])
def test_window_closed_form_values(name):
    w = F.get_window(name, 9)
    assert np.allclose(w, w[::-1]), "symmetric window should be symmetric"
    assert w[4] == pytest.approx(1.0)  # peak at centre
    ends = {"hann": 0.0, "hamming": 0.08, "blackman": 0.0}[name]
    assert w[0] == pytest.approx(ends, abs=1e-12)


@pytest.mark.parametrize("name,scipy_name", [("hann", "hann"), ("hamming", "hamming"), ("blackman", "blackman")])
@pytest.mark.parametrize("sym", [True, False])
def test_windows_match_scipy(name, scipy_name, sym):
    sw = pytest.importorskip("scipy.signal.windows")
    for n in (8, 31, 64):
        assert np.allclose(F.get_window(name, n, symmetric=sym), getattr(sw, scipy_name)(n, sym=sym), atol=1e-14)


def test_enbw_known_values():
    assert F.equivalent_noise_bandwidth(F.rectangular(1024)) == pytest.approx(1.0)
    assert F.equivalent_noise_bandwidth(F.hanning(1024, False)) == pytest.approx(1.5, rel=1e-3)
    assert F.equivalent_noise_bandwidth(F.blackman(1024, False)) == pytest.approx(1.727, rel=2e-3)


def test_window_reduces_leakage():
    # an off-bin tone leaks far less with a Hann window
    fs, n = 1000, 1000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 100.5 * t)
    _, rect = A.amplitude_spectrum(x, fs, "rectangular")
    _, hann = A.amplitude_spectrum(x, fs, "hann")
    far = slice(200, 400)  # 100+ Hz away from the tone
    assert hann[far].max() < rect[far].max() / 100


def test_unknown_window():
    with pytest.raises(ValueError):
        F.get_window("kaiser", 8)


# --------------------------------------------------------------------------
# FFT filters
# --------------------------------------------------------------------------

def test_low_pass(three_tones):
    y = F.low_pass(three_tones, FS, 100)
    assert tone_amplitude(y, 50) == pytest.approx(1.0, abs=1e-9)
    assert tone_amplitude(y, 150) < 1e-9 and tone_amplitude(y, 300) < 1e-9


def test_high_pass(three_tones):
    y = F.high_pass(three_tones, FS, 200)
    assert tone_amplitude(y, 300) == pytest.approx(1.0, abs=1e-9)
    assert tone_amplitude(y, 50) < 1e-9 and tone_amplitude(y, 150) < 1e-9


def test_band_pass(three_tones):
    y = F.band_pass(three_tones, FS, 100, 200)
    assert tone_amplitude(y, 150) == pytest.approx(1.0, abs=1e-9)
    assert tone_amplitude(y, 50) < 1e-9 and tone_amplitude(y, 300) < 1e-9


def test_band_stop(three_tones):
    y = F.band_stop(three_tones, FS, 100, 200)
    assert tone_amplitude(y, 150) < 1e-9
    assert tone_amplitude(y, 50) == pytest.approx(1.0, abs=1e-9)


def test_filters_return_real_same_length(three_tones):
    for y in (F.low_pass(three_tones, FS, 100), F.band_pass(three_tones, FS, 10, 20, transition=5)):
        assert y.dtype == np.float64 and y.size == three_tones.size


def test_low_plus_high_is_identity(three_tones):
    # complementary brick-walls partition the spectrum exactly
    lp = F.fft_filter(three_tones, FS, "lowpass", 100)
    resp_hp = 1 - F.frequency_response(np.fft.fftfreq(three_tones.size, 1 / FS), "lowpass", 100)
    hp = np.fft.ifft(np.fft.fft(three_tones) * resp_hp).real
    assert rel_err(lp + hp, three_tones) < 1e-12


def test_transition_band_is_monotone():
    f = np.linspace(0, 500, 1001)
    H = F.frequency_response(f, "lowpass", 100, transition=40)
    assert H[0] == 1.0 and H[-1] == 0.0
    assert np.all(np.diff(H) <= 1e-15)
    assert H[np.argmin(np.abs(f - 100))] == pytest.approx(0.5)


def test_bad_band_edges():
    with pytest.raises(ValueError):
        F.frequency_response(np.arange(10), "bandpass", (200, 100))


# --------------------------------------------------------------------------
# FIR design
# --------------------------------------------------------------------------

def test_fir_lowpass_response():
    taps = F.design_fir(101, 100, FS, "lowpass", "hamming")
    f, H = F.fir_frequency_response(taps, 8192, FS)
    assert np.sum(taps) == pytest.approx(1.0)
    assert np.allclose(taps, taps[::-1]), "linear phase => symmetric taps"
    assert np.all(np.abs(H[f < 70] - 1) < 0.01)
    assert H[f > 140].max() < 10 ** (-50 / 20)


def test_fir_window_attenuation_ordering():
    att = {}
    for w in ("rectangular", "hamming", "blackman"):
        f, H = F.fir_frequency_response(F.design_fir(101, 100, FS, "lowpass", w), 8192, FS)
        att[w] = 20 * np.log10(H[f > 160].max())
    assert att["blackman"] < att["hamming"] < att["rectangular"]
    assert att["rectangular"] > -45 and att["blackman"] < -70


@pytest.mark.parametrize("kind,cut,keep,remove", [
    ("highpass", 200, 300, 50),
    ("bandpass", (100, 200), 150, 300),
    ("bandstop", (100, 200), 300, 150),
])
def test_fir_kinds(kind, cut, keep, remove, three_tones):
    y = F.fir_filter(three_tones, F.design_fir(201, cut, FS, kind, "blackman"))
    mid = slice(200, -200)  # ignore filter start-up
    t = np.arange(three_tones.size)[mid] / FS
    a_keep = 2 * np.abs(np.mean(y[mid] * np.exp(-2j * np.pi * keep * t)))
    a_rem = 2 * np.abs(np.mean(y[mid] * np.exp(-2j * np.pi * remove * t)))
    assert a_keep == pytest.approx(1.0, abs=0.02)
    assert a_rem < 0.01


def test_fir_even_taps_rejected():
    with pytest.raises(ValueError):
        F.design_fir(100, 50, FS)


def test_moving_average():
    x = np.arange(20, dtype=float)
    y = F.moving_average(x, 5)
    assert np.allclose(y[2:-2], x[2:-2])


# --------------------------------------------------------------------------
# spectral analysis
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", ["rectangular", "hann", "blackman"])
def test_periodogram_parseval(window, rng):
    x = rng.standard_normal(1000)
    f, p = A.periodogram(x, FS, window="rectangular" if window == "rectangular" else window)
    if window == "rectangular":
        assert np.sum(p) * (f[1] - f[0]) == pytest.approx(np.var(x), rel=1e-10)
    else:
        # windowed: equal in expectation, not exactly
        assert np.sum(p) * (f[1] - f[0]) == pytest.approx(np.var(x), rel=0.15)


def test_periodogram_matches_scipy(rng):
    ss = pytest.importorskip("scipy.signal")
    x = rng.standard_normal(512)
    f, p = A.periodogram(x, FS)
    f2, p2 = ss.periodogram(x, FS)
    assert np.allclose(f, f2) and np.allclose(p, p2)


def test_welch_matches_scipy(rng):
    ss = pytest.importorskip("scipy.signal")
    x = rng.standard_normal(4096)
    f, p = A.welch_psd(x, FS, 256)
    f2, p2 = ss.welch(x, FS, nperseg=256)
    assert np.allclose(f, f2) and np.allclose(p, p2)


def test_welch_reduces_variance(rng):
    x = rng.standard_normal(16384)
    _, p1 = A.periodogram(x, FS)
    _, p2 = A.welch_psd(x, FS, 512)
    cv = lambda p: np.std(p[5:-5]) / np.mean(p[5:-5])
    assert cv(p2) < cv(p1) / 4


def test_amplitude_spectrum_scaling():
    _, x = G.sine_wave(125, 1.0, FS, amplitude=3.0)
    f, a = A.amplitude_spectrum(x, FS)
    assert a[np.argmin(np.abs(f - 125))] == pytest.approx(3.0, rel=1e-9)


@pytest.mark.parametrize("f0", [100.0, 123.4, 250.75, 333.3])
def test_peak_frequency_sub_bin(f0):
    _, x = G.sine_wave(f0, 1.0, FS)
    assert A.peak_frequency(x, FS) == pytest.approx(f0, abs=0.05)  # bin width is 1 Hz


def test_find_peaks_multiple(three_tones):
    f, a = A.amplitude_spectrum(three_tones, FS)
    found = sorted(p for p, _ in A.find_peaks(f, a, 3))
    assert np.allclose(found, [50, 150, 300], atol=0.01)


def test_stft_shape_and_istft_reconstruction(rng):
    x = rng.standard_normal(4000)
    f, t, Z = A.stft(x, FS, 256, 64)
    assert Z.shape == (129, len(t))
    y = A.istft(Z, FS, 256, 64, length=x.size)
    assert np.allclose(y[256:-256], x[256:-256], atol=1e-12)


def test_spectrogram_tracks_tone_switch():
    t = np.arange(2000) / FS
    x = np.where(t < 1, np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 300 * t))
    f, times, S = A.spectrogram(x, FS, 128, 32)
    early, late = times < 0.8, times > 1.2
    assert abs(f[np.argmax(S[:, early].mean(axis=1))] - 100) < 8
    assert abs(f[np.argmax(S[:, late].mean(axis=1))] - 300) < 8


@pytest.mark.parametrize("n", [1, 2, 50, 301])
def test_autocorrelation_fft_matches_direct(n, rng):
    x = rng.standard_normal(n)
    if n == 1:
        assert A.autocorrelation(x, normalize=False)[0] == 0.0
        return
    assert np.allclose(A.autocorrelation(x), A.autocorrelation_direct(x), atol=1e-12)


def test_autocorrelation_of_periodic_signal():
    _, x = G.sine_wave(10, 2.0, FS)  # period 100 samples
    r = A.autocorrelation(x)
    assert r[0] == 1.0
    assert r[100] > 0.9 and r[50] < -0.9


def test_band_power_and_centroid(three_tones):
    f, p = A.welch_psd(three_tones, FS, 1000)
    total = A.band_power(f, p, 0, FS / 2)
    assert A.band_power(f, p, 140, 160) == pytest.approx(total / 3, rel=0.05)
    assert A.spectral_centroid(f, p) == pytest.approx((50 + 150 + 300) / 3, rel=0.02)
