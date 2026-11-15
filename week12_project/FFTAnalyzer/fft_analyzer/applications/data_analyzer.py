"""
data_analyzer.py
-------------------
Part 5.3 of the FFTAnalyzer project. Frequency-domain time-series
analytics: detect periodicities with the periodogram, confirm them with
the autocorrelation (computed through the FFT, Wiener-Khinchin),
measure seasonality, decompose a series into trend + seasonal +
residual, and smooth it with convolution-based filters.

Two kinds of evidence for a period P, and why both are used:

  * Periodogram peak at frequency 1/P. Good at finding *several*
    periodicities at once, but its resolution is 1/n cycles per sample,
    so a weekly cycle in 2 years of daily data sits between bins
    730/7 = 104.3, and a long period like a year (1/365) is only ~2
    bins from zero, close to where the trend's energy piles up.
  * Autocorrelation peak at lag P. Directly in the units you care about
    (samples), and robust for long periods, but harmonics and multiples
    (a lag of 2P correlates too) make it ambiguous on its own.

detect_periodicities() takes candidates from the periodogram and
refines/verifies each one by looking for an ACF local maximum near the
candidate lag.

The trend is removed first (a linear fit, or a long moving average)
because a trend is a huge, very-low-frequency component that would
otherwise dominate the spectrum and the ACF.
"""

import csv

import numpy as np

from ..core.fft import linear_convolve
from ..signal import analysis
from ..signal.filters import get_window


# ---------------------------------------------------------------------------
# data loading / synthetic data
# ---------------------------------------------------------------------------

def load_csv_column(path, column, date_column=None):
    """
    Read one numeric column from a CSV with a header row. Returns
    (values, dates or None). Blank / non-numeric cells are linearly
    interpolated so the series stays evenly sampled, which the FFT needs.
    """
    values, dates = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                values.append(float(row[column]))
            except (ValueError, TypeError):
                values.append(np.nan)
            if date_column:
                dates.append(row[date_column])
    v = np.array(values, dtype=np.float64)
    bad = np.isnan(v)
    if bad.any() and (~bad).any():
        idx = np.arange(v.size)
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v, (dates if date_column else None)


def synthetic_sales(days=730, seed=0):
    """
    Two years of fake daily sales: linear growth, a weekly cycle (weekend
    peak), a yearly cycle (holiday season), and noise. Known ground truth:
    periods 7 and 365.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    trend = 200 + 0.15 * t
    weekly = 25 * np.sin(2 * np.pi * t / 7 + 0.6)
    yearly = 40 * np.sin(2 * np.pi * t / 365 - 1.2)
    noise = rng.normal(0, 8, days)
    return trend + weekly + yearly + noise


def synthetic_prices(days=1000, seed=1, cycle=21, cycle_amp=0.004):
    """
    Geometric-Brownian-motion style price series with a weak hidden
    monthly (21 trading day) cycle in its daily log returns. Known ground
    truth: period 21 in the returns, and nothing periodic in the price
    itself beyond a random walk.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    r = 0.0003 + 0.01 * rng.standard_normal(days) + cycle_amp * np.sin(2 * np.pi * t / cycle)
    return 100.0 * np.exp(np.cumsum(r))


def log_returns(prices):
    p = np.asarray(prices, dtype=np.float64)
    return np.diff(np.log(p))


# ---------------------------------------------------------------------------
# smoothing (convolution)
# ---------------------------------------------------------------------------

def _same(y, n, k):
    d = (k - 1) // 2
    return y[d:d + n]


def _edge_corrected(x, kernel):
    """
    Convolve, then divide by the convolution of a ones-vector with the
    same kernel. Near the ends the kernel hangs off the data; this
    re-normalizes by how much of the kernel actually overlapped, instead
    of treating the missing samples as zeros (which drags the ends
    toward zero).
    """
    x = np.asarray(x, dtype=np.float64)
    n, k = x.size, kernel.size
    num = _same(linear_convolve(x, kernel), n, k)
    den = _same(linear_convolve(np.ones(n), kernel), n, k)
    return num / den


def moving_average(x, window):
    """Centred box-car average via FFT convolution, edge-corrected."""
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window % 2 == 0:
        # an even box has no centre sample; use a 2 x window "centred" MA
        k = np.ones(window + 1)
        k[0] = k[-1] = 0.5
        k /= k.sum()
    else:
        k = np.ones(window) / window
    return _edge_corrected(x, k)


def gaussian_smooth(x, sigma):
    """Gaussian kernel smoothing (std `sigma` samples) via FFT convolution."""
    r = int(np.ceil(4 * sigma))
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return _edge_corrected(x, g / g.sum())


def windowed_smooth(x, width, window="hann"):
    """Smoothing with a Hann / Hamming / Blackman shaped kernel."""
    w = get_window(window, int(width) | 1)  # odd length
    return _edge_corrected(x, w / w.sum())


def exponential_smooth(x, alpha):
    """
    Exponential smoothing written as a convolution with the truncated
    kernel alpha (1 - alpha)^k, k = 0..K (causal, so it lags).
    """
    K = int(np.ceil(np.log(1e-6) / np.log(1 - alpha))) if 0 < alpha < 1 else 1
    k = alpha * (1 - alpha) ** np.arange(K + 1)
    x = np.asarray(x, dtype=np.float64)
    y = linear_convolve(x, k)[: x.size]
    norm = linear_convolve(np.ones(x.size), k)[: x.size]
    return y / norm


# ---------------------------------------------------------------------------
# periodicity / seasonality
# ---------------------------------------------------------------------------

def detrend(x, method="linear", window=None):
    """Remove a linear fit (default) or a long centred moving average."""
    x = np.asarray(x, dtype=np.float64)
    if method == "linear":
        t = np.arange(x.size)
        A = np.column_stack([t, np.ones_like(t)]).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ coef, A @ coef
    if method == "moving_average":
        trend = moving_average(x, window or max(3, x.size // 10))
        return x - trend, trend
    if method == "none":
        return x - x.mean(), np.full_like(x, x.mean())
    raise ValueError("method must be linear, moving_average or none")


def autocorrelation(x, max_lag=None):
    """Normalized ACF via FFT (Wiener-Khinchin), see signal/analysis.py."""
    return analysis.autocorrelation(x, normalize=True, max_lag=max_lag)


def _acf_peak_near(acf, lag, tol=0.15):
    """Highest local ACF maximum within lag * (1 +/- tol)."""
    lo = max(1, int(np.floor(lag * (1 - tol))))
    hi = min(acf.size - 2, int(np.ceil(lag * (1 + tol))))
    best = None
    for k in range(lo, hi + 1):
        if acf[k] >= acf[k - 1] and acf[k] >= acf[k + 1]:
            if best is None or acf[k] > acf[best]:
                best = k
    return best


def detect_periodicities(x, num=3, detrend_method="linear", min_period=2, max_period=None,
                         acf_threshold=None):
    """
    Find the strongest periodic components of an evenly sampled series.

    1. Detrend, then periodogram (Hann window, zero-padded 8x to
       interpolate the spectrum between bins).
    2. Take the top spectral peaks with period in [min_period, max_period]
       (default max: n / 2, need at least two full cycles).
    3. For each, look for an ACF local max within 15% of that lag. If one
       exists, clears `acf_threshold` (default 2.58 / sqrt(n), the 99%
       band for white noise's ACF) and hasn't already been claimed by a
       stronger period, the period is confirmed and reported as the
       spectral estimate rounded to an integer.

    The ACF is used to *confirm*, not to *locate*: an earlier version
    took the ACF peak's lag as the period, and it was consistently worse
    (343 for a true 365-day cycle seen only twice, 24 for a true 21-day
    cycle buried in noise), because with few cycles and noise the ACF
    peak is broad and its maximum drifts. The interpolated spectral peak
    was within 3% in both cases. The ACF lag is still reported, as
    `period_acf`.

    Returns a list of dicts {period, period_spectral, period_acf,
    frequency, power, acf, confirmed}, strongest first.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    resid, _ = detrend(x, detrend_method)
    max_period = max_period or n / 2
    w = get_window("hann", n, symmetric=False)
    m = 1
    while m < 8 * n:
        m *= 2
    buf = np.zeros(m)
    buf[:n] = resid * w
    from ..core.fft import fft
    P = np.abs(fft(buf)[: m // 2 + 1]) ** 2
    freqs = np.arange(m // 2 + 1) / m
    valid = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    P_masked = np.where(valid, P, 0.0)
    # a Hann main lobe is 4 bins wide (+/-2 bins of 1/n), so peaks closer
    # than ~2.5 bins are the same component's side of the lobe
    peaks = analysis.find_peaks(freqs, P_masked, num_peaks=num * 4,
                                min_distance_hz=2.5 / n, refine=True)
    acf = autocorrelation(resid)
    if acf_threshold is None:
        acf_threshold = 2.58 / np.sqrt(n)
    out = []
    used_lags = set()
    for f, p in peaks:
        if f <= 0:
            continue
        period_spec = 1.0 / f
        lag = _acf_peak_near(acf, period_spec)
        confirmed = lag is not None and acf[lag] >= acf_threshold and lag not in used_lags
        if confirmed:
            used_lags.add(lag)
        period = int(round(period_spec)) if confirmed else float(period_spec)
        if any(o["confirmed"] and o["period"] == period for o in out):
            continue
        out.append({
            "period": period,
            "period_spectral": float(period_spec),
            "period_acf": int(lag) if lag is not None else None,
            "frequency": float(f),
            "power": float(p),
            "acf": float(acf[lag]) if lag is not None else float("nan"),
            "confirmed": bool(confirmed),
        })
        if len(out) >= num:
            break
    return out


def seasonal_profile(x, period):
    """
    Average value at each phase of the cycle (e.g. each weekday), after
    removing the trend with a centred moving average one period long.
    Returns the profile, zero-mean, length `period`.
    """
    x = np.asarray(x, dtype=np.float64)
    period = int(round(period))
    trend = moving_average(x, period)
    resid = x - trend
    prof = np.array([resid[i::period].mean() for i in range(period)])
    return prof - prof.mean()


def decompose(x, period):
    """
    Classical additive decomposition x = trend + seasonal + residual:
      trend    = centred moving average of length `period` (convolution)
      seasonal = the seasonal profile tiled across the series
      residual = what's left
    Returns a dict of the three arrays plus the seasonal strength
    max(0, 1 - var(residual) / var(seasonal + residual))
    (Hyndman & Athanasopoulos), 0 = no seasonality, near 1 = strong.
    """
    x = np.asarray(x, dtype=np.float64)
    period = int(round(period))
    trend = moving_average(x, period)
    prof = seasonal_profile(x, period)
    seasonal = np.resize(prof, x.size)
    resid = x - trend - seasonal
    strength = max(0.0, 1.0 - np.var(resid) / np.var(seasonal + resid))
    return {"trend": trend, "seasonal": seasonal, "residual": resid,
            "period": period, "seasonal_strength": float(strength)}


# ---------------------------------------------------------------------------
# convenience wrapper
# ---------------------------------------------------------------------------

class TimeSeriesAnalyzer:
    """Ties the functions above to one series, with plotting."""

    def __init__(self, values, sample_spacing=1.0, name="series"):
        self.values = np.asarray(values, dtype=np.float64)
        self.dt = float(sample_spacing)
        self.name = name

    @classmethod
    def from_csv(cls, path, column, date_column=None, **kw):
        v, _ = load_csv_column(path, column, date_column)
        return cls(v, name=column, **kw)

    def periodicities(self, num=3, **kw):
        return detect_periodicities(self.values, num=num, **kw)

    def seasonality(self, period=None):
        """Decompose at `period`, or at the strongest confirmed detected period."""
        if period is None:
            period = self.dominant_period()
            if period is None:
                return None
        return decompose(self.values, period)

    def dominant_period(self, candidates=5):
        """
        The confirmed period with the strongest autocorrelation. Not simply
        the largest spectral peak: in two years of daily data the yearly
        cycle has the most power, but it's only seen twice, so it makes a
        poor period for a moving-average decomposition; the weekly cycle,
        seen 104 times, has ACF ~0.93 and is the one to decompose on.
        """
        conf = [p for p in self.periodicities(candidates) if p["confirmed"]]
        if not conf:
            return None
        return max(conf, key=lambda p: p["acf"])["period"]

    def smooth(self, method="moving_average", **kw):
        funcs = {"moving_average": moving_average, "gaussian": gaussian_smooth,
                 "window": windowed_smooth, "exponential": exponential_smooth}
        return funcs[method](self.values, **kw)

    def acf(self, max_lag=None):
        return autocorrelation(detrend(self.values)[0], max_lag)

    def plot_report(self, save_path=None, period=None, max_lag=None):
        """Four panels: series + smooth, periodogram, ACF, decomposition."""
        import matplotlib.pyplot as plt
        from ..visualization.plots import COLORS
        x = self.values
        n = x.size
        found = [p for p in self.periodicities(3) if p["confirmed"]]
        if period is None:
            period = self.dominant_period() or 7
        dec = decompose(x, period)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        a = axes[0, 0]
        a.plot(x, color=COLORS[5], lw=0.7, label="data")
        a.plot(dec["trend"], color=COLORS[0], lw=1.6, label=f"trend (MA {period})")
        a.set_title(f"{self.name}", loc="left", fontsize=11)
        a.legend(frameon=False)

        a = axes[0, 1]
        resid, _ = detrend(x)
        f, P = analysis.periodogram(resid, 1.0, window="hann")
        a.semilogy(f[1:], P[1:], color=COLORS[0], lw=0.8)
        for p in found:
            a.axvline(p["frequency"], color=COLORS[1], ls="--", lw=0.8)
            a.text(p["frequency"], P[1:].max(), f" P={p['period']}", fontsize=8,
                   color=COLORS[1], va="top")
        a.set_xlabel("frequency (cycles / sample)")
        a.set_title("Periodogram (detrended), confirmed periods marked", loc="left", fontsize=11)

        a = axes[1, 0]
        r = self.acf(max_lag or min(n - 1, 3 * max(period, 7) + 10))
        a.stem(np.arange(r.size), r, markerfmt=" ", basefmt=" ", linefmt=COLORS[0])
        a.axhline(1.96 / np.sqrt(n), color=COLORS[5], ls=":", lw=0.8)
        a.axhline(-1.96 / np.sqrt(n), color=COLORS[5], ls=":", lw=0.8)
        a.set_xlabel("lag")
        a.set_title("Autocorrelation (via FFT)", loc="left", fontsize=11)

        a = axes[1, 1]
        show = min(n, 6 * period)
        a.plot(dec["seasonal"][:show], color=COLORS[2], lw=1.2, label="seasonal")
        a.plot(dec["residual"][:show], color=COLORS[5], lw=0.6, label="residual")
        a.set_title(f"Seasonal component, period {period} "
                    f"(strength {dec['seasonal_strength']:.2f})", loc="left", fontsize=11)
        a.legend(frameon=False)
        for a in axes.ravel():
            a.spines[["top", "right"]].set_visible(False)
            a.grid(alpha=0.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig
