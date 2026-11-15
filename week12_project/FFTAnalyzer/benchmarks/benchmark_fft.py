"""
benchmark_fft.py
-------------------
Benchmarks and stability experiments for the report. Run from the
FFTAnalyzer/ folder:

    python3 benchmarks/benchmark_fft.py            # full run, ~1-2 minutes
    python3 benchmarks/benchmark_fft.py --quick    # smaller sizes, a few seconds

Writes to benchmarks/results/:
  fft_runtime.png / fft_runtime.csv           naive DFT vs recursive vs iterative vs Bluestein vs numpy
  fft_accuracy.png / fft_accuracy.csv         error vs n: O(n^2) DFT vs FFT
  convolution_runtime.png / .csv              direct O(nm) vs FFT convolution
  matmul_runtime.png / matmul_runtime.csv     triple loop vs vectorized vs Strassen vs numpy
  qr_stability.png / qr_stability.csv         orthogonality loss vs cond (CGS / MGS / Householder)
  condition_numbers.png / .csv                Hilbert, Vandermonde, random matrices
  lu_error_vs_condition.png                   forward error and residual vs cond
  summary.txt                                 the headline numbers quoted in the report

Timing is the best of several repeats (min, not mean: the minimum is
the run least disturbed by everything else on the machine).
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fft_analyzer.core import fft as F
from fft_analyzer.core import matrix as M
from fft_analyzer.core import numerical as N
from fft_analyzer.visualization import plots as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
rng = np.random.default_rng(0)
summary = []


def best_time(fn, *args, repeats=3, budget=2.0):
    """Min wall time over up to `repeats` runs (fewer if one run is slow)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
        if sum(times) > budget:
            break
    return min(times)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def log(line):
    print(line)
    summary.append(line)


# ---------------------------------------------------------------------------
# 1. FFT runtime
# ---------------------------------------------------------------------------

def bench_fft_runtime():
    exps = range(4, 17 if QUICK else 21)
    sizes = [2 ** e for e in exps]
    limits = {"naive DFT": 2 ** 12, "recursive FFT": 2 ** (14 if QUICK else 18),
              "iterative FFT": 2 ** 20, "Bluestein (n+1)": 2 ** 20, "numpy.fft": 2 ** 20}
    funcs = {"naive DFT": F.dft, "recursive FFT": F.fft_recursive, "iterative FFT": F.fft_iterative,
             "Bluestein (n+1)": None, "numpy.fft": np.fft.fft}
    times = {k: [] for k in funcs}
    rows = []
    for n in sizes:
        x = rng.standard_normal(n)
        row = [n]
        for name, fn in funcs.items():
            if n > limits[name]:
                row.append("")
                continue
            if name == "Bluestein (n+1)":
                xb = rng.standard_normal(n + 1)  # a non-power-of-two length
                t = best_time(F.fft, xb)
            else:
                t = best_time(fn, x)
            times[name].append(t)
            row.append(f"{t:.6g}")
        rows.append(row)
        print(f"  fft n=2^{int(np.log2(n)):2d}: " + "  ".join(f"{k}={v[-1]:.4g}s" for k, v in times.items()
                                                              if n <= limits[k]))
    write_csv("fft_runtime.csv", ["n"] + list(funcs), rows)

    # reference lines: n log n anchored on the iterative FFT, n^2 on the DFT
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate(funcs):
        ax.loglog(sizes[: len(times[k])], times[k], "o-", ms=3.5, lw=1.3, color=P.COLORS[i], label=k)
    s = np.array(sizes, dtype=float)
    it = np.array(times["iterative FFT"])
    ax.loglog(s, s * np.log2(s) / (s[-1] * np.log2(s[-1])) * it[-1], "--", color=P.COLORS[5], lw=0.8)
    ax.text(s[-1], it[-1] * 0.5, "n log n", fontsize=8, color=P.COLORS[5], ha="right")
    d = np.array(times["naive DFT"])
    sd = s[: d.size]
    ax.loglog(s[:d.size + 3], (s[:d.size + 3] / sd[-1]) ** 2 * d[-1], ":", color=P.COLORS[5], lw=0.8)
    ax.text(s[d.size + 2], (s[d.size + 2] / sd[-1]) ** 2 * d[-1], " n^2", fontsize=8, color=P.COLORS[5])
    P._style(ax, "FFT runtime, best of 3", "n (samples)", "seconds")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "fft_runtime.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    n_dft = limits["naive DFT"]
    k = sizes.index(n_dft)
    log(f"FFT, n=2^20: iterative {times['iterative FFT'][-1]:.3f}s, Bluestein (n=2^20+1) "
        f"{times['Bluestein (n+1)'][-1]:.3f}s, numpy {times['numpy.fft'][-1]:.4f}s "
        f"({times['iterative FFT'][-1] / times['numpy.fft'][-1]:.0f}x numpy)")
    log(f"FFT vs naive DFT at n={n_dft}: DFT {times['naive DFT'][-1]:.3f}s, iterative FFT "
        f"{times['iterative FFT'][k]:.5f}s ({times['naive DFT'][-1] / times['iterative FFT'][k]:.0f}x faster)")
    r = len(times["recursive FFT"]) - 1
    log(f"recursive vs iterative at n={sizes[r]}: {times['recursive FFT'][r]:.3f}s vs "
        f"{times['iterative FFT'][r]:.4f}s ({times['recursive FFT'][r] / times['iterative FFT'][r]:.0f}x)")
    if not QUICK:
        tA, tB = times["iterative FFT"][sizes.index(2 ** 18)], times["iterative FFT"][-1]
        log(f"iterative FFT 2^18 -> 2^20 (4x data): {tB / tA:.2f}x time (n log n predicts 4.44x)")
    return times


# ---------------------------------------------------------------------------
# 2. FFT accuracy
# ---------------------------------------------------------------------------

def bench_fft_accuracy():
    sizes = [2 ** e for e in range(4, 13 if QUICK else 15)]
    rows = []
    e_dft, e_fft, e_rt = [], [], []
    for n in sizes:
        x = rng.standard_normal(n)
        # reference: numpy's pocketfft, itself an O(eps log n) accurate FFT
        ref = np.fft.fft(x)
        e_fft.append(N.relative_error(F.fft(x), ref))
        e_rt.append(N.relative_error(F.ifft(F.fft(x)), x))
        if n <= 4096:
            # the textbook O(n^2) sum, accumulated term by term in a loop over
            # j (what a direct implementation does), not the matrix product
            acc = np.zeros(n, complex)
            k = np.arange(n)
            for j in range(n):
                acc += x[j] * np.exp(-2j * np.pi * ((j * k) % n) / n)
            e_dft.append(N.relative_error(acc, ref))
        rows.append([n, e_fft[-1], e_rt[-1], e_dft[-1] if n <= 4096 else ""])
    write_csv("fft_accuracy.csv", ["n", "fft_vs_numpy", "ifft_fft_roundtrip", "dft_vs_numpy"], rows)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.loglog(sizes, e_fft, "o-", ms=3.5, color=P.COLORS[0], label="iterative FFT vs numpy")
    ax.loglog(sizes, e_rt, "s-", ms=3.5, color=P.COLORS[2], label="ifft(fft(x)) vs x")
    ax.loglog(sizes[: len(e_dft)], e_dft, "^-", ms=3.5, color=P.COLORS[1], label="direct O(n^2) sum vs numpy")
    s = np.array(sizes, float)
    eps = np.finfo(float).eps
    ax.loglog(s, eps * np.log2(s), "--", color=P.COLORS[5], lw=0.8, label="eps log2 n")
    ax.loglog(s, eps * np.sqrt(s), ":", color=P.COLORS[5], lw=0.8, label="eps sqrt(n)")
    P._style(ax, "Relative error vs transform size", "n", "relative error")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "fft_accuracy.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    i = sizes.index(4096)
    log(f"FFT accuracy at n=4096: FFT vs numpy {e_fft[i]:.2e}, direct DFT sum vs numpy {e_dft[i]:.2e}; "
        f"round trip at n={sizes[-1]}: {e_rt[-1]:.2e}")


# ---------------------------------------------------------------------------
# 3. convolution
# ---------------------------------------------------------------------------

def bench_convolution():
    n = 2 ** 14
    x = rng.standard_normal(n)
    ks = [2 ** e for e in range(1, 12 if QUICK else 14)]
    t_dir, t_fft = [], []
    rows = []
    for k in ks:
        h = rng.standard_normal(k)
        t_dir.append(best_time(F.direct_convolve, x, h))
        t_fft.append(best_time(F.linear_convolve, x, h))
        rows.append([k, t_dir[-1], t_fft[-1]])
    write_csv("convolution_runtime.csv", ["kernel_length", "direct_s", "fft_s"], rows)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.loglog(ks, t_dir, "o-", ms=3.5, color=P.COLORS[1], label="direct (vectorized over signal)")
    ax.loglog(ks, t_fft, "o-", ms=3.5, color=P.COLORS[0], label="FFT (zero-padded)")
    P._style(ax, f"Linear convolution, signal length {n}", "kernel length", "seconds")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, "convolution_runtime.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    cross = next((k for k, a, b in zip(ks, t_dir, t_fft) if b < a), None)
    log(f"convolution (signal 2^14): FFT wins from kernel length {cross}; at k={ks[-1]} "
        f"direct {t_dir[-1]:.3f}s vs FFT {t_fft[-1]:.4f}s")


# ---------------------------------------------------------------------------
# 4. matrix multiplication
# ---------------------------------------------------------------------------

def bench_matmul():
    sizes = [16, 32, 64, 128, 256] + ([] if QUICK else [512])
    limits = {"triple loop (pure Python)": 128 if QUICK else 256, "vectorized i-k-j": 512,
              "Strassen (cutoff 64)": 512, "numpy @ (BLAS)": 512}
    funcs = {"triple loop (pure Python)": M.matmul_naive, "vectorized i-k-j": M.matmul,
             "Strassen (cutoff 64)": lambda A, B: M.strassen(A, B, 64), "numpy @ (BLAS)": np.matmul}
    times = {k: [] for k in funcs}
    errs = {k: [] for k in funcs}
    rows = []
    for n in sizes:
        A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        ref = A @ B
        row = [n]
        for name, fn in funcs.items():
            if n > limits[name]:
                row.append("")
                continue
            t = best_time(fn, A, B, budget=5.0)
            times[name].append(t)
            errs[name].append(N.relative_error(fn(A, B), ref))
            row.append(f"{t:.6g}")
        rows.append(row)
        print(f"  matmul n={n}: " + "  ".join(f"{k.split()[0]}={v[-1]:.4g}s" for k, v in times.items()
                                                 if n <= limits[k]))
    write_csv("matmul_runtime.csv", ["n"] + list(funcs), rows)
    fig = P.plot_runtime(sizes, [times[k] for k in funcs], list(funcs), "Matrix multiplication runtime",
                         reference=[("n^3", lambda n: n ** 3.0), ("n^2.807", lambda n: n ** 2.807)])
    fig.savefig(os.path.join(OUT, "matmul_runtime.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    m = sizes[-1]
    log(f"matmul n={m}: vectorized {times['vectorized i-k-j'][-1]:.3f}s, Strassen "
        f"{times['Strassen (cutoff 64)'][-1]:.3f}s, numpy {times['numpy @ (BLAS)'][-1]:.4f}s; "
        f"triple loop at n={limits['triple loop (pure Python)']}: {times['triple loop (pure Python)'][-1]:.2f}s")
    log(f"matmul error vs numpy at n={m}: vectorized {errs['vectorized i-k-j'][-1]:.1e}, "
        f"Strassen {errs['Strassen (cutoff 64)'][-1]:.1e}")


# ---------------------------------------------------------------------------
# 5. stability: condition numbers, QR orthogonality, LU error
# ---------------------------------------------------------------------------

def bench_stability():
    ns = list(range(2, 14))
    conds = {"Hilbert": [], "Vandermonde (equispaced [0,1])": [], "random Gaussian": []}
    rows = []
    for n in ns:
        H = N.hilbert_matrix(n)
        V = N.vandermonde_matrix(np.linspace(0, 1, n))
        R = np.random.default_rng(n).standard_normal((n, n))
        c = [N.condition_number(H), N.condition_number(V), N.condition_number(R)]
        for k, v in zip(conds, c):
            conds[k].append(v)
        rows.append([n] + c + [np.linalg.cond(H)])
    write_csv("condition_numbers.csv", ["n", "hilbert", "vandermonde", "random", "hilbert_numpy"], rows)
    fig = P.plot_condition_numbers(ns, list(conds.values()), list(conds),
                                   save_path=os.path.join(OUT, "condition_numbers.png"))
    plt.close(fig)
    log(f"cond_2(Hilbert): n=8 {conds['Hilbert'][6]:.2e}, n=10 {conds['Hilbert'][8]:.2e}, "
        f"n=12 {conds['Hilbert'][10]:.2e} (1/eps = {1 / np.finfo(float).eps:.1e})")

    # QR orthogonality loss on Hilbert matrices
    qn = list(range(2, 13))
    loss = {"classical Gram-Schmidt": [], "modified Gram-Schmidt": [], "Householder": []}
    fns = [M.qr_classical_gram_schmidt, M.qr_modified_gram_schmidt, M.qr_householder]
    hc = []
    rows = []
    for n in qn:
        H = N.hilbert_matrix(n)
        hc.append(np.linalg.cond(H))
        row = [n, hc[-1]]
        for (k, f) in zip(loss, fns):
            try:
                Q, _ = f(H)
                loss[k].append(np.linalg.norm(Q.T @ Q - np.eye(n)))
            except np.linalg.LinAlgError:
                loss[k].append(np.nan)
            row.append(loss[k][-1])
        rows.append(row)
    write_csv("qr_stability.csv", ["n", "cond", "cgs", "mgs", "householder"], rows)
    fig = P.plot_error_vs_condition(hc, list(loss.values()), list(loss),
                                    "QR on Hilbert matrices: ||Q^T Q - I||_F vs cond",
                                    save_path=os.path.join(OUT, "qr_stability.png"))
    plt.close(fig)
    i = qn.index(10)
    log(f"QR orthogonality loss, Hilbert n=10 (cond {hc[i]:.1e}): CGS {loss['classical Gram-Schmidt'][i]:.1e}, "
        f"MGS {loss['modified Gram-Schmidt'][i]:.1e}, Householder {loss['Householder'][i]:.1e}")

    # LU solve: forward error and residual vs condition number
    fe, res = [], []
    for n in qn:
        H = N.hilbert_matrix(n)
        rpt = N.stability_report(H)
        fe.append(rpt["forward_relative_error"])
        res.append(rpt["relative_residual"])
    fig = P.plot_error_vs_condition(hc, [fe, res], ["forward error ||x - x*|| / ||x*||",
                                                    "relative residual ||b - Ax|| / (||A|| ||x||)"],
                                    "LU with partial pivoting on Hilbert systems",
                                    save_path=os.path.join(OUT, "lu_error_vs_condition.png"))
    plt.close(fig)
    log(f"LU solve, Hilbert n=10: forward error {fe[i]:.1e}, residual {res[i]:.1e}; "
        f"n=12: forward error {fe[qn.index(12)]:.1e}")

    # pivoting
    A = np.array([[1e-20, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    L, U = M.lu_no_pivot(A)
    xb = M.back_substitution(U, M.forward_substitution(L, b, unit_diagonal=True))
    log(f"tiny pivot [[1e-20,1],[1,1]] x=[1,1]: no pivoting gives {xb.round(6).tolist()}, "
        f"partial pivoting gives {M.solve(A, b).round(12).tolist()}")

    for r in N.cancellation_examples():
        log(f"cancellation: {r['name']:30s} naive rel err {r['naive_rel_err']:.1e}, stable {r['stable_rel_err']:.1e}")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    print("FFT runtime...")
    bench_fft_runtime()
    print("FFT accuracy...")
    bench_fft_accuracy()
    print("convolution...")
    bench_convolution()
    print("matrix multiplication...")
    bench_matmul()
    print("stability...")
    bench_stability()
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(f"# benchmark summary ({'quick' if QUICK else 'full'} run, numpy {np.__version__}, "
                 f"python {sys.version.split()[0]})\n")
        fh.write("\n".join(summary) + "\n")
    print(f"done in {time.perf_counter() - t0:.1f}s, results in {OUT}")


if __name__ == "__main__":
    main()
