# FFTAnalyzer

## Description
A numerical computing toolkit built around a from-scratch Cooley-Tukey
FFT: the linear algebra underneath it, the signal and image processing
on top of it, three applications, visualization and animation, a test
suite and a benchmark suite, and a written report. No `np.fft`,
`np.linalg` or scipy calls inside the library itself (they appear only
in the tests, as independent references).

- **FFT** (`core/fft.py`): recursive and iterative (bit-reversal)
  radix-2 Cooley-Tukey, Bluestein's algorithm for any length, inverse
  FFT by the conjugation trick, 2D FFT, circular convolution, linear
  convolution by zero-padding, 2D convolution. 2^20 points in about
  0.17 s.
- **Matrices** (`core/matrix.py`): classical O(n^3) multiplication (pure
  Python and vectorized), Strassen, LU with partial pivoting (and
  without, to show why pivoting matters), solve / determinant / inverse
  via LU, QR by classical and modified Gram-Schmidt and by Householder,
  least squares via QR, power method, inverse iteration, and unshifted
  QR iteration.
- **Stability** (`core/numerical.py`): machine epsilon measured
  empirically, absolute / relative / ULP error, L1 / L2 / Linf and matrix
  norms, condition numbers (exact, and Hager's O(n^2) estimator), an
  "is this matrix ill-conditioned?" check, and five catastrophic
  cancellation examples, each with a naive and a stable formula.
- **Signals** (`signal/`): sine / multi-tone / chirp / white / pink
  noise generators; FFT low / high / band-pass / band-stop filters;
  Hann, Hamming, Blackman windows; windowed-sinc FIR design; amplitude
  spectrum, periodogram, Welch PSD, STFT / inverse STFT, spectrogram,
  sub-bin peak detection, FFT autocorrelation.
- **Images** (`image/`): Gaussian blur, unsharp-mask sharpening, Sobel
  and Laplacian edges, low-pass and notch de-noising, all in the
  frequency domain; 2D DCT (via FFT and as an 8x8 matrix), JPEG
  quantization tables with quality scaling, YCbCr and 4:2:0 chroma
  subsampling, PSNR and an entropy estimate of the compressed size.
- **Applications** (`applications/`): `AudioProcessor` (WAV in/out with
  the standard library, hum removal, filtering, spectral-subtraction
  denoising, spectrograms), `ImageProcessor` (filters, compression,
  quality sweeps, saving reconstructions and coefficients),
  `TimeSeriesAnalyzer` (periodicity detection, seasonal decomposition,
  convolution smoothing, FFT autocorrelation).
- **Visualization** (`visualization/`): spectra, spectrograms, 2D FFT
  heatmaps, filter responses, condition-number and error plots; GIF
  animations of the FFT's butterfly stages and of a low-pass sweeping
  across a square wave.

## Structure
```
FFTAnalyzer/
├── fft_analyzer/
│   ├── core/            # fft.py, matrix.py, numerical.py
│   ├── signal/          # filters.py, generators.py, analysis.py
│   ├── image/           # filters.py, compression.py
│   ├── applications/    # audio_processor.py, image_processor.py, data_analyzer.py
│   └── visualization/   # plots.py, animations.py
├── tests/               # 290 tests: test_fft, test_matrix, test_filters, test_image_processing, test_numerical
├── examples/            # signal_demo.py, audio_demo.py, image_demo.py (+ output/ once run)
├── benchmarks/          # benchmark_fft.py, results/ (figures, CSVs, summary.txt)
├── docs/                # chapter11_report.md
├── setup.py
└── README.md
```

## Setup
```bash
pip install numpy matplotlib
pip install Pillow          # optional: JPEG/other image formats, and GIF animations
pip install pytest scipy    # for the tests; scipy is optional (12 cross-checks skip without it)
```
Or `pip install -e .[test,images]` from this folder.

## Running Tests
From inside `FFTAnalyzer/`:
```bash
pytest tests/
```
All 290 tests pass (278 pass and 12 skip if scipy isn't installed).
They cover FFT correctness against numpy and the O(n^2) DFT for
power-of-two and arbitrary lengths, the DFT's algebraic properties
(Parseval, conjugate symmetry, shift theorem), the convolution theorem,
the 2^20 performance goal, every decomposition against LAPACK, the
expected stability behaviour (pivoting, Gram-Schmidt vs Householder,
condition numbers), filter behaviour on known tones, compression error
vs quality, and every plot and animation rendering to a file.

## Running the Demos
```bash
python3 examples/signal_demo.py
python3 examples/audio_demo.py      # or: python3 examples/audio_demo.py your_clip.wav
python3 examples/image_demo.py      # or: python3 examples/image_demo.py your_photo.png
```
Each writes figures (and WAVs / GIFs / a compressed image) to
`examples/output/`. With no arguments the audio and image demos use
built-in synthetic inputs, so nothing needs downloading.

## Running the Benchmark
```bash
python3 benchmarks/benchmark_fft.py          # ~30 s
python3 benchmarks/benchmark_fft.py --quick  # a few seconds
```
Writes runtime, accuracy, convolution, matrix multiplication,
QR-stability, condition-number and LU-error plots plus CSVs and a
`summary.txt` to `benchmarks/results/`.

Headline numbers from the run in `results/`: at n = 2^20 the iterative
FFT took 0.175 s (numpy: 0.019 s), at n = 4096 it was about 1,600x
faster than the naive DFT, and the recursive FFT was about 22x slower
than the iterative one. On the 10x10 Hilbert matrix (cond 1.6e13),
||Q^T Q - I|| was 3.5 for classical Gram-Schmidt, 4.4e-4 for modified,
and 1.7e-15 for Householder.

## A note on scope
- Strassen is implemented (it was optional). Eigenvalues go as far as
  the power method, inverse iteration and *unshifted* QR iteration; the
  unshifted version can't separate eigenvalues of equal magnitude (a
  test documents this), which shifts would fix.
- The JPEG pipeline stops before entropy coding: no zig-zag run-length
  + Huffman bitstream. The compressed size is estimated from the
  Shannon entropy of the quantized coefficients, and
  `ImageProcessor.compress(save_coefficients=...)` saves the actual
  coefficients as a zlib-compressed `.npz`.
- No sample audio or images ship with the project; the demos generate
  synthetic inputs with known ground truth, so results (SNR, RMSE,
  detected periods) can be scored, not just looked at.

## Report
`docs/chapter11_report.md` covers the FFT and linear algebra theory,
spectral analysis, the audio and image results, the stability results
(including three real bugs that only showed up on hard inputs), the
benchmark plots, and final reflections.
