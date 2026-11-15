"""
FFTAnalyzer: a numerical computing toolkit built around a from-scratch
Cooley-Tukey FFT.

    core/            fft.py, matrix.py, numerical.py
    signal/          filters.py, generators.py, analysis.py
    image/           filters.py, compression.py
    applications/    audio_processor.py, image_processor.py, data_analyzer.py
    visualization/   plots.py, animations.py

The most used entry points are re-exported here.
"""

from .core.fft import (
    circular_convolve,
    convolve2d,
    fft,
    fft2,
    fft_iterative,
    fft_recursive,
    fftfreq,
    ifft,
    ifft2,
    linear_convolve,
)
from .core.matrix import (
    determinant,
    lu_decompose,
    matmul,
    power_method,
    qr_decompose,
    solve,
    strassen,
)
from .core.numerical import condition_number, is_ill_conditioned, relative_error

__version__ = "1.0.0"

__all__ = [
    "fft", "ifft", "fft2", "ifft2", "fft_recursive", "fft_iterative", "fftfreq",
    "circular_convolve", "linear_convolve", "convolve2d",
    "matmul", "strassen", "lu_decompose", "solve", "determinant", "qr_decompose", "power_method",
    "condition_number", "is_ill_conditioned", "relative_error",
]
