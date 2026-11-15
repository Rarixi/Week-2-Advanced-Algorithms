"""
conftest.py
--------------
Shared fixtures for the FFTAnalyzer test suite. The independent
references the suite checks against are numpy's own np.fft and
np.linalg (LAPACK), plus the deliberately naive O(n^2) DFT and O(nm)
convolution in core/fft.py. scipy is used for a few extra cross-checks
(windows, DCT, Welch) only when it's installed; those tests skip
cleanly otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def test_image():
    from fft_analyzer.applications.image_processor import make_test_image
    return make_test_image(128)


def rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(b.ravel())
    return np.linalg.norm((a - b).ravel()) / (denom if denom else 1.0)
