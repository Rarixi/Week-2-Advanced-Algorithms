from setuptools import find_packages, setup

setup(
    name="fft_analyzer",
    version="1.0.0",
    description="Numerical computing toolkit built around a from-scratch Cooley-Tukey FFT: "
                "matrix decompositions, stability analysis, signal and image processing.",
    packages=find_packages(include=["fft_analyzer", "fft_analyzer.*"]),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "matplotlib>=3.7"],
    extras_require={
        "images": ["Pillow>=9"],      # JPEG/other formats and GIF animations; PNG works without it
        "test": ["pytest>=7", "scipy>=1.10"],  # scipy is only a reference in the tests
    },
)
