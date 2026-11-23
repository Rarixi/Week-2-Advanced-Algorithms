from setuptools import find_packages, setup

setup(
    name="advds",
    version="1.0.0",
    description="Advanced data structures: segment/Fenwick/persistent trees, succinct rank/select "
                "and wavelet trees, cache-oblivious algorithms, with applications.",
    packages=find_packages(include=["advds", "advds.*"]),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "matplotlib>=3.7"],
    extras_require={"test": ["pytest>=7"]},
)
