"""Smoke tests: every demo runs, and the benchmark's quick mode produces its outputs."""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "examples", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["range_query_demo", "vcs_demo", "dna_demo", "cache_demo"])
def test_demo_runs(name, capsys):
    _load(name).main()
    assert capsys.readouterr().out.strip()


def test_sample_fasta_is_present_and_valid():
    from advds.applications.dna_index import read_fasta
    recs = read_fasta(os.path.join(ROOT, "data", "sample_fragment.fasta"))
    seq = recs["synthetic_fragment_50kb"]
    assert len(seq) == 50000 and set(seq) <= set("ACGT")


@pytest.mark.slow
def test_benchmark_quick_mode(tmp_path):
    """Runs the whole benchmark in --quick mode (~15 s). Skip with -m 'not slow'."""
    # writes to a temp folder so the committed full-run results are never overwritten
    out = subprocess.run([sys.executable, "-m", "advds.benchmarks.performance", "--quick",
                          "--out", str(tmp_path)],
                         cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    results = str(tmp_path)
    for f in ("summary.txt", "tables.md", "range_query_ops.png", "rank_select.png",
              "wavelet_compression.png", "cache_layouts.png", "persistent_overhead.png"):
        assert os.path.exists(os.path.join(results, f)), f
