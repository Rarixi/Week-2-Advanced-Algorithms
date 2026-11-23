"""Bit vector, rank/select (vs numpy cumulative sums), wavelet tree (vs brute force)."""
from collections import Counter

import numpy as np
import pytest

from advds.succinct.bit_vector import BitVector
from advds.succinct.rank_select import (NaiveRankSelect, RankSelect, popcount64,
                                        select_in_word)
from advds.succinct.wavelet_tree import WaveletTree


# --- bit vector ------------------------------------------------------------

def test_bitvector_basics():
    bv = BitVector.from_string("10110")
    assert len(bv) == 5 and bv.to_list() == [1, 0, 1, 1, 0] and bv.count() == 3
    bv.set(1)
    bv.clear(0)
    bv.flip(4)
    assert str(bv) == "01111"
    with pytest.raises(IndexError):
        bv[5]


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 1000])
def test_bitvector_numpy_round_trip(n):
    bits = np.random.default_rng(n).random(n) < 0.3
    bv = BitVector.from_numpy(bits)
    assert np.array_equal(bv.to_numpy(), bits)
    assert bv.size_in_bits() == 64 * ((n + 63) // 64)


def test_tables():
    rng = np.random.default_rng(0)
    for w in rng.integers(0, 2 ** 63, 200, dtype=np.int64).tolist() + [0, 2 ** 64 - 1]:
        assert popcount64(w) == bin(w).count("1")
        ones = [i for i in range(64) if (w >> i) & 1]
        for r, pos in enumerate(ones):
            assert select_in_word(w, r) == pos


# --- rank / select ------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 63, 64, 65, 511, 512, 513, 4096, 20000])
@pytest.mark.parametrize("density", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_rank_select_exhaustive(n, density):
    bits = np.random.default_rng(n).random(n) < density
    rs = RankSelect(BitVector.from_numpy(bits))
    cs = np.concatenate([[0], np.cumsum(bits)])
    for i in range(n + 1):
        assert rs.rank1(i) == cs[i] and rs.rank0(i) == i - cs[i]
    ones, zeros = np.flatnonzero(bits), np.flatnonzero(~bits)
    for k in range(1, ones.size + 1):
        assert rs.select1(k) == ones[k - 1]
    for k in range(1, zeros.size + 1):
        assert rs.select0(k) == zeros[k - 1]


def test_rank_select_inverse():
    bits = np.random.default_rng(1).random(5000) < 0.4
    rs = RankSelect(BitVector.from_numpy(bits))
    for k in range(1, rs.ones + 1, 7):
        assert rs.rank1(rs.select1(k)) == k - 1
        assert rs.access(rs.select1(k)) == 1


def test_rank_select_doc_example():
    rs = RankSelect(BitVector.from_string("0110100111"))
    assert (rs.rank1(5), rs.select1(3), rs.rank0(5), rs.select0(2)) == (3, 4, 2, 3)


def test_select_out_of_range():
    rs = RankSelect(BitVector.from_string("0101"))
    with pytest.raises(IndexError):
        rs.select1(3)
    with pytest.raises(IndexError):
        rs.select0(0)
    with pytest.raises(IndexError):
        rs.rank1(5)


def test_naive_matches_indexed():
    bits = np.random.default_rng(2).random(3000) < 0.5
    bv = BitVector.from_numpy(bits)
    a, b = RankSelect(bv), NaiveRankSelect(bv)
    for i in range(0, 3001, 37):
        assert a.rank1(i) == b.rank1(i)
    for k in range(1, a.ones + 1, 41):
        assert a.select1(k) == b.select1(k)
    for k in range(1, a.zeros + 1, 41):
        assert a.select0(k) == b.select0(k)


def test_overhead_is_sublinear_fraction():
    rs = RankSelect(BitVector.from_numpy(np.random.default_rng(3).random(1 << 20) < 0.5))
    assert 0.3 < rs.overhead_ratio() < 0.6


# --- wavelet tree ---------------------------------------------------------------

@pytest.mark.parametrize("shape", ["balanced", "huffman"])
@pytest.mark.parametrize("seed", range(8))
def test_wavelet_against_brute_force(shape, seed):
    rng = np.random.default_rng(seed)
    n, sigma = int(rng.integers(1, 250)), int(rng.integers(1, 17))
    s = rng.integers(0, sigma, n) if seed % 2 else np.minimum(rng.geometric(0.4, n) - 1, sigma - 1)
    wt = WaveletTree(s, sigma, shape)
    L = s.tolist()
    assert wt.to_list() == L
    for c in range(sigma):
        occ = [i for i, x in enumerate(L) if x == c]
        assert wt.count(c) == len(occ)
        for i in range(0, n + 1, max(1, n // 13)):
            assert wt.rank(c, i) == L[:i].count(c)
        for k in range(1, len(occ) + 1):
            assert wt.select(c, k) == occ[k - 1]
    for _ in range(30):
        l = int(rng.integers(0, n))
        r = int(rng.integers(l, n))
        sub = L[l:r + 1]
        c = int(rng.integers(0, sigma))
        assert wt.range_count(l, r, c) == sub.count(c)
        assert wt.range_frequencies(l, r) == dict(Counter(sub))
        if shape == "balanced":
            k = int(rng.integers(0, len(sub)))
            assert wt.quantile(l, r, k) == sorted(sub)[k]
            a = int(rng.integers(0, sigma))
            b = int(rng.integers(a, sigma))
            assert wt.range_count_values(l, r, a, b) == sum(a <= x <= b for x in sub)


def test_wavelet_doc_example():
    wt = WaveletTree([3, 1, 4, 1, 5, 2, 6, 5], sigma=7)
    assert (wt.access(2), wt.rank(1, 4), wt.select(5, 2), wt.range_count(0, 7, 5),
            wt.quantile(2, 6, 0), wt.range_count_values(0, 7, 2, 4)) == (4, 2, 7, 2, 1, 3)
    assert (wt.range_min(0, 7), wt.range_max(0, 7), wt.range_median(0, 7)) == (1, 6, 3)


def test_wavelet_text_and_compression():
    text = "abracadabra " * 300 + "the quick brown fox jumps over the lazy dog " * 50
    bal = WaveletTree.from_text(text)
    huf = WaveletTree.from_text(text, shape="huffman")
    assert bal.to_text() == text and huf.to_text() == text
    assert bal.count("a") == text.count("a") and huf.rank("r", 100) == text[:100].count("r")
    sb, sh = bal.compression_stats(), huf.compression_stats()
    assert sb["payload_bits"] <= len(text) * sb["fixed_width_bits_per_symbol"]   # midpoint splits: some codes are shorter
    # Huffman shape: payload within 1 bit/symbol of the entropy, below the balanced shape
    assert sh["h0_bits_per_symbol"] <= sh["payload_bits_per_symbol"] < sh["h0_bits_per_symbol"] + 1
    assert sh["payload_bits"] < sb["payload_bits"]
    with pytest.raises(ValueError):
        huf.quantile(0, 10, 0)


def test_wavelet_edge_cases():
    wt = WaveletTree([0, 0, 0], sigma=1)
    assert wt.to_list() == [0, 0, 0] and wt.select(0, 3) == 2 and wt.quantile(0, 2, 1) == 0
    assert WaveletTree([2, 2], sigma=5).rank(4, 2) == 0
    with pytest.raises(ValueError):
        WaveletTree([5], sigma=3)
    with pytest.raises(IndexError):
        WaveletTree([1, 2], sigma=3).select(0, 1)
