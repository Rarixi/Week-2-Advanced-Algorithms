"""
dna_demo.py: wavelet-tree DNA index.

    python3 examples/dna_demo.py                     # bundled synthetic fragment
    python3 examples/dna_demo.py my_genome.fasta     # or any FASTA file (first record)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from advds.applications.dna_index import DNAIndex, naive_count, read_fasta

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample_fragment.fasta")


def main(path=DATA):
    records = read_fasta(path)
    name, seq = next(iter(records.items()))
    print(f"{name}: {len(seq):,} bases from {os.path.basename(path)}")
    t0 = time.perf_counter()
    idx = DNAIndex(seq)
    print(f"index built in {time.perf_counter() - t0:.2f}s")

    n = len(seq)
    print("\ncomposition of the whole sequence:", idx.composition(0, n - 1))
    print(f"GC content: {idx.gc_content(0, n - 1):.3f}")
    l, r = 1000, 1999
    print(f"A's in [{l}, {r}]: {idx.count('A', l, r)}")
    print(f"smallest, median, largest base in [{l}, {r}]:",
          idx.kth_smallest(l, r, 0), idx.median_base(l, r), idx.kth_smallest(l, r, r - l))
    gc = [(s, round(idx.gc_content(s, s + 9999), 3)) for s in range(0, n - 9999, 10000)]
    print("GC content per 10 kb window:", gc)

    print("\npattern frequency (FM-index backward search vs scanning the string):")
    for p in ["GATTACA", "TTAGGG", "CGCG", "ACGTACGTAC"]:
        t0 = time.perf_counter()
        c = idx.pattern_count(p)
        t_fm = (time.perf_counter() - t0) * 1e6
        t0 = time.perf_counter()
        c2 = naive_count(seq, p)
        t_scan = (time.perf_counter() - t0) * 1e6
        print(f"  {p:<11} {c:>4} occurrences  FM {t_fm:6.0f} us   scan {t_scan:6.0f} us   agree={c == c2}")
    print("  first GATTACA positions:", idx.find("GATTACA")[:10])

    rep = idx.size_report()
    print(f"\nsize: plain text {rep['plain_text_bits'] / 8 / 1024:.0f} KiB, wavelet tree "
          f"{rep['wavelet_tree_bits'] / 8 / 1024:.0f} KiB ({rep['wavelet_tree_bits_per_base']:.2f} bits/base), "
          f"FM-index {rep['fm_index_bits'] / 8 / 1024:.0f} KiB ({rep['fm_index_bits_per_base']:.2f} bits/base)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA)
