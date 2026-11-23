"""
cache_demo.py: cache-oblivious transpose, multiplication and vEB layout,
measured with the ideal-cache simulator.

    python3 examples/cache_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from advds.cache_oblivious import matrix_ops as M
from advds.cache_oblivious.cache_sim import CacheHierarchy, simulate
from advds.cache_oblivious.layouts import compare_layouts, veb_order


def main():
    n = 256
    print(f"transpose of a {n}x{n} matrix, 32 KiB cache, 64-byte lines:")
    for m in ("naive", "blocked", "recursive"):
        print(f"  {m:<10} {simulate(M.trace_transpose(n, m, block=8), 32 * 1024, 64):>8} misses")
    print(f"  (compulsory: {2 * n * n * 8 // 64})")

    n = 64
    print(f"\nmatrix multiply {n}x{n}, L1 8 KiB + L2 64 KiB:")
    for m in ("ijk", "ikj", "blocked", "recursive"):
        h = CacheHierarchy((("L1", 8 * 1024, 64), ("L2", 64 * 1024, 64)))
        print(f"  {m:<10} {h.run(M.trace_matmul(n, m, block=8))}")

    print("\nvan Emde Boas order of a height-4 tree (heap indices):", veb_order(4).tolist())
    print("\nmisses per search in a 65,535-key search tree (cold cache):")
    for row in compare_layouts(16, num_queries=500, block_sizes=(64, 4096)):
        print(f"  {row['layout']:<24} block {row['block_bytes']:>5} B: {row['cold_misses']:.2f}")


if __name__ == "__main__":
    main()
