"""
benchmark_search.py
-----------------------
Part 5 of the string search project. Runs KMP, Rabin-Karp, the suffix
array, and the suffix tree against the same generated texts and the
same search pattern, measuring preprocessing time, search latency, and
memory separately, since those three numbers tell very different
stories, KMP and Rabin-Karp have essentially no preprocessing (their
"setup" scales with the pattern, not the text), while the suffix array
and suffix tree pay real, text sized preprocessing cost upfront in
exchange for near free repeated queries afterward, exactly the tradeoff
core/search_engine.py's recommend() is built to surface.

One deliberate scope decision, carried over honestly rather than hidden:
the suffix tree here (core/suffix_tree.py, the compressed trie option)
has O(n^2) worst case construction time, so it gets its own, smaller
size sweep (TREE_SIZES) rather than running it out to the same 1,000,000
character ceiling KMP, Rabin-Karp, and the suffix array reach. Running it
at that scale would mean the benchmark either taking many extra minutes
or silently cutting corners on what "medium" means, neither of which
belongs in a results table.

The 100MB "large" dataset tier the assignment lists as optional was
skipped entirely, for the same honest reason, scaled up, suffix array
construction in particular would turn a few second benchmark run into
one lasting many minutes without teaching this report anything the
smaller sizes don't already show clearly.
"""

import csv
import os
import random
import time
import tracemalloc

from core.kmp import kmp_search
from core.rabin_karp import collision_rate_demo, rabin_karp_search
from core.suffix_array import build_lcp_array, build_suffix_array, sa_search
from core.suffix_tree import build_suffix_trie, count_nodes, substring_positions
from utils.generators import random_string, text_with_planted_pattern
from visualization.draw_suffix_tree import draw_suffix_tree
from visualization.plot_suffix_array import plot_suffix_array

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PATTERN = "benchmarkpat"  # 12 characters, fixed across every run for a clean apples to apples comparison
PLANTED_OCCURRENCES = 5
ALPHABET = "abcd"

# KMP, Rabin-Karp, and the suffix array all run the full sweep, "small"
# (10KB) through the low end of "medium" (1MB).
SIZES = [10_000, 50_000, 100_000, 300_000, 600_000, 1_000_000]

# The suffix tree's O(n^2) worst case construction gets its own, smaller
# sweep, a subset of SIZES so results merge into one table cleanly.
TREE_SIZES = [10_000, 50_000, 100_000]

COLLISION_MODULI = [101, 1009, 10007, 104729, (1 << 31) - 1, (1 << 61) - 1]


def _time(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _peak_memory_kb(fn, *args, **kwargs):
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def benchmark_scaling():
    """KMP vs Rabin-Karp vs suffix array across SIZES. Every text has
    PATTERN planted at PLANTED_OCCURRENCES known positions, so every
    algorithm's result gets checked against that ground truth, not just
    against each other."""
    rng = random.Random(42)
    rows = []
    print("Running scaling sweep (KMP vs Rabin-Karp vs suffix array)...")
    for size in SIZES:
        text, planted = text_with_planted_pattern(
            size, PATTERN, PLANTED_OCCURRENCES, alphabet=ALPHABET, rng=rng
        )

        kmp_matches, kmp_time = _time(kmp_search, text, PATTERN)
        assert sorted(kmp_matches) == planted
        kmp_memory = _peak_memory_kb(kmp_search, text, PATTERN)

        rk_result, rk_time = _time(rabin_karp_search, text, PATTERN)
        assert sorted(rk_result["matches"]) == planted
        rk_memory = _peak_memory_kb(rabin_karp_search, text, PATTERN)

        sa, sa_build_time = _time(build_suffix_array, text)
        lcp, lcp_build_time = _time(build_lcp_array, text, sa)
        sa_preprocess_time = sa_build_time + lcp_build_time
        sa_matches, sa_search_time = _time(sa_search, text, sa, PATTERN)
        assert sorted(sa_matches) == planted
        sa_memory = _peak_memory_kb(build_suffix_array, text)

        rows.append({
            "size": size,
            "kmp_time": kmp_time, "kmp_memory_kb": kmp_memory,
            "rk_time": rk_time, "rk_memory_kb": rk_memory,
            "sa_preprocess_time": sa_preprocess_time, "sa_search_time": sa_search_time,
            "sa_memory_kb": sa_memory,
            "text": text, "sa": sa, "lcp": lcp,
        })
        print(
            f"  size={size:>9}  kmp={kmp_time:.4f}s  rk={rk_time:.4f}s  "
            f"sa_build={sa_preprocess_time:.4f}s  sa_search={sa_search_time:.6f}s"
        )
    return rows


def benchmark_tree_scaling():
    """The suffix tree's own, smaller sweep over TREE_SIZES."""
    rng = random.Random(43)
    rows = []
    print("Running suffix tree sweep (smaller sizes, O(n^2) worst case construction)...")
    for size in TREE_SIZES:
        text, planted = text_with_planted_pattern(
            size, PATTERN, PLANTED_OCCURRENCES, alphabet=ALPHABET, rng=rng
        )
        root, build_time = _time(build_suffix_trie, text)
        matches, search_time = _time(substring_positions, text, root, PATTERN)
        assert sorted(matches) == planted
        memory = _peak_memory_kb(build_suffix_trie, text)
        nodes = count_nodes(root)

        rows.append({
            "size": size,
            "tree_preprocess_time": build_time, "tree_search_time": search_time,
            "tree_memory_kb": memory, "tree_nodes": nodes,
        })
        print(f"  size={size:>9}  build={build_time:.4f}s  search={search_time:.6f}s  nodes={nodes}")
    return rows


def benchmark_collisions():
    """Rabin-Karp's collision rate at a range of moduli, from a
    deliberately tiny 101 up to the project's real default of 2^61 - 1,
    the direct empirical version of the modulus choice story told in
    core/rabin_karp.py's module docstring."""
    print("Running Rabin-Karp collision analysis across moduli...")
    rng = random.Random(44)
    text = random_string(20_000, alphabet=ALPHABET, rng=rng)
    pattern = text[5000:5006]
    demo = collision_rate_demo(text, pattern, moduli=COLLISION_MODULI)
    for modulus, stats in demo.items():
        print(f"  modulus={modulus:<20} collision_rate={stats['collision_rate']:.4f}")
    return demo


def save_runtime_comparison_figure(scaling_rows, tree_rows, out_path):
    sizes = [r["size"] for r in scaling_rows]
    kmp_times = [r["kmp_time"] for r in scaling_rows]
    rk_times = [r["rk_time"] for r in scaling_rows]
    sa_total = [r["sa_preprocess_time"] + r["sa_search_time"] for r in scaling_rows]
    tree_sizes = [r["size"] for r in tree_rows]
    tree_total = [r["tree_preprocess_time"] + r["tree_search_time"] for r in tree_rows]

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, kmp_times, marker="o", label="KMP")
    plt.plot(sizes, rk_times, marker="o", label="Rabin-Karp")
    plt.plot(sizes, sa_total, marker="o", label="Suffix array (build + search)")
    plt.plot(tree_sizes, tree_total, marker="s", linestyle="--", label="Suffix tree (build + search)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("text size (characters)")
    plt.ylabel("total time (s)")
    plt.title("Runtime comparison: KMP vs Rabin-Karp vs suffix array vs suffix tree")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_collision_analysis_figure(demo, out_path):
    moduli = list(demo.keys())
    rates = [demo[m]["collision_rate"] for m in moduli]

    plt.figure(figsize=(9, 5))
    plt.bar([str(m) for m in moduli], rates, color="crimson")
    plt.xlabel("modulus")
    plt.ylabel("collision rate (hash matches that failed verification)")
    plt.title("Rabin-Karp collision rate by modulus")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_sa_lcp_stats_figure(scaling_rows, out_path):
    # The smallest scaling instance makes the clearest picture, plot_suffix_array
    # already truncates to the first 80 ranks for readability.
    smallest = scaling_rows[0]
    plot_suffix_array(
        smallest["text"], smallest["sa"], smallest["lcp"], out_path=out_path,
        title=f"Suffix array and LCP ({smallest['size']} character text)",
    )


def save_heatmap_figure(scaling_rows, tree_rows, out_path):
    algorithms = ["KMP", "Rabin-Karp", "Suffix array", "Suffix tree"]
    all_sizes = sorted({r["size"] for r in scaling_rows} | {r["size"] for r in tree_rows})
    matrix = np.full((len(algorithms), len(all_sizes)), np.nan)

    scaling_by_size = {r["size"]: r for r in scaling_rows}
    tree_by_size = {r["size"]: r for r in tree_rows}

    for j, size in enumerate(all_sizes):
        if size in scaling_by_size:
            r = scaling_by_size[size]
            matrix[0, j] = r["kmp_time"]
            matrix[1, j] = r["rk_time"]
            matrix[2, j] = r["sa_preprocess_time"] + r["sa_search_time"]
        if size in tree_by_size:
            t = tree_by_size[size]
            matrix[3, j] = t["tree_preprocess_time"] + t["tree_search_time"]

    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels([str(s) for s in all_sizes], rotation=45, ha="right")
    ax.set_xlabel("text size (characters)")
    ax.set_title("Total time (build + search) by algorithm and dataset size, blank = not run at that size")
    fig.colorbar(im, ax=ax, label="total time (s)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def export_csv(scaling_rows, tree_rows, collision_demo, path):
    tree_by_size = {r["size"]: r for r in tree_rows}
    fieldnames = [
        "size", "kmp_time_s", "kmp_memory_kb", "rk_time_s", "rk_memory_kb",
        "sa_preprocess_time_s", "sa_search_time_s", "sa_memory_kb",
        "tree_preprocess_time_s", "tree_search_time_s", "tree_memory_kb", "tree_nodes",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scaling_rows:
            t = tree_by_size.get(r["size"])
            writer.writerow({
                "size": r["size"],
                "kmp_time_s": r["kmp_time"], "kmp_memory_kb": r["kmp_memory_kb"],
                "rk_time_s": r["rk_time"], "rk_memory_kb": r["rk_memory_kb"],
                "sa_preprocess_time_s": r["sa_preprocess_time"], "sa_search_time_s": r["sa_search_time"],
                "sa_memory_kb": r["sa_memory_kb"],
                "tree_preprocess_time_s": t["tree_preprocess_time"] if t else "",
                "tree_search_time_s": t["tree_search_time"] if t else "",
                "tree_memory_kb": t["tree_memory_kb"] if t else "",
                "tree_nodes": t["tree_nodes"] if t else "",
            })

    collision_path = path.replace("comparison_table.csv", "collision_table.csv")
    with open(collision_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modulus", "hash_matches", "collisions", "collision_rate", "verifications"])
        writer.writeheader()
        for modulus, stats in collision_demo.items():
            writer.writerow({"modulus": modulus, **stats})


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scaling_rows = benchmark_scaling()
    tree_rows = benchmark_tree_scaling()
    collision_demo = benchmark_collisions()

    print("Saving figures...")
    save_runtime_comparison_figure(scaling_rows, tree_rows, os.path.join(RESULTS_DIR, "runtime_comparison.png"))
    save_collision_analysis_figure(collision_demo, os.path.join(RESULTS_DIR, "collision_analysis.png"))
    save_sa_lcp_stats_figure(scaling_rows, os.path.join(RESULTS_DIR, "sa_lcp_stats.png"))
    save_heatmap_figure(scaling_rows, tree_rows, os.path.join(RESULTS_DIR, "heatmap_dataset_sizes.png"))

    print("Rendering a small suffix tree demo plot...")
    demo_text = "bananarama"
    demo_root = build_suffix_trie(demo_text)
    draw_suffix_tree(demo_text, demo_root, out_path=os.path.join(RESULTS_DIR, "suffix_tree_demo.png"))

    print("Writing comparison table...")
    export_csv(scaling_rows, tree_rows, collision_demo, os.path.join(RESULTS_DIR, "comparison_table.csv"))

    largest = scaling_rows[-1]
    largest_tree = tree_rows[-1]
    print("\n" + "=" * 70)
    print("CHAPTER 10 STRING SEARCH BENCHMARK - SUMMARY")
    print("=" * 70)
    print(f"Largest full-sweep instance: {largest['size']} characters")
    print(f"  KMP: {largest['kmp_time']:.4f}s")
    print(f"  Rabin-Karp: {largest['rk_time']:.4f}s")
    print(f"  Suffix array: build {largest['sa_preprocess_time']:.4f}s, search {largest['sa_search_time']:.6f}s")
    print(f"Largest suffix tree instance: {largest_tree['size']} characters")
    print(f"  Suffix tree: build {largest_tree['tree_preprocess_time']:.4f}s, "
          f"search {largest_tree['tree_search_time']:.6f}s, {largest_tree['tree_nodes']} nodes")
    print(f"Every algorithm found exactly the {PLANTED_OCCURRENCES} planted occurrences on every instance: True")
    print("=" * 70)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
