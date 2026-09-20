"""
week1_representation_benchmark.py
----------------------------------
Part 1 deliverable (benchmark half): compares the adjacency LIST vs. adjacency MATRIX
representations built into src/graphs/graph.py -- construction time, memory footprint, and
get_neighbors() lookup time -- for n = 100, 1,000, 10,000, as required by the Part 1 spec.

Each graph gets a fixed ~2 edges/node (same sparse-ish density graph_generator.py uses for
"sparse"), so both representations are storing the same topology; the only thing that
differs is HOW they store it. Matrix cells are preallocated for all n*n pairs regardless of
edge count, so its memory cost is dominated by n, not by how many edges actually exist --
that's the whole point of the comparison.

Run with: python3 week1_representation_benchmark.py
"""

import csv
import os
import random
import sys
import time
import tracemalloc

from src.graphs.graph import Graph

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZES = [100, 1_000, 10_000]
EDGES_PER_NODE = 2
LOOKUP_SAMPLES = 2_000  # random get_neighbors() calls averaged for the timing figure


def build_graph(representation, n, rng):
    """Directed graph, n nodes, ~EDGES_PER_NODE*n edges, no self-loops, no duplicate
    (u, v) pairs -- same edge SET is used for both representations (rng is re-seeded
    identically before each build) so list vs. matrix are compared on identical topology."""
    g = Graph(directed=True, representation=representation)
    for i in range(n):
        g.add_node(i)

    target_edges = n * EDGES_PER_NODE
    seen = set()
    while len(seen) < target_edges:
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v or (u, v) in seen:
            continue
        seen.add((u, v))
        g.add_edge(u, v, weight=rng.randint(1, 20))
    return g


def measure(representation, n):
    tracemalloc.start()
    start = time.perf_counter()
    g = build_graph(representation, n, random.Random(42))
    build_time = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Lookup timing: get_neighbors() on LOOKUP_SAMPLES random existing nodes (sampled with
    # replacement), timed as a batch and divided down to a mean per-call time in microseconds.
    sample_rng = random.Random(7)
    nodes = list(range(n))
    lookup_nodes = [sample_rng.choice(nodes) for _ in range(LOOKUP_SAMPLES)]
    start = time.perf_counter()
    for node in lookup_nodes:
        g.get_neighbors(node)
    lookup_time = (time.perf_counter() - start) / LOOKUP_SAMPLES

    return {
        "representation": representation,
        "n": n,
        "edges": target_edges_for(n),
        "build_time_s": round(build_time, 4),
        "peak_memory_kb": round(peak / 1024, 2),
        "mean_lookup_time_us": round(lookup_time * 1e6, 3),
    }


def target_edges_for(n):
    return n * EDGES_PER_NODE


def main():
    rows = []
    print("### Adjacency list vs. adjacency matrix: build time, memory, lookup time ###")
    for n in SIZES:
        for representation in ["list", "matrix"]:
            row = measure(representation, n)
            rows.append(row)
            print(
                f"{representation:7s} n={n:6d}  "
                f"build={row['build_time_s']:8.4f}s  "
                f"peak_mem={row['peak_memory_kb']:12.2f}KB  "
                f"lookup={row['mean_lookup_time_us']:8.3f}us"
            )

    csv_path = os.path.join(RESULTS_DIR, "representation_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["representation", "n", "edges", "build_time_s", "peak_memory_kb", "mean_lookup_time_us"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Plot: memory and lookup time vs. n, list vs. matrix, log-scale y axes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(representation, field):
        matching = [r for r in rows if r["representation"] == representation]
        matching.sort(key=lambda r: r["n"])
        return [r["n"] for r in matching], [r[field] for r in matching]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for representation, style in [("list", "-o"), ("matrix", "--s")]:
        ns, mem = series(representation, "peak_memory_kb")
        axes[0].plot(ns, mem, style, label=representation)
    axes[0].set_xlabel("n (number of nodes)")
    axes[0].set_ylabel("peak traced memory (KB)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Memory: adjacency list vs. matrix")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for representation, style in [("list", "-o"), ("matrix", "--s")]:
        ns, lookup = series(representation, "mean_lookup_time_us")
        axes[1].plot(ns, lookup, style, label=representation)
    axes[1].set_xlabel("n (number of nodes)")
    axes[1].set_ylabel("mean get_neighbors() time (microseconds)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Lookup time: adjacency list vs. matrix")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "representation_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
