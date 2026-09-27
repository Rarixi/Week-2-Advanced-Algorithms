"""
week4_graph_benchmark.py
-------------------------
Part 4 deliverable: benchmarks and visualizes BFS, DFS, and Dijkstra (Parts 1-3) across
four kinds of test graphs built by src/utils/graph_generator.py (sparse, dense, random,
and weighted).

What this script does:
    1. Generates sparse / dense / random / weighted graphs at a few sizes.
    2. Benchmarks BFS vs. DFS: mean traversal TIME (time.perf_counter) and peak MEMORY
       usage (tracemalloc) for each graph type/size.
    3. Benchmarks Dijkstra's heap-based implementation across all four graph densities.
    4. Visualizes one small graph's BFS and DFS traversal order with networkx + matplotlib
       (node color/label = the step at which each node was first visited).
    5. Saves performance plots (time and memory vs. graph size, Dijkstra vs. density) as
       PNGs under results/.
    6. Exports every benchmark row to results/comparison_table.csv.

Run with:
    python3 week4_graph_benchmark.py

(Needs networkx and matplotlib in addition to the stdlib -- both are third-party
dependencies this script assumes are already installed, same as any other package this
project needs.)
"""

import csv
import os
import random
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ^ This file lives in benchmarks/, one level below the project root where src/ actually
#   is -- ".." goes up to that root so "from src.graphs..." imports below can resolve.

import matplotlib
matplotlib.use("Agg")  # write PNGs directly, no need for a display
import matplotlib.pyplot as plt
import networkx as nx

from src.utils.graph_generator import (
    generate_sparse_graph,
    generate_dense_graph,
    generate_random_graph,
    generate_weighted_graph,
)
from src.graphs.bfs import bfs_full
from src.graphs.dfs import dfs_full
from src.graphs.dijkstra import dijkstra

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(42)

GRAPH_TYPES = {
    "sparse": generate_sparse_graph,
    "dense": generate_dense_graph,
    "random": generate_random_graph,
    "weighted": generate_weighted_graph,
}
SIZES = [100, 400, 1000]
TIME_REPEATS = 3


def count_edges(graph):
    """Real out-degree sum -- generators' requested edge counts can undercount slightly
    due to random duplicate picks (see graph_generator.py's docstring)."""
    return sum(len(graph.get_neighbors(node)) for node in graph.nodes)


def measure_time_and_memory(fn, repeats=TIME_REPEATS):
    """Mean wall-clock time over `repeats` calls, and PEAK traced memory from one separate
    call. These are measured separately on purpose: tracemalloc's own bookkeeping adds
    overhead that would inflate the timing numbers if both were measured in the same call."""
    total_time = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        total_time += time.perf_counter() - start
    mean_time = total_time / repeats

    tracemalloc.start()
    fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return mean_time, peak


def main():
    rows = []
    graphs = {}  # (graph_type, n) -> Graph, reused between the BFS/DFS and Dijkstra sections

    # --- Part A: generate graphs, benchmark BFS vs. DFS (time + memory) ---
    print("### BFS vs. DFS: time and memory, across graph types and sizes ###")
    for graph_type, generator in GRAPH_TYPES.items():
        for n in SIZES:
            g = generator(n)
            graphs[(graph_type, n)] = g
            edges = count_edges(g)

            bfs_time, bfs_mem = measure_time_and_memory(lambda: bfs_full(g))
            dfs_time, dfs_mem = measure_time_and_memory(lambda: dfs_full(g, algorithm="iterative"))

            rows.append({
                "graph_type": graph_type, "n": n, "edges": edges, "algorithm": "bfs",
                "time_us": round(bfs_time * 1e6, 2), "peak_memory_kb": round(bfs_mem / 1024, 2),
            })
            rows.append({
                "graph_type": graph_type, "n": n, "edges": edges, "algorithm": "dfs",
                "time_us": round(dfs_time * 1e6, 2), "peak_memory_kb": round(dfs_mem / 1024, 2),
            })
            print(
                f"{graph_type:9s} n={n:5d} edges={edges:8d}  "
                f"bfs={rows[-2]['time_us']:10.2f}us/{rows[-2]['peak_memory_kb']:8.2f}KB   "
                f"dfs={rows[-1]['time_us']:10.2f}us/{rows[-1]['peak_memory_kb']:8.2f}KB"
            )

    # --- Part B: Dijkstra across the same four densities ---
    print("\n### Dijkstra (heap-based): time and memory, by graph density ###")
    for (graph_type, n), g in graphs.items():
        edges = count_edges(g)
        dijkstra_time, dijkstra_mem = measure_time_and_memory(lambda: dijkstra(g, 0))
        rows.append({
            "graph_type": graph_type, "n": n, "edges": edges, "algorithm": "dijkstra",
            "time_us": round(dijkstra_time * 1e6, 2), "peak_memory_kb": round(dijkstra_mem / 1024, 2),
        })
        print(
            f"{graph_type:9s} n={n:5d} edges={edges:8d}  "
            f"dijkstra={rows[-1]['time_us']:10.2f}us/{rows[-1]['peak_memory_kb']:8.2f}KB"
        )

    # --- Export comparison_table.csv ---
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_type", "n", "edges", "algorithm", "time_us", "peak_memory_kb"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # --- Visualization 1: BFS/DFS traversal order on one small, readable graph ---
    random.seed(99)  # independent of the benchmark loop above, so this example graph's
                      # exact shape doesn't shift if the benchmark sizes/order change later
    small_graph = generate_sparse_graph(12, edges_per_node=2, directed=True, weighted=True)

    nx_graph = nx.DiGraph()
    for node in small_graph.nodes:
        nx_graph.add_node(node)
    for node in small_graph.nodes:
        for neighbor, weight in small_graph.get_neighbors(node).items():
            nx_graph.add_edge(node, neighbor, weight=weight)

    layout = nx.spring_layout(nx_graph, seed=42)  # same layout reused for both plots below,
                                                    # so BFS vs. DFS order is a fair visual
                                                    # comparison on identical node positions

    bfs_order = bfs_full(small_graph)
    dfs_order = dfs_full(small_graph, algorithm="iterative")
    # bfs_full/dfs_full (not plain bfs/dfs_iterative) guarantee every node gets a visit
    # order, even if small_graph isn't fully connected from a single source.

    for label, order in [("bfs", bfs_order), ("dfs", dfs_order)]:
        order_index = {node: i for i, node in enumerate(order)}
        node_colors = [order_index[node] for node in nx_graph.nodes()]

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_edges(nx_graph, layout, alpha=0.35, arrows=True, arrowsize=12)
        nodes_drawn = nx.draw_networkx_nodes(
            nx_graph, layout, node_color=node_colors, cmap=plt.cm.viridis, node_size=600
        )
        nx.draw_networkx_labels(
            nx_graph, layout,
            labels={node: f"{node}\n#{order_index[node]}" for node in nx_graph.nodes()},
            font_size=8, font_color="white",
        )
        plt.colorbar(nodes_drawn, label="Visit order (0 = visited first)")
        plt.title(f"{label.upper()} traversal order, source = node 0")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"{label}_traversal_order.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Wrote {out_path}")

    # --- Visualization 2: performance plots ---
    def rows_for(graph_type, algorithm):
        matching = [r for r in rows if r["graph_type"] == graph_type and r["algorithm"] == algorithm]
        matching.sort(key=lambda r: r["n"])
        return [r["n"] for r in matching], [r["time_us"] for r in matching], [r["peak_memory_kb"] for r in matching]

    # Time vs. n: BFS and DFS, sparse vs. dense (the two extremes -- the full picture,
    # including random and weighted, is in comparison_table.csv).
    plt.figure(figsize=(8, 6))
    for graph_type in ["sparse", "dense"]:
        for algorithm, style in [("bfs", "-o"), ("dfs", "--s")]:
            ns, times, _ = rows_for(graph_type, algorithm)
            plt.plot(ns, times, style, label=f"{graph_type} / {algorithm}")
    plt.xlabel("n (number of nodes)")
    plt.ylabel("mean time (microseconds)")
    plt.yscale("log")
    plt.title("BFS vs. DFS traversal time: sparse vs. dense graphs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bfs_dfs_time_vs_n.png"), dpi=150)
    plt.close()

    # Memory vs. n: same shape, peak memory instead of time.
    plt.figure(figsize=(8, 6))
    for graph_type in ["sparse", "dense"]:
        for algorithm, style in [("bfs", "-o"), ("dfs", "--s")]:
            ns, _, mems = rows_for(graph_type, algorithm)
            plt.plot(ns, mems, style, label=f"{graph_type} / {algorithm}")
    plt.xlabel("n (number of nodes)")
    plt.ylabel("peak traced memory (KB)")
    plt.yscale("log")
    plt.title("BFS vs. DFS peak memory: sparse vs. dense graphs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bfs_dfs_memory_vs_n.png"), dpi=150)
    plt.close()

    # Dijkstra vs. density: grouped bar chart, one group of bars per graph type, one bar
    # per size within each group.
    plt.figure(figsize=(9, 6))
    graph_type_names = list(GRAPH_TYPES.keys())
    bar_width = 0.25
    x_positions = range(len(graph_type_names))
    for i, n in enumerate(SIZES):
        heights = []
        for graph_type in graph_type_names:
            match = next(r for r in rows if r["graph_type"] == graph_type and r["n"] == n and r["algorithm"] == "dijkstra")
            heights.append(match["time_us"])
        offsets = [x + (i - 1) * bar_width for x in x_positions]
        plt.bar(offsets, heights, width=bar_width, label=f"n={n}")
    plt.xticks(list(x_positions), graph_type_names)
    plt.ylabel("mean time (microseconds)")
    plt.yscale("log")
    plt.title("Dijkstra (heap-based) runtime by graph density")
    plt.legend(title="graph size")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dijkstra_density_comparison.png"), dpi=150)
    plt.close()

    print(f"\nAll plots written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
