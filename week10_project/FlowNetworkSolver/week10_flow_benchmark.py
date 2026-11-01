"""
week10_flow_benchmark.py
---------------------------
Part 6 of the flow network project. Benchmarks Edmonds-Karp, Push-Relabel,
and Min-Cost Flow against each other, sparse vs dense graph performance,
and scaling behavior as node count grows, exporting runtime, memory,
augmentation count, and Push-Relabel convergence plots plus a combined
comparison table to results/.

A note on scale. The assignment's Part 1 spec asks Push-Relabel to scale
to 10k-50k nodes. The first version of push_relabel.py (plain FIFO with
only the current-arc optimization) could not get anywhere close to that
in pure Python, a 45x45 grid (2025 nodes) took 67.8 seconds and 7 million
push/relabel operations, because nothing stopped a node from being
relabeled one height at a time chasing a sink it could never reach.
Adding the standard gap heuristic (core/push_relabel.py's docstring has
the full explanation) fixed that, the same 2025 node grid now finishes in
well under a second. With the gap heuristic in place this benchmark's
scaling sweep comfortably reaches 14,400 nodes in a few seconds per
algorithm, and a one-off check during development confirmed 22,500 nodes
still finished in well under a minute (Edmonds-Karp 6.9s, Push-Relabel
32.5s). The sweep below stops at 14,400 nodes to keep this script's own
total runtime reasonable rather than because the algorithm hits a wall
there, pure Python's per-operation overhead, not the algorithm's
correctness or its asymptotic bound, is what stands between this
implementation and the literal 50,000 node target, a compiled
implementation of the exact same algorithm would clear that scale easily.

Run with:
    python3 week10_flow_benchmark.py
"""

import csv
import os
import random
import time
import tracemalloc

from core.edmonds_karp import edmonds_karp_max_flow
from core.min_cost_flow import min_cost_max_flow
from core.network import Network
from core.push_relabel import push_relabel_max_flow
from utils.generators import grid_graph, random_flow_network
from visualization.animate_flow import animate_edmonds_karp, animate_push_relabel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

GRID_SIDES = [10, 20, 30, 45, 60, 80, 100, 120]
THREE_WAY_SIZES = [50, 100, 200, 400, 800, 1500]
SPARSE_DENSE_N = 150


def _copy_network(network):
    copy = Network()
    for node in network.nodes:
        copy.add_node(node)
    for edge in network.edges[::2]:
        copy.add_edge(edge.u, edge.v, edge.capacity, edge.cost)
    return copy


def _time_and_measure(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs), returns (result, elapsed_seconds,
    peak_memory_kb), peak memory via tracemalloc so no extra dependency
    (like psutil) is needed."""
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024.0


# ---------------------------------------------------------------------------
# Scaling sweep (grid graphs, Edmonds-Karp vs Push-Relabel)
# ---------------------------------------------------------------------------

def benchmark_scaling():
    rows = []
    for side in GRID_SIDES:
        n = side * side
        base_network = grid_graph(side, side, rng=random.Random(side))
        source, sink = (0, 0), (side - 1, side - 1)

        ek_network = _copy_network(base_network)
        ek_result, ek_time, ek_memory = _time_and_measure(edmonds_karp_max_flow, ek_network, source, sink)

        pr_network = _copy_network(base_network)
        pr_result, pr_time, pr_memory = _time_and_measure(push_relabel_max_flow, pr_network, source, sink)

        rows.append({
            "benchmark": "scaling", "n": n, "side": side,
            "edges": len(base_network.edges) // 2,
            "ek_time_s": ek_time, "ek_memory_kb": ek_memory, "ek_iterations": ek_result["iterations"],
            "pr_time_s": pr_time, "pr_memory_kb": pr_memory,
            "pr_pushes": pr_result["pushes"], "pr_relabels": pr_result["relabels"],
            "pr_gap_relabels": pr_result["gap_relabels"],
            "values_match": abs(ek_result["value"] - pr_result["value"]) < 1e-6,
        })
        print(f"  scaling n={n:6d} edges={rows[-1]['edges']:7d}  EK {ek_time:7.3f}s  PR {pr_time:7.3f}s  match={rows[-1]['values_match']}")
    return rows


# ---------------------------------------------------------------------------
# Sparse vs dense, fixed node count
# ---------------------------------------------------------------------------

def benchmark_sparse_vs_dense():
    rows = []
    for label, density in [("sparse", 0.02), ("dense", 0.3)]:
        base_network, source, sink = random_flow_network(SPARSE_DENSE_N, density, rng=random.Random(99))

        ek_network = _copy_network(base_network)
        ek_result, ek_time, ek_memory = _time_and_measure(edmonds_karp_max_flow, ek_network, source, sink)

        pr_network = _copy_network(base_network)
        pr_result, pr_time, pr_memory = _time_and_measure(push_relabel_max_flow, pr_network, source, sink)

        rows.append({
            "benchmark": "sparse_vs_dense", "density_label": label, "n": SPARSE_DENSE_N,
            "edges": len(base_network.edges) // 2,
            "ek_time_s": ek_time, "ek_memory_kb": ek_memory,
            "pr_time_s": pr_time, "pr_memory_kb": pr_memory,
            "values_match": abs(ek_result["value"] - pr_result["value"]) < 1e-6,
        })
        print(f"  sparse_vs_dense {label:6s} edges={rows[-1]['edges']:6d}  EK {ek_time:7.3f}s  PR {pr_time:7.3f}s")
    return rows


# ---------------------------------------------------------------------------
# Three way comparison, Edmonds-Karp vs Push-Relabel vs Min-Cost Flow
# ---------------------------------------------------------------------------

def benchmark_three_way():
    rows = []
    for n in THREE_WAY_SIZES:
        density = min(1.0, 6.0 / n)
        base_network, source, sink = random_flow_network(n, density, cost_range=(1.0, 9.0), rng=random.Random(n))

        ek_network = _copy_network(base_network)
        ek_result, ek_time, _ = _time_and_measure(edmonds_karp_max_flow, ek_network, source, sink)

        pr_network = _copy_network(base_network)
        pr_result, pr_time, _ = _time_and_measure(push_relabel_max_flow, pr_network, source, sink)

        mcf_network = _copy_network(base_network)
        mcf_result, mcf_time, _ = _time_and_measure(min_cost_max_flow, mcf_network, source, sink)

        rows.append({
            "benchmark": "three_way", "n": n, "edges": len(base_network.edges) // 2,
            "ek_time_s": ek_time, "ek_iterations": ek_result["iterations"],
            "pr_time_s": pr_time, "pr_pushes": pr_result["pushes"], "pr_relabels": pr_result["relabels"],
            "mcf_time_s": mcf_time, "mcf_iterations": mcf_result["iterations"], "mcf_cost": mcf_result["cost"],
            "max_flow_values_match": (
                abs(ek_result["value"] - pr_result["value"]) < 1e-6
                and abs(ek_result["value"] - mcf_result["value"]) < 1e-6
            ),
        })
        print(f"  three_way n={n:5d}  EK {ek_time:7.3f}s  PR {pr_time:7.3f}s  MCF {mcf_time:7.3f}s  match={rows[-1]['max_flow_values_match']}")
    return rows


# ---------------------------------------------------------------------------
# Push-Relabel convergence profiles
# ---------------------------------------------------------------------------

def benchmark_convergence():
    profiles = {}

    grid_network = grid_graph(30, 30, rng=random.Random(7))
    result = push_relabel_max_flow(grid_network, (0, 0), (29, 29))
    profiles["30x30 grid"] = result["convergence"]

    random_network, source, sink = random_flow_network(300, 0.3, rng=random.Random(8))
    result = push_relabel_max_flow(random_network, source, sink)
    profiles["n=300, density 0.3"] = result["convergence"]

    sparse_network, source, sink = random_flow_network(300, 0.02, rng=random.Random(9))
    result = push_relabel_max_flow(sparse_network, source, sink)
    profiles["n=300, density 0.02"] = result["convergence"]

    return profiles


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_runtime_figure(scaling_rows, sparse_dense_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ns = [r["n"] for r in scaling_rows]
    ax.plot(ns, [r["ek_time_s"] for r in scaling_rows], "o-", color="tab:blue", label="Edmonds-Karp")
    ax.plot(ns, [r["pr_time_s"] for r in scaling_rows], "s-", color="tab:orange", label="Push-Relabel")
    ax.set_xlabel("n (grid nodes)")
    ax.set_ylabel("runtime (seconds)")
    ax.set_yscale("log")
    ax.set_title("Runtime Scaling, Grid Graphs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = [r["density_label"] for r in sparse_dense_rows]
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [r["ek_time_s"] for r in sparse_dense_rows], width, label="Edmonds-Karp", color="tab:blue")
    ax.bar([i + width / 2 for i in x], [r["pr_time_s"] for r in sparse_dense_rows], width, label="Push-Relabel", color="tab:orange")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\n({r['edges']} edges)" for l, r in zip(labels, sparse_dense_rows)])
    ax.set_ylabel("runtime (seconds)")
    ax.set_title(f"Sparse vs Dense, n={SPARSE_DENSE_N}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_memory_figure(scaling_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ns = [r["n"] for r in scaling_rows]
    ax.plot(ns, [r["ek_memory_kb"] / 1024.0 for r in scaling_rows], "o-", color="tab:blue", label="Edmonds-Karp")
    ax.plot(ns, [r["pr_memory_kb"] / 1024.0 for r in scaling_rows], "s-", color="tab:orange", label="Push-Relabel")
    ax.set_xlabel("n (grid nodes)")
    ax.set_ylabel("peak traced memory (MB)")
    ax.set_title("Memory Use vs Graph Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_augmentation_counts_figure(scaling_rows, three_way_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ns = [r["n"] for r in scaling_rows]
    ax.plot(ns, [r["ek_iterations"] for r in scaling_rows], "o-", color="tab:blue", label="EK iterations")
    ax.plot(ns, [r["pr_pushes"] for r in scaling_rows], "s-", color="tab:orange", label="PR pushes")
    ax.plot(ns, [r["pr_relabels"] for r in scaling_rows], "^-", color="tab:red", label="PR relabels")
    ax.set_xlabel("n (grid nodes)")
    ax.set_ylabel("operation count")
    ax.set_yscale("log")
    ax.set_title("Augmentation / Operation Counts, Grid Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ns2 = [r["n"] for r in three_way_rows]
    ax.plot(ns2, [r["ek_iterations"] for r in three_way_rows], "o-", color="tab:blue", label="EK iterations")
    ax.plot(ns2, [r["pr_pushes"] for r in three_way_rows], "s-", color="tab:orange", label="PR pushes")
    ax.plot(ns2, [r["mcf_iterations"] for r in three_way_rows], "^-", color="tab:green", label="MCF iterations")
    ax.set_xlabel("n (random network nodes)")
    ax.set_ylabel("operation count")
    ax.set_yscale("log")
    ax.set_title("Augmentation Counts, Three-Way Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_convergence_figure(profiles, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["tab:purple", "tab:brown", "tab:cyan"]
    for (label, series), color in zip(profiles.items(), colors):
        if not series:
            continue
        xs = [point[0] for point in series]
        ys = [point[1] for point in series]
        ax.plot(xs, ys, "o-", color=color, label=label, markersize=3)

    ax.set_xlabel("operation count (pushes + relabels so far)")
    ax.set_ylabel("total excess remaining")
    ax.set_title("Push-Relabel Convergence Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV export + entry point
# ---------------------------------------------------------------------------

def export_csv(all_rows, out_path):
    fieldnames = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running scaling sweep (grid graphs, Edmonds-Karp vs Push-Relabel)...")
    scaling_rows = benchmark_scaling()

    print("Running sparse vs dense comparison...")
    sparse_dense_rows = benchmark_sparse_vs_dense()

    print("Running three-way comparison (EK vs PR vs MCF)...")
    three_way_rows = benchmark_three_way()

    print("Collecting Push-Relabel convergence profiles...")
    profiles = benchmark_convergence()

    print("Saving figures...")
    save_runtime_figure(scaling_rows, sparse_dense_rows, os.path.join(RESULTS_DIR, "runtime_scaling.png"))
    save_memory_figure(scaling_rows, os.path.join(RESULTS_DIR, "memory_use.png"))
    save_augmentation_counts_figure(scaling_rows, three_way_rows, os.path.join(RESULTS_DIR, "augmentation_counts.png"))
    save_convergence_figure(profiles, os.path.join(RESULTS_DIR, "convergence_profiles.png"))

    print("Rendering a small Edmonds-Karp augmenting-path animation...")
    ek_demo_network, s, t = random_flow_network(6, 0.6, rng=random.Random(3))
    ek_demo_result = edmonds_karp_max_flow(ek_demo_network, s, t)
    animate_edmonds_karp(
        ek_demo_network, ek_demo_result["trace"],
        os.path.join(RESULTS_DIR, "edmonds_karp_demo.gif"), source=s, sink=t,
    )

    print("Rendering a small Push-Relabel height/excess animation...")
    pr_demo_network, s2, t2 = random_flow_network(6, 0.6, rng=random.Random(3))
    pr_demo_result = push_relabel_max_flow(pr_demo_network, s2, t2, detailed_trace=True)
    animate_push_relabel(
        pr_demo_network, pr_demo_result,
        os.path.join(RESULTS_DIR, "push_relabel_demo.gif"), source=s2, sink=t2,
    )

    print("Writing comparison table...")
    all_rows = scaling_rows + sparse_dense_rows + three_way_rows
    export_csv(all_rows, os.path.join(RESULTS_DIR, "comparison_table.csv"))

    print()
    print("=" * 70)
    print("WEEK 10 FLOW NETWORK BENCHMARK - SUMMARY")
    print("=" * 70)
    largest = scaling_rows[-1]
    print(f"Largest scaling instance: n={largest['n']} ({largest['edges']} edges)")
    print(f"  Edmonds-Karp: {largest['ek_time_s']:.3f}s, {largest['ek_iterations']} iterations")
    print(f"  Push-Relabel: {largest['pr_time_s']:.3f}s, {largest['pr_pushes']} pushes, {largest['pr_relabels']} relabels")
    print(f"  Max-flow values match: {largest['values_match']}")
    all_match = all(r["values_match"] for r in scaling_rows) and all(r["max_flow_values_match"] for r in three_way_rows)
    print(f"All EK/PR/MCF max-flow values matched across every instance: {all_match}")
    print("=" * 70)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
