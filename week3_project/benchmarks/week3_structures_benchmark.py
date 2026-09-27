"""
week3_structures_benchmark.py
------------------------------
Extends the Week 1-2 benchmarking framework to cover Week 3's data
structures: MinHeap, AVLTree, ChainingHashTable, OpenAddressingHashTable,
a plain Python list (as a "naive" baseline), and Python's built-in dict
(as a "gold standard" reference).

For each structure and each input size n, we build a fresh, pre-populated
instance (build cost is NOT timed) and then time a single insert, search,
and delete operation performed while the structure already holds n items.
This measures how the COST OF ONE OPERATION scales with n -- which is
exactly what Big-O notation describes -- rather than the cost of building
the whole structure from scratch.

Outputs (all saved under benchmarks/results/):
    heap_performance.png        Heap insert / extract-min timing vs n
    tree_performance.png        AVL Tree vs Python dict, insert/search timing vs n
    hash_performance.png        Separate chaining vs open addressing, insert/search/delete timing vs n
    complexity_comparison.png   The big picture: Linear (List) vs Logarithmic (AVL, Heap) vs Constant (Hash Table)
    comparison_table.csv        Every raw timing this script collected, one row per (structure, operation, n)

Run with:
    python3 benchmarks/week3_structures_benchmark.py
"""

import csv
import os
import random
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_structures.heap import MinHeap
from src.data_structures.avl_tree import AVLTree
from src.data_structures.hash_table import ChainingHashTable, OpenAddressingHashTable


SIZES = [100, 300, 1000, 3000, 10000, 30000]
TRIALS = 7
RANDOM_SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

results_rows = []


def record(structure: str, operation: str, n: int, avg_seconds: float) -> None:
    results_rows.append(
        {"structure": structure, "operation": operation, "n": n, "avg_seconds": avg_seconds}
    )


def average_time(measure_fn, teardown_fn=None, trials: int = TRIALS) -> float:
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        measure_fn()
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        if teardown_fn is not None:
            teardown_fn()
    return statistics.mean(samples)


def benchmark_list(n: int) -> None:
    random.seed(RANDOM_SEED)
    data = list(range(n))
    random.shuffle(data)
    lst = list(data)

    new_value = n
    target = data[-1]

    insert_time = average_time(
        measure_fn=lambda: lst.append(new_value),
        teardown_fn=lambda: lst.pop(),
    )
    search_time = average_time(
        measure_fn=lambda: (target in lst),
    )
    delete_time = average_time(
        measure_fn=lambda: lst.remove(target),
        teardown_fn=lambda: lst.append(target),
    )

    record("List", "insert", n, insert_time)
    record("List", "search", n, search_time)
    record("List", "delete", n, delete_time)


def benchmark_heap(n: int) -> None:
    random.seed(RANDOM_SEED)
    data = list(range(n))
    random.shuffle(data)
    heap = MinHeap(data)

    new_value = n
    existing_value = data[-1]

    insert_time = average_time(
        measure_fn=lambda: heap.insert(new_value),
    )
    search_time = average_time(
        measure_fn=lambda: (existing_value in heap._data),
    )
    delete_time = average_time(
        measure_fn=lambda: heap.extract_min(),
        teardown_fn=lambda: heap.insert(existing_value),
    )

    record("Heap", "insert", n, insert_time)
    record("Heap", "search", n, search_time)
    record("Heap", "delete (extract_min)", n, delete_time)


def benchmark_avl_tree(n: int) -> None:
    random.seed(RANDOM_SEED)
    data = list(range(n))
    random.shuffle(data)
    tree = AVLTree()
    for v in data:
        tree.insert(v)

    new_value = n
    existing_value = data[-1]

    insert_time = average_time(
        measure_fn=lambda: tree.insert(new_value),
        teardown_fn=lambda: tree.delete(new_value),
    )
    search_time = average_time(
        measure_fn=lambda: tree.search(existing_value),
    )
    delete_time = average_time(
        measure_fn=lambda: tree.delete(existing_value),
        teardown_fn=lambda: tree.insert(existing_value),
    )

    record("AVL Tree", "insert", n, insert_time)
    record("AVL Tree", "search", n, search_time)
    record("AVL Tree", "delete", n, delete_time)


def benchmark_dict(n: int) -> None:
    random.seed(RANDOM_SEED)
    data = list(range(n))
    random.shuffle(data)
    d = {v: v for v in data}

    new_key = n
    existing_key = data[-1]

    insert_time = average_time(
        measure_fn=lambda: d.__setitem__(new_key, new_key),
        teardown_fn=lambda: d.__delitem__(new_key),
    )
    search_time = average_time(
        measure_fn=lambda: d.get(existing_key),
    )

    record("dict (built-in)", "insert", n, insert_time)
    record("dict (built-in)", "search", n, search_time)


def benchmark_hash_tables(n: int) -> None:
    random.seed(RANDOM_SEED)
    data = list(range(n))
    random.shuffle(data)

    for label, table_cls in [
        ("Hash Table (Chaining)", ChainingHashTable),
        ("Hash Table (Open Addressing)", OpenAddressingHashTable),
    ]:
        table = table_cls()
        for v in data:
            table.insert(v, v)

        new_key = n
        existing_key = data[-1]

        insert_time = average_time(
            measure_fn=lambda t=table, k=new_key: t.insert(k, k),
            teardown_fn=lambda t=table, k=new_key: t.delete(k),
        )
        search_time = average_time(
            measure_fn=lambda t=table, k=existing_key: t.get(k),
        )
        delete_time = average_time(
            measure_fn=lambda t=table, k=existing_key: t.delete(k),
            teardown_fn=lambda t=table, k=existing_key: t.insert(k, k),
        )

        record(label, "insert", n, insert_time)
        record(label, "search", n, search_time)
        record(label, "delete", n, delete_time)


def run_all_benchmarks() -> None:
    for n in SIZES:
        print(f"Benchmarking n={n} ...")
        benchmark_list(n)
        benchmark_heap(n)
        benchmark_avl_tree(n)
        benchmark_dict(n)
        benchmark_hash_tables(n)


def save_csv() -> None:
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    rows = sorted(results_rows, key=lambda r: (r["structure"], r["operation"], r["n"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure", "operation", "n", "avg_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _series(structure: str, operation: str):
    matching = [r for r in results_rows if r["structure"] == structure and r["operation"] == operation]
    matching.sort(key=lambda r: r["n"])
    xs = [r["n"] for r in matching]
    ys = [r["avg_seconds"] * 1e6 for r in matching]
    return xs, ys


def plot_heap_performance() -> None:
    plt.figure(figsize=(8, 5))
    for op, style in [("insert", "o-"), ("delete (extract_min)", "s-")]:
        xs, ys = _series("Heap", op)
        plt.plot(xs, ys, style, label=op)
    plt.xlabel("n (number of elements in the heap)")
    plt.ylabel("Time per operation (microseconds)")
    plt.title("Heap: Insert & Extract-Min Timing vs Size -- O(log n)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, "heap_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_tree_performance() -> None:
    plt.figure(figsize=(8, 5))
    for structure, op, style in [
        ("AVL Tree", "insert", "o-"),
        ("AVL Tree", "search", "s-"),
        ("dict (built-in)", "insert", "o--"),
        ("dict (built-in)", "search", "s--"),
    ]:
        xs, ys = _series(structure, op)
        plt.plot(xs, ys, style, label=f"{structure} {op}")
    plt.xlabel("n (number of elements)")
    plt.ylabel("Time per operation (microseconds)")
    plt.title("AVL Tree (O(log n)) vs Python dict (O(1)): Insert/Search Timing vs Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, "tree_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_hash_performance() -> None:
    plt.figure(figsize=(8, 5))
    for structure, style in [
        ("Hash Table (Chaining)", "-"),
        ("Hash Table (Open Addressing)", "--"),
    ]:
        for op, marker in [("insert", "o"), ("search", "s"), ("delete", "^")]:
            xs, ys = _series(structure, op)
            plt.plot(xs, ys, style, marker=marker, label=f"{structure} {op}")
    plt.xlabel("n (number of elements)")
    plt.ylabel("Time per operation (microseconds)")
    plt.title("Hash Table: Separate Chaining vs Open Addressing -- O(1) average")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, "hash_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_complexity_comparison() -> None:
    plt.figure(figsize=(9, 6))
    series_to_plot = [
        ("List", "search", "List search -- O(n) [Linear]", "o-"),
        ("AVL Tree", "search", "AVL Tree search -- O(log n) [Logarithmic]", "s-"),
        ("Heap", "insert", "Heap insert -- O(log n) [Logarithmic]", "^-"),
        ("Hash Table (Chaining)", "search", "Hash Table search -- O(1) [Constant]", "d-"),
    ]
    for structure, op, label, style in series_to_plot:
        xs, ys = _series(structure, op)
        plt.plot(xs, ys, style, label=label)
    plt.xlabel("n (number of elements)")
    plt.ylabel("Time per operation (microseconds)")
    plt.title("Growth Rate Comparison: Linear vs Logarithmic vs Constant")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, "complexity_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    run_all_benchmarks()
    save_csv()
    plot_heap_performance()
    plot_tree_performance()
    plot_hash_performance()
    plot_complexity_comparison()
    print("Done. See benchmarks/results/ for all outputs.")
