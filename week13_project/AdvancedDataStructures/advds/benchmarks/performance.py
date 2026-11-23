"""
performance.py
-----------------
Part 5 of the AdvancedDataStructures project: the benchmark suite.

Run from the AdvancedDataStructures/ folder:

    python3 -m advds.benchmarks.performance            # full run, ~1-2 minutes
    python3 -m advds.benchmarks.performance --quick    # smaller sizes, ~15 s
    python3 -m advds.benchmarks.performance --out DIR  # write somewhere else

Writes to benchmark_results/:
  range_query_ops.png       segment vs Fenwick: build, update, query time vs n
  range_add.png             lazy segment tree vs range Fenwick vs naive loop
  persistent_overhead.png   persistent vs ephemeral update/query cost, memory vs full copies
  rank_select.png           indexed (O(1)) vs naive (O(n)) rank / select
  wavelet_compression.png   bits per symbol: original, balanced, Huffman, entropy
  cache_transpose.png       simulated misses, transpose: naive / blocked / recursive
  cache_matmul.png          simulated misses, multiplication: ijk / ikj / blocked / recursive
  cache_layouts.png         misses per search: sorted / BFS / vEB, for several block sizes
  numpy_transpose.png       real wall-clock: transpose variants with numpy leaves
  *.csv                     the numbers behind each plot
  tables.md                 comparison tables (memory, compression, queries)
  summary.txt               headline numbers quoted in the report

Timings are the best of several repeats; per-operation times average a
batch of random operations. Memory is reported two ways: the size the
structure would have in a compact C-like layout (bits / words), and the
Python heap actually allocated (tracemalloc), because Python objects
are far larger than the machine words the theory counts.
"""

import csv
import gc
import math
import os
import random
import sys
import time
import tracemalloc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..applications.dna_index import DNAIndex, naive_count, random_genome
from ..cache_oblivious import matrix_ops as MO
from ..cache_oblivious.cache_sim import CacheHierarchy, simulate
from ..cache_oblivious.layouts import compare_layouts
from ..succinct.bit_vector import BitVector
from ..succinct.rank_select import NaiveRankSelect, RankSelect
from ..succinct.wavelet_tree import WaveletTree
from ..trees.fenwick_tree import FenwickTree, RangeFenwickTree
from ..trees.persistent_tree import PersistentSegmentTree
from ..trees.segment_tree import IterativeSegmentTree, SegmentTree

QUICK = "--quick" in sys.argv
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "benchmark_results")
if "--out" in sys.argv:                       # e.g. the test suite writes to a temp folder
    OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
COLORS = ["#2a6fdb", "#e0703a", "#2f9e6e", "#8e5bd0", "#c23b5a", "#7a7a7a"]
summary = []
tables = []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log(line):
    print(line)
    summary.append(line)


def best(fn, repeats=3):
    """Best wall time of `repeats` calls."""
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def per_op_us(fn, ops, repeats=3):
    """Microseconds per call, fn applied to each element of `ops`."""
    def run():
        for o in ops:
            fn(*o)
    return best(run, repeats) / len(ops) * 1e6


def heap_bytes(build):
    """Python heap bytes allocated by build() and still alive afterwards."""
    gc.collect()
    tracemalloc.start()
    obj = build()
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del obj
    return current


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def md_table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    tables.append("\n".join(lines))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. segment tree vs Fenwick tree
# ---------------------------------------------------------------------------

def bench_range_trees():
    sizes = [2 ** e for e in range(10, 17 if QUICK else 19, 2)]
    nops = 2000
    rows = []
    res = {k: {"build": [], "update": [], "query": []} for k in
           ("Fenwick", "segment (iterative)", "segment (lazy)")}
    for n in sizes:
        rng = random.Random(n)
        a = [rng.randint(0, 1000) for _ in range(n)]
        upd = [(rng.randrange(n), rng.randint(0, 1000)) for _ in range(nops)]
        qry = []
        for _ in range(nops):
            l = rng.randrange(n)
            qry.append((l, rng.randrange(l, n)))
        ft = FenwickTree(a)
        it = IterativeSegmentTree(a)
        st = SegmentTree(a)
        res["Fenwick"]["build"].append(best(lambda: FenwickTree(a), 2))
        res["segment (iterative)"]["build"].append(best(lambda: IterativeSegmentTree(a), 2))
        res["segment (lazy)"]["build"].append(best(lambda: SegmentTree(a), 2 if n < 2 ** 17 else 1))
        res["Fenwick"]["update"].append(per_op_us(ft.add, upd))
        res["segment (iterative)"]["update"].append(per_op_us(it.update, upd))
        res["segment (lazy)"]["update"].append(per_op_us(st.update, upd))
        res["Fenwick"]["query"].append(per_op_us(ft.range_sum, qry))
        res["segment (iterative)"]["query"].append(per_op_us(it.query, qry))
        res["segment (lazy)"]["query"].append(per_op_us(st.query_sum, qry))
        rows.append([n] + [f"{res[k][m][-1]:.4g}" for k in res for m in ("build", "update", "query")])
        print(f"  range trees n={n}: " + ", ".join(f"{k} q={res[k]['query'][-1]:.2f}us" for k in res))
    write_csv("range_query_ops.csv", ["n"] + [f"{k} {m}" for k in res for m in ("build_s", "update_us", "query_us")], rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (m, lab) in zip(axes, [("build", "build time (s)"), ("update", "point update (µs/op)"),
                                    ("query", "range sum query (µs/op)")]):
        for i, k in enumerate(res):
            ax.plot(sizes, res[k][m], "o-", ms=3.5, color=COLORS[i], label=k)
        ax.set_xscale("log", base=2)
        if m == "build":
            ax.set_yscale("log")
        style(ax, lab, "n", "")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "range_query_ops.png")

    n = sizes[-1]
    f, i_, s = res["Fenwick"], res["segment (iterative)"], res["segment (lazy)"]
    log(f"range trees n={n}: query Fenwick {f['query'][-1]:.2f}us, iterative segment "
        f"{i_['query'][-1]:.2f}us, lazy segment {s['query'][-1]:.2f}us; update Fenwick "
        f"{f['update'][-1]:.2f}us, iterative {i_['update'][-1]:.2f}us, lazy {s['update'][-1]:.2f}us")
    growth = s["query"][-1] / s["query"][0]
    log(f"lazy segment query time grew {growth:.2f}x from n={sizes[0]} to n={n} "
        f"({n // sizes[0]}x more data; log2 ratio {math.log2(n) / math.log2(sizes[0]):.2f})")
    log(f"build n={n}: Fenwick {f['build'][-1]:.3f}s, iterative segment {i_['build'][-1]:.3f}s, "
        f"lazy segment {s['build'][-1]:.3f}s")

    # memory at a fixed size
    n = 2 ** 16
    a = list(range(n))
    mem_rows = [
        ["Fenwick", n + 1, heap_bytes(lambda: FenwickTree(a))],
        ["range Fenwick (2 trees)", 2 * (n + 1), heap_bytes(lambda: RangeFenwickTree(a))],
        ["segment (iterative)", 2 * n, heap_bytes(lambda: IterativeSegmentTree(a))],
        ["segment (lazy, sum/min/max + 2 tags)", 5 * 4 * n, heap_bytes(lambda: SegmentTree(a))],
    ]
    md_table(f"Memory footprint, n = {n}",
             ["structure", "machine words (compact layout)", "words per element", "Python heap (MiB)"],
             [[name, w, f"{w / n:.1f}", f"{b / 2 ** 20:.2f}"] for name, w, b in mem_rows])
    write_csv("memory_range_trees.csv", ["structure", "words", "python_bytes"], mem_rows)
    log(f"memory n={n}: Fenwick {mem_rows[0][1] / n:.1f} words/elem ({mem_rows[0][2] / 2 ** 20:.2f} MiB Python), "
        f"lazy segment {mem_rows[3][1] / n:.0f} words/elem ({mem_rows[3][2] / 2 ** 20:.2f} MiB Python)")


def bench_range_add():
    n = 2 ** 14 if QUICK else 2 ** 16
    rng = random.Random(1)
    a = [rng.randint(0, 100) for _ in range(n)]
    widths = [1, 16, 256, 4096] + ([] if QUICK else [n])
    st = SegmentTree(a)
    rf = RangeFenwickTree(a)
    arr = list(a)
    rows = []
    t_seg, t_fen, t_naive = [], [], []
    for w in widths:
        ops = []
        for _ in range(300):
            l = rng.randrange(n - w + 1)
            ops.append((l, l + w - 1, 1))

        def naive(l, r, v):
            for i in range(l, r + 1):
                arr[i] += v
        t_seg.append(per_op_us(st.range_add, ops))
        t_fen.append(per_op_us(rf.range_add, ops))
        t_naive.append(per_op_us(naive, ops, repeats=1))
        rows.append([w, t_seg[-1], t_fen[-1], t_naive[-1]])
    write_csv("range_add.csv", ["width", "lazy_segment_us", "range_fenwick_us", "naive_loop_us"], rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lab, t) in enumerate([("lazy segment tree", t_seg), ("range Fenwick (2 BITs)", t_fen),
                                  ("loop over the range", t_naive)]):
        ax.loglog(widths, t, "o-", ms=3.5, color=COLORS[i], label=lab)
    style(ax, f"Range add of width w, n = {n}", "range width w", "µs per update")
    ax.legend(frameon=False)
    save(fig, "range_add.png")
    log(f"range add n={n}, width {widths[-1]}: lazy segment {t_seg[-1]:.1f}us, range Fenwick "
        f"{t_fen[-1]:.1f}us, naive loop {t_naive[-1]:.0f}us")


# ---------------------------------------------------------------------------
# 2. persistent tree overhead
# ---------------------------------------------------------------------------

def bench_persistent():
    n = 2 ** 14 if QUICK else 2 ** 16
    updates = 2000 if QUICK else 5000
    rng = random.Random(2)
    a = [rng.randint(0, 100) for _ in range(n)]
    ops = [(rng.randrange(n), rng.randint(0, 100)) for _ in range(updates)]
    st = SegmentTree(a)
    t_eph = per_op_us(st.update, ops, repeats=1)

    pst = PersistentSegmentTree(a)
    base_nodes = pst.node_count
    v = [0]

    def pupdate(i, x):
        v[0] = pst.update(v[0], i, x)
    t0 = time.perf_counter()
    for o in ops:
        pupdate(*o)
    t_per = (time.perf_counter() - t0) / len(ops) * 1e6
    nodes_per_update = (pst.node_count - base_nodes) / updates

    qs = []
    for _ in range(2000):
        l = rng.randrange(n)
        qs.append((l, rng.randrange(l, n)))
    q_eph = per_op_us(st.query_sum, qs)
    latest = pst.num_versions - 1
    q_new = per_op_us(lambda l, r: pst.query_sum(latest, l, r), qs)
    q_old = per_op_us(lambda l, r: pst.query_sum(0, l, r), qs)

    # memory growth with number of versions
    checkpoints = [0, 500, 1000, 2000] + ([] if QUICK else [5000])
    stats = []
    for k in checkpoints:
        if k > updates:
            continue
        stored = base_nodes + round(nodes_per_update * k)
        full = (2 * n - 1) * (k + 1)
        stats.append([k + 1, stored, full, full / stored])
    write_csv("persistent_overhead.csv", ["versions", "nodes_stored", "nodes_full_copies", "saving_factor"], stats)
    d0 = time.perf_counter()
    diff = pst.diff(latest - 1, latest)
    d_us = (time.perf_counter() - d0) * 1e6

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["ephemeral\nupdate", "persistent\nupdate", "ephemeral\nquery", "persistent\nquery (newest)",
              "persistent\nquery (version 0)"]
    vals = [t_eph, t_per, q_eph, q_new, q_old]
    a1.bar(range(5), vals, color=[COLORS[0], COLORS[1], COLORS[0], COLORS[1], COLORS[1]])
    a1.set_xticks(range(5))
    a1.set_xticklabels(labels, fontsize=8)
    style(a1, f"Cost per operation, n = {n}", "", "µs")
    vs = [s[0] for s in stats]
    a2.semilogy(vs, [s[1] for s in stats], "o-", color=COLORS[1], label="persistent (shared nodes)")
    a2.semilogy(vs, [s[2] for s in stats], "o-", color=COLORS[5], label="one full copy per version")
    style(a2, "Nodes stored vs number of versions", "versions", "nodes")
    a2.legend(frameon=False)
    save(fig, "persistent_overhead.png")

    md_table(f"Persistent segment tree, n = {n}, {updates} point updates",
             ["metric", "value"],
             [["ephemeral update (µs)", f"{t_eph:.1f}"], ["persistent update (µs)", f"{t_per:.1f}"],
              ["new nodes per update", f"{nodes_per_update:.1f}"],
              ["query, ephemeral (µs)", f"{q_eph:.1f}"], ["query, newest version (µs)", f"{q_new:.1f}"],
              ["query, version 0 (µs)", f"{q_old:.1f}"],
              ["nodes stored for all versions", stats[-1][1]], ["nodes if each version were copied", stats[-1][2]],
              ["space saving factor", f"{stats[-1][3]:.0f}x"],
              ["diff of consecutive versions (µs)", f"{d_us:.0f}"]])
    log(f"persistent n={n}: update {t_per:.1f}us vs ephemeral {t_eph:.1f}us ({t_per / t_eph:.2f}x), "
        f"{nodes_per_update:.0f} new nodes/update (log2 n + 1 = {int(math.log2(n)) + 1}); "
        f"query newest {q_new:.1f}us, version 0 {q_old:.1f}us, ephemeral {q_eph:.1f}us")
    log(f"persistent {stats[-1][0]} versions: {stats[-1][1]} nodes stored vs {stats[-1][2]} for full copies "
        f"({stats[-1][3]:.0f}x smaller); consecutive-version diff {d_us:.0f}us, {len(diff)} change(s)")


# ---------------------------------------------------------------------------
# 3. rank / select
# ---------------------------------------------------------------------------

def bench_rank_select():
    sizes = [2 ** e for e in range(12, 21 if QUICK else 23, 2)]
    rows = []
    r_idx, s_idx, r_naive, s_naive, builds, overheads = [], [], [], [], [], []
    rng = np.random.default_rng(3)
    for n in sizes:
        bits = rng.random(n) < 0.5
        bv = BitVector.from_numpy(bits)
        rs = RankSelect(bv)
        nv = NaiveRankSelect(bv)
        builds.append(best(lambda: RankSelect(bv), 1))
        overheads.append(rs.overhead_ratio())
        pos = [(int(i),) for i in rng.integers(0, n + 1, 3000)]
        ks = [(int(k),) for k in rng.integers(1, rs.ones + 1, 3000)]
        r_idx.append(per_op_us(rs.rank1, pos))
        s_idx.append(per_op_us(rs.select1, ks))
        if n <= 2 ** 18:
            r_naive.append(per_op_us(nv.rank1, pos[:200], repeats=1))
            s_naive.append(per_op_us(nv.select1, ks[:200], repeats=1))
        rows.append([n, r_idx[-1], s_idx[-1], r_naive[-1] if n <= 2 ** 18 else "",
                     s_naive[-1] if n <= 2 ** 18 else "", builds[-1], overheads[-1]])
        print(f"  rank/select n=2^{int(math.log2(n))}: rank {r_idx[-1]:.2f}us select {s_idx[-1]:.2f}us")
    write_csv("rank_select.csv", ["n_bits", "rank_us", "select_us", "naive_rank_us", "naive_select_us",
                                  "build_s", "overhead_ratio"], rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.loglog(sizes, r_idx, "o-", color=COLORS[0], label="rank1, indexed")
    a1.loglog(sizes, s_idx, "o-", color=COLORS[2], label="select1, indexed")
    a1.loglog(sizes[:len(r_naive)], r_naive, "o--", color=COLORS[1], label="rank1, popcount scan")
    a1.loglog(sizes[:len(s_naive)], s_naive, "o--", color=COLORS[3], label="select1, word scan")
    style(a1, "Query time vs bit-vector length (50% ones)", "n (bits)", "µs per query")
    a1.legend(frameon=False, fontsize=8)
    a2.semilogx(sizes, [100 * o for o in overheads], "o-", color=COLORS[1])
    a2.set_ylim(0, 60)
    style(a2, "Index size as % of the bit vector", "n (bits)", "% overhead")
    save(fig, "rank_select.png")
    log(f"rank/select n=2^{int(math.log2(sizes[-1]))}: rank {r_idx[-1]:.2f}us (vs {r_idx[0]:.2f}us at 2^"
        f"{int(math.log2(sizes[0]))}), select {s_idx[-1]:.2f}us; naive at 2^{int(math.log2(sizes[len(r_naive) - 1]))}: "
        f"rank {r_naive[-1]:.0f}us, select {s_naive[-1]:.0f}us; index overhead {100 * overheads[-1]:.1f}%")


# ---------------------------------------------------------------------------
# 4. wavelet tree compression and queries
# ---------------------------------------------------------------------------

def _corpus_text():
    """English-like text that ships with the project: the library's own docstrings and comments."""
    pkg = os.path.join(ROOT, "advds")
    parts = []
    for dirpath, _, files in sorted(os.walk(pkg)):
        for f in sorted(files):
            if f.endswith(".py"):
                with open(os.path.join(dirpath, f)) as fh:
                    parts.append(fh.read())
    return "".join(parts)


def bench_wavelet():
    n_dna = 200_000 if QUICK else 1_000_000
    dna, _ = random_genome(n_dna, seed=4)
    text = _corpus_text()
    rng = np.random.default_rng(5)
    zipf = np.minimum(rng.zipf(1.5, 200_000) - 1, 255)
    datasets = [("DNA (A,C,G,T)", dna), ("library source text", text), ("Zipf bytes (skewed)", zipf)]
    rows = []
    for name, data in datasets:
        for shape in ("balanced", "huffman"):
            if isinstance(data, str):
                t0 = time.perf_counter()
                wt = WaveletTree.from_text(data, shape)
            else:
                t0 = time.perf_counter()
                wt = WaveletTree(data, sigma=256, shape=shape)
            build = time.perf_counter() - t0
            st = wt.compression_stats()
            n = st["n"]
            idx = rng.integers(0, n, 2000).tolist()
            sym = wt.access(0)
            t_access = per_op_us(wt.access, [(i,) for i in idx[:1000]], repeats=1)
            t_rank = per_op_us(lambda i: wt.rank(sym, i), [(i,) for i in idx[:1000]], repeats=1)
            if shape == "balanced":
                qs = []
                for _ in range(1000):
                    l = int(rng.integers(0, n))
                    r = int(rng.integers(l, min(n, l + 10_000)))
                    qs.append((l, r, (r - l) // 2))
                t_q = per_op_us(wt.quantile, qs, repeats=1)
            else:
                t_q = float("nan")
            rows.append([name, shape, n, st["sigma"], round(st["h0_bits_per_symbol"], 3),
                         round(st["payload_bits_per_symbol"], 3), round(st["total_bits_per_symbol"], 3),
                         round(st["compression_ratio_payload"], 2), round(st["compression_ratio_total"], 2),
                         round(build, 2), round(t_access, 2), round(t_rank, 2),
                         "" if math.isnan(t_q) else round(t_q, 2)])
            print(f"  wavelet {name} {shape}: {st['payload_bits_per_symbol']:.2f} bits/sym payload, "
                  f"{st['total_bits_per_symbol']:.2f} with index")
    header = ["data", "shape", "n", "sigma", "H0 bits/sym", "payload bits/sym", "total bits/sym",
              "ratio vs 8-bit (payload)", "ratio vs 8-bit (total)", "build s", "access µs", "rank µs",
              "median (quantile) µs"]
    write_csv("wavelet.csv", header, rows)
    md_table("Wavelet tree compression and query cost", header, rows)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(datasets))
    w = 0.2
    get = lambda name, shape, col: next(r[col] for r in rows if r[0] == name and r[1] == shape)
    names = [d[0] for d in datasets]
    ax.bar(x - 1.5 * w, [8] * len(names), w, color=COLORS[5], label="original (8 bits)")
    ax.bar(x - 0.5 * w, [get(nm, "balanced", 5) for nm in names], w, color=COLORS[0], label="balanced payload")
    ax.bar(x + 0.5 * w, [get(nm, "huffman", 5) for nm in names], w, color=COLORS[1], label="Huffman payload")
    ax.bar(x + 1.5 * w, [get(nm, "balanced", 4) for nm in names], w, color=COLORS[2], label="entropy H0")
    for i, nm in enumerate(names):
        ax.text(x[i] + 0.5 * w, get(nm, "huffman", 6) + 0.2, f"{get(nm, 'huffman', 6):.1f}\nwith index",
                ha="center", fontsize=7, color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    style(ax, "Wavelet tree size, bits per symbol", "", "bits per symbol")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "wavelet_compression.png")

    d_bal = next(r for r in rows if r[0].startswith("DNA") and r[1] == "balanced")
    t_bal = next(r for r in rows if r[0].startswith("library") and r[1] == "balanced")
    t_huf = next(r for r in rows if r[0].startswith("library") and r[1] == "huffman")
    z_huf = next(r for r in rows if r[0].startswith("Zipf") and r[1] == "huffman")
    z_bal = next(r for r in rows if r[0].startswith("Zipf") and r[1] == "balanced")
    log(f"wavelet DNA n={d_bal[2]}: {d_bal[5]} bits/base payload ({d_bal[7]}x smaller than 8-bit), "
        f"{d_bal[6]} with rank/select index ({d_bal[8]}x); access {d_bal[10]}us, rank {d_bal[11]}us, median {d_bal[12]}us")
    log(f"wavelet text n={t_bal[2]} sigma={t_bal[3]}: H0 {t_bal[4]}, balanced {t_bal[5]} bits/sym, "
        f"Huffman {t_huf[5]} bits/sym payload, {t_huf[6]} with index")
    log(f"wavelet Zipf n={z_bal[2]}: H0 {z_bal[4]}, balanced {z_bal[5]}, Huffman {z_huf[5]} bits/sym "
        f"(payload ratio {z_huf[7]}x vs 8-bit)")

    # DNA index end to end
    idx = DNAIndex(dna)
    pats = ["GATTACA", "TTAGGG", "ACGTACGT", "CG"]
    t_fm = per_op_us(idx.pattern_count, [(p,) for p in pats], repeats=3)
    t0 = time.perf_counter()
    for p in pats:
        naive_count(dna, p)
    t_scan = (time.perf_counter() - t0) / len(pats) * 1e6
    rep = idx.size_report()
    log(f"DNA FM-index n={n_dna}: pattern count {t_fm:.0f}us vs str.find scan {t_scan:.0f}us; "
        f"FM-index {rep['fm_index_bits_per_base']:.2f} bits/base vs {64 * (n_dna + 1) / n_dna:.0f} for a plain suffix array")


# ---------------------------------------------------------------------------
# 5. cache-oblivious algorithms
# ---------------------------------------------------------------------------

L1 = 32 * 1024
LINE = 64


def bench_cache_matrix():
    # transpose
    t_sizes = [64, 128, 256, 512] + ([] if QUICK else [1024])
    rows = []
    res = {m: [] for m in ("naive", "blocked", "recursive")}
    for n in t_sizes:
        for m in res:
            res[m].append(simulate(MO.trace_transpose(n, m, block=8), L1, LINE))
        rows.append([n] + [res[m][-1] for m in res] + [2 * n * n * 8 // LINE])
    write_csv("cache_transpose.csv", ["n", "naive", "blocked_8", "recursive", "compulsory"], rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(res):
        ax.loglog(t_sizes, res[m], "o-", ms=3.5, color=COLORS[i],
                  label={"naive": "naive (row order)", "blocked": "blocked 8x8 (cache-aware)",
                         "recursive": "recursive (cache-oblivious)"}[m])
    ax.loglog(t_sizes, [r[-1] for r in rows], "--", color=COLORS[5], lw=0.8, label="compulsory (2n²/8 lines)")
    style(ax, "Transpose: simulated L1 misses (32 KiB, 64 B lines)", "n", "misses")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "cache_transpose.png")
    big = t_sizes[-1]
    log(f"transpose n={big} (32 KiB L1): naive {res['naive'][-1]} misses, blocked {res['blocked'][-1]}, "
        f"recursive {res['recursive'][-1]} (compulsory {rows[-1][-1]}; naive {res['naive'][-1] / res['recursive'][-1]:.1f}x more)")

    # multiplication, two cache levels
    m_sizes = [16, 32, 48, 64] + ([] if QUICK else [96])
    methods = ("ijk", "ikj", "blocked", "recursive")
    mm = {m: [] for m in methods}
    mm_l2 = {m: [] for m in methods}
    rows = []
    for n in m_sizes:
        for m in methods:
            h = CacheHierarchy((("L1", 8 * 1024, LINE), ("L2", 64 * 1024, LINE)))
            rep = h.run(MO.trace_matmul(n, m, block=8))
            mm[m].append(rep["L1"])
            mm_l2[m].append(rep["L2"])
        rows.append([n] + [mm[m][-1] for m in methods] + [mm_l2[m][-1] for m in methods])
        print(f"  matmul n={n}: " + ", ".join(f"{m}={mm[m][-1]}" for m in methods))
    write_csv("cache_matmul.csv", ["n"] + [f"L1 {m}" for m in methods] + [f"L2 {m}" for m in methods], rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(methods):
        a1.loglog(m_sizes, mm[m], "o-", ms=3.5, color=COLORS[i], label=m)
        a2.loglog(m_sizes, mm_l2[m], "o-", ms=3.5, color=COLORS[i], label=m)
    style(a1, "Matrix multiply: L1 misses (8 KiB)", "n", "misses")
    style(a2, "Same run: L2 misses (64 KiB)", "n", "misses")
    a1.legend(frameon=False, fontsize=8)
    save(fig, "cache_matmul.png")
    n = m_sizes[-1]
    log(f"matmul n={n}: L1 (8 KiB) misses ijk {mm['ijk'][-1]}, ikj {mm['ikj'][-1]}, blocked-8 {mm['blocked'][-1]}, "
        f"recursive {mm['recursive'][-1]}; L2 (64 KiB) ijk {mm_l2['ijk'][-1]}, recursive {mm_l2['recursive'][-1]}, "
        f"blocked-8 {mm_l2['blocked'][-1]}")
    md_table(f"Simulated cache misses, matrix multiply n = {n} (L1 8 KiB, L2 64 KiB, 64 B lines)",
             ["method", "L1 misses", "L2 misses"],
             [[m, mm[m][-1], mm_l2[m][-1]] for m in methods])

    # traversal (n = 1024: one column spans 1024 lines, more than the 512 an L1 of 32 KiB holds;
    # at n = 512 a column fits exactly and column order would look as good as row order)
    n = 1024
    trav = {o: simulate(MO.trace_traversal(n, o), L1, LINE) for o in ("row", "column", "recursive")}
    log(f"traversal n={n}: misses row {trav['row']}, column {trav['column']}, recursive {trav['recursive']}")

    # pure-Python timings (interpreter-bound)
    n = 64 if QUICK else 96
    rng = np.random.default_rng(6)
    A, B = rng.standard_normal((n, n)).tolist(), rng.standard_normal((n, n)).tolist()
    py = {"ijk": best(lambda: MO.matmul_naive(A, B, "ijk"), 2),
          "ikj": best(lambda: MO.matmul_naive(A, B, "ikj"), 2),
          "recursive (leaf 1)": best(lambda: MO.matmul_recursive(A, B, 1), 1),
          "recursive (leaf 16)": best(lambda: MO.matmul_recursive(A, B, 16), 2)}
    md_table(f"Pure-Python matrix multiply, n = {n} (wall clock, seconds)", ["method", "seconds"],
             [[k, f"{v:.3f}"] for k, v in py.items()])
    log(f"pure Python matmul n={n}: " + ", ".join(f"{k} {v:.3f}s" for k, v in py.items()))


def bench_numpy_transpose():
    sizes = [512, 1024, 2048] + ([] if QUICK else [4096])
    funcs = {"naive (row -> column)": MO.transpose_naive_np,
             "blocked 64 (cache-aware)": MO.transpose_blocked_np,
             "recursive (cache-oblivious)": MO.transpose_recursive_np,
             "numpy A.T.copy()": lambda A: A.T.copy()}
    res = {k: [] for k in funcs}
    rows = []
    for n in sizes:
        A = np.random.default_rng(n).standard_normal((n, n))
        for k, f in funcs.items():
            res[k].append(best(lambda: f(A), 3))
        rows.append([n] + [res[k][-1] for k in funcs])
    write_csv("numpy_transpose.csv", ["n"] + list(funcs), rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(funcs):
        ax.loglog(sizes, res[k], "o-", ms=3.5, color=COLORS[i], label=k)
    style(ax, "Transpose wall-clock time (numpy leaves)", "n", "seconds")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "numpy_transpose.png")
    n = sizes[-1]
    log(f"numpy transpose n={n}: naive {res['naive (row -> column)'][-1]:.3f}s, blocked "
        f"{res['blocked 64 (cache-aware)'][-1]:.3f}s, recursive {res['recursive (cache-oblivious)'][-1]:.3f}s, "
        f"A.T.copy() {res['numpy A.T.copy()'][-1]:.3f}s")


def bench_layouts():
    height = 18 if QUICK else 20
    blocks = (64, 256, 1024, 4096)
    rows = compare_layouts(height, num_queries=1000 if QUICK else 3000, block_sizes=blocks)
    write_csv("cache_layouts.csv", list(rows[0].keys()), [list(r.values()) for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    layouts = list(dict.fromkeys(r["layout"] for r in rows))
    for i, lay in enumerate(layouts):
        a1.semilogx(blocks, [r["cold_misses"] for r in rows if r["layout"] == lay], "o-", base=2,
                    color=COLORS[i], label=lay)
        a2.semilogx(blocks, [r["warm_misses"] for r in rows if r["layout"] == lay], "o-", base=2,
                    color=COLORS[i], label=lay)
    n = rows[0]["n"]
    a1.semilogx(blocks, [math.log2(n + 1) / math.log2(b / 8) for b in blocks], "--", base=2,
                color=COLORS[5], lw=0.8, label="log_B N (B in nodes)")
    style(a1, f"Blocks touched per search, cold cache (N = {n:,})", "block size (bytes)", "misses / search")
    style(a2, "Misses per search, warm 32 KiB cache", "block size (bytes)", "misses / search")
    a1.legend(frameon=False, fontsize=8)
    save(fig, "cache_layouts.png")
    md_table(f"Static search tree, N = {n:,} keys, misses per search",
             ["layout", "block (bytes)", "cold cache", "warm 32 KiB cache"],
             [[r["layout"], r["block_bytes"], f"{r['cold_misses']:.2f}", f"{r['warm_misses']:.2f}"] for r in rows])
    g = {(r["layout"], r["block_bytes"]): r for r in rows}
    log(f"vEB layout N={n}: cold misses/search at 64 B: sorted {g[('sorted (binary search)', 64)]['cold_misses']:.2f}, "
        f"BFS {g[('BFS / heap', 64)]['cold_misses']:.2f}, vEB {g[('van Emde Boas', 64)]['cold_misses']:.2f}; "
        f"at 4096 B: sorted {g[('sorted (binary search)', 4096)]['cold_misses']:.2f}, BFS "
        f"{g[('BFS / heap', 4096)]['cold_misses']:.2f}, vEB {g[('van Emde Boas', 4096)]['cold_misses']:.2f}")
    log(f"vEB warm 32 KiB cache, 64 B lines: sorted {g[('sorted (binary search)', 64)]['warm_misses']:.2f}, "
        f"BFS {g[('BFS / heap', 64)]['warm_misses']:.2f}, vEB {g[('van Emde Boas', 64)]['warm_misses']:.2f}")


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    steps = [("segment vs Fenwick", bench_range_trees), ("range add", bench_range_add),
             ("persistent overhead", bench_persistent), ("rank/select", bench_rank_select),
             ("wavelet tree", bench_wavelet), ("cache-oblivious matrices", bench_cache_matrix),
             ("numpy transpose", bench_numpy_transpose), ("vEB layout", bench_layouts)]
    for name, fn in steps:
        print(f"{name}...")
        fn()
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(f"# benchmark summary ({'quick' if QUICK else 'full'} run, python {sys.version.split()[0]}, "
                 f"numpy {np.__version__})\n")
        fh.write("\n".join(summary) + "\n")
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("# Benchmark tables\n\nGenerated by `python3 -m advds.benchmarks.performance`"
                 f"{' --quick' if QUICK else ''}.\n\n" + "\n\n".join(tables) + "\n")
    print(f"done in {time.perf_counter() - t0:.0f}s, results in {OUT}")


if __name__ == "__main__":
    main()
