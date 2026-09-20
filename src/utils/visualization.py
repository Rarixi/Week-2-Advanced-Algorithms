"""
visualization.py
-----------------
Plotting helpers for Part 4 (benchmarks/week4_performance.py): drawing a graph's BFS/DFS
traversal order with networkx + matplotlib, and rendering the performance comparison charts
(time/memory vs. graph size, Dijkstra vs. density) as PNGs.

Pulled out into its own module for the same reason src/utils/graph_generator.py is separate
from the benchmark script that calls it: week4_performance.py's job is to RUN the
benchmarks and decide what to plot; this module's job is to know HOW to draw each kind of
plot. Neither needs to know much about the other.
"""

import matplotlib

matplotlib.use("Agg")  # write PNGs directly, no need for a display
import matplotlib.pyplot as plt
import networkx as nx


def build_networkx_graph(graph):
    """Convert one of our own Graph objects (src/graphs/graph.py) into an equivalent
    networkx graph, so networkx's layout and drawing functions can be used on it.
    DiGraph if `graph` is directed, plain Graph otherwise -- this matters for whether
    edges get drawn with arrowheads."""
    nx_graph = nx.DiGraph() if graph.directed else nx.Graph()
    for node in graph.nodes:
        nx_graph.add_node(node)
    for node in graph.nodes:
        for neighbor, weight in graph.get_neighbors(node).items():
            nx_graph.add_edge(node, neighbor, weight=weight)
    return nx_graph


def plot_traversal_order(graph, order, algorithm_label, output_path, layout_seed=42):
    """Draw `graph`, coloring and labeling every node by the step at which `order` (the
    return value of bfs_full()/dfs_full()/bfs()/dfs_iterative(), etc.) first visited it.
    Saves the figure to `output_path` and closes it.

    `layout_seed` is fixed (not randomized per call) so that calling this twice on the
    SAME graph with two different orders -- e.g. once for BFS, once for DFS -- produces
    the same node layout both times. That's what makes the two resulting images a fair
    side-by-side comparison: any difference you see is genuinely about visit order, not
    about the nodes having moved around on the page.
    """
    nx_graph = build_networkx_graph(graph)
    layout = nx.spring_layout(nx_graph, seed=layout_seed)

    order_index = {node: i for i, node in enumerate(order)}
    node_colors = [order_index[node] for node in nx_graph.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(nx_graph, layout, alpha=0.35, arrows=graph.directed, arrowsize=12)
    nodes_drawn = nx.draw_networkx_nodes(
        nx_graph, layout, node_color=node_colors, cmap=plt.cm.viridis, node_size=600
    )
    nx.draw_networkx_labels(
        nx_graph, layout,
        labels={node: f"{node}\n#{order_index[node]}" for node in nx_graph.nodes()},
        font_size=8, font_color="white",
    )
    plt.colorbar(nodes_drawn, label="Visit order (0 = visited first)")
    plt.title(f"{algorithm_label.upper()} traversal order, source = node 0")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bfs_dfs_metric_vs_n(rows, graph_types, metric_key, ylabel, title, output_path):
    """Line plot of one metric (e.g. "time_us" or "peak_memory_kb") vs. n, with one line
    per (graph_type, algorithm) combination -- BFS solid circles, DFS dashed squares.

    `rows` is the same list of dicts week4_performance.py builds for comparison_table.csv
    (each needs at least graph_type, n, algorithm, and metric_key). `graph_types` picks
    which graph types to actually draw, since plotting all four at once tends to be too
    busy to read -- the full picture across every graph type still lives in the CSV.
    """
    def rows_for(graph_type, algorithm):
        matching = [r for r in rows if r["graph_type"] == graph_type and r["algorithm"] == algorithm]
        matching.sort(key=lambda r: r["n"])
        return [r["n"] for r in matching], [r[metric_key] for r in matching]

    plt.figure(figsize=(8, 6))
    for graph_type in graph_types:
        for algorithm, style in [("bfs", "-o"), ("dfs", "--s")]:
            ns, values = rows_for(graph_type, algorithm)
            plt.plot(ns, values, style, label=f"{graph_type} / {algorithm}")
    plt.xlabel("n (number of nodes)")
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_dijkstra_density_comparison(rows, graph_types, sizes, output_path):
    """Grouped bar chart: one group of bars per graph type, one bar per size within each
    group, height = Dijkstra's mean time (microseconds) on that (graph_type, n) pair.
    `rows` is the same comparison_table.csv row list, filtered here to algorithm=="dijkstra".
    """
    plt.figure(figsize=(9, 6))
    bar_width = 0.25
    x_positions = range(len(graph_types))

    for i, n in enumerate(sizes):
        heights = []
        for graph_type in graph_types:
            match = next(
                r for r in rows
                if r["graph_type"] == graph_type and r["n"] == n and r["algorithm"] == "dijkstra"
            )
            heights.append(match["time_us"])
        offsets = [x + (i - 1) * bar_width for x in x_positions]
        plt.bar(offsets, heights, width=bar_width, label=f"n={n}")

    plt.xticks(list(x_positions), graph_types)
    plt.ylabel("mean time (microseconds)")
    plt.yscale("log")
    plt.title("Dijkstra (heap-based) runtime by graph density")
    plt.legend(title="graph size")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

