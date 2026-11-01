"""
plot_network.py
------------------
Part 3 of the flow network project. Static graph visualization with
matplotlib and networkx, drawing every node and edge with its capacity
(and current flow, if the network has any) labeled directly on the edge,
flow written as "flow/capacity" so a saturated edge is visible at a
glance.
"""

import os

import matplotlib
import networkx as nx


def _build_digraph(network, min_cut=None):
    graph = nx.DiGraph()
    for node in network.nodes:
        graph.add_node(node)

    for i in range(0, len(network.edges), 2):
        edge = network.edges[i]
        if edge.capacity <= 0:
            continue
        flow = network.flow[i]
        graph.add_edge(edge.u, edge.v, capacity=edge.capacity, flow=flow, cost=edge.cost)

    return graph


def plot_network(network, source=None, sink=None, show_flow=True, show_cost=False, out_path=None):
    """
    Draws `network` (every forward edge with positive capacity, backward
    bookkeeping edges are never drawn), labeling each edge with its
    capacity, its current flow over capacity if show_flow is True and
    the network has any, and its cost if show_cost is True. source and
    sink, if given, are drawn in a distinct color. Saves to out_path if
    given, otherwise returns the Figure.
    """
    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = _build_digraph(network)
    fig, ax = plt.subplots(figsize=(9, 7))

    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except Exception:
        pos = nx.spring_layout(graph, seed=42)

    node_colors = []
    for node in graph.nodes():
        if node == source:
            node_colors.append("tab:green")
        elif node == sink:
            node_colors.append("tab:red")
        else:
            node_colors.append("tab:blue")

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_size=8)
    nx.draw_networkx_edges(graph, pos, ax=ax, arrowsize=15, connectionstyle="arc3,rad=0.08")

    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        parts = []
        if show_flow:
            parts.append(f"{data['flow']:.1f}/{data['capacity']:.1f}")
        else:
            parts.append(f"{data['capacity']:.1f}")
        if show_cost:
            parts.append(f"cost {data['cost']:.1f}")
        edge_labels[(u, v)] = ", ".join(parts)

    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels, font_size=7)

    ax.set_title("Flow Network")
    ax.axis("off")
    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return None
    return fig
