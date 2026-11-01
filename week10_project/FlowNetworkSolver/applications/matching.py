"""
matching.py
--------------
Part 2 of the flow network project. Bipartite matching reduced to
max-flow, a source connected to every left partition node (capacity 1
each), a capacity-1 edge for every allowed left-right pairing, and every
right partition node connected to a sink (capacity 1 each). Every unit
of max-flow through this network corresponds to one matched pair, since
capacity-1 edges on both sides of the pairing edges mean no left or
right node can be used twice, so max-flow here is exactly maximum
bipartite matching, not just an upper bound on it.
"""

import os

from core.edmonds_karp import edmonds_karp_max_flow
from core.network import Network

_SOURCE = "__source__"
_SINK = "__sink__"


def build_matching_network(left_nodes, right_nodes, allowed_pairs):
    """Builds the source/left/right/sink reduction described above.
    Returns the Network, source label, and sink label."""
    network = Network()
    network.add_node(_SOURCE)
    network.add_node(_SINK)

    for left in left_nodes:
        network.add_edge(_SOURCE, left, 1.0)
    for right in right_nodes:
        network.add_edge(right, _SINK, 1.0)
    for left, right in allowed_pairs:
        network.add_edge(left, right, 1.0)

    return network, _SOURCE, _SINK


def bipartite_matching(left_nodes, right_nodes, allowed_pairs, solver=edmonds_karp_max_flow):
    """Builds the flow network reduction, solves it with `solver`
    (edmonds_karp_max_flow by default, push_relabel_max_flow also works,
    both report a "value" and leave the winning flow on the network),
    and returns the matching as a list of (left, right) pairs."""
    network, source, sink = build_matching_network(left_nodes, right_nodes, allowed_pairs)
    solver(network, source, sink)

    matching = []
    for (u, v), flow in network.flow_matrix().items():
        if u in left_nodes and v in right_nodes and flow > 1e-9:
            matching.append((u, v))
    return matching


def visualize_matching(left_nodes, right_nodes, matching, allowed_pairs=None, out_path=None):
    """
    Draws the bipartite graph, left nodes in a column on the left, right
    nodes in a column on the right, every allowed pairing drawn as a
    light gray edge and every matched pairing drawn on top in a bold
    highlighted color. Saves to out_path if given, otherwise returns the
    matplotlib Figure for the caller to show or save itself.
    """
    import matplotlib
    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, max(4, 0.6 * max(len(left_nodes), len(right_nodes)))))

    left_y = {node: -i for i, node in enumerate(left_nodes)}
    right_y = {node: -i for i, node in enumerate(right_nodes)}
    matched_set = set(matching)

    if allowed_pairs:
        for left, right in allowed_pairs:
            if (left, right) in matched_set:
                continue
            ax.plot([0, 1], [left_y[left], right_y[right]], color="lightgray", linewidth=1, zorder=1)

    for left, right in matching:
        ax.plot([0, 1], [left_y[left], right_y[right]], color="tab:green", linewidth=2.5, zorder=2)

    for node, y in left_y.items():
        ax.scatter([0], [y], s=300, color="tab:blue", zorder=3)
        ax.text(-0.05, y, str(node), ha="right", va="center")
    for node, y in right_y.items():
        ax.scatter([1], [y], s=300, color="tab:orange", zorder=3)
        ax.text(1.05, y, str(node), ha="left", va="center")

    ax.set_title(f"Bipartite Matching ({len(matching)} pairs)")
    ax.axis("off")
    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return None
    return fig
