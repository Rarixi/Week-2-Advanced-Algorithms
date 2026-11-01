"""
animate_flow.py
------------------
Part 3 of the flow network project. Animates an algorithm's run step by
step, saved as a GIF (via matplotlib's Pillow writer, so no external
ffmpeg dependency is needed). Edmonds-Karp's animation highlights each
round's augmenting path in turn against the fixed background network.
Push-Relabel's animation instead colors every node by its current height
label and labels it with its current excess, replaying the detailed
trace a push_relabel_max_flow(..., detailed_trace=True) call produces,
so you can watch height labels climb and excess drain toward the sink
over the run.
"""

import os

import matplotlib
import networkx as nx


def _layout_for(network):
    graph = nx.DiGraph()
    for node in network.nodes:
        graph.add_node(node)
    for i in range(0, len(network.edges), 2):
        edge = network.edges[i]
        if edge.capacity > 0:
            graph.add_edge(edge.u, edge.v)
    try:
        return graph, nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except Exception:
        return graph, nx.spring_layout(graph, seed=42)


def animate_edmonds_karp(network, trace, out_path, source=None, sink=None, fps=1):
    """
    Builds a GIF from an edmonds_karp_max_flow() trace, one frame per
    augmenting path, the background network drawn in light gray and that
    round's path highlighted in bold. `network` should be the same
    Network the trace was produced on (used only for its structure and
    layout, not its current flow values).
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    graph, pos = _layout_for(network)
    fig, ax = plt.subplots(figsize=(8, 6))

    node_colors = [
        "tab:green" if n == source else "tab:red" if n == sink else "tab:blue"
        for n in graph.nodes()
    ]

    def draw_frame(i):
        ax.clear()
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=450)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_size=7)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="lightgray", arrowsize=12)

        running_total = sum(entry["bottleneck"] for entry in trace[:i + 1])
        if i < len(trace):
            entry = trace[i]
            nx.draw_networkx_edges(
                graph, pos, ax=ax, edgelist=entry["path"],
                edge_color="tab:orange", width=3, arrowsize=18,
            )
            ax.set_title(
                f"Edmonds-Karp, iteration {i + 1}/{len(trace)}, "
                f"pushing {entry['bottleneck']:.1f}, total flow {running_total:.1f}"
            )
        ax.axis("off")

    frame_count = max(1, len(trace))
    anim = FuncAnimation(fig, draw_frame, frames=frame_count, interval=1000 // max(1, fps))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate_push_relabel(network, result, out_path, source=None, sink=None, fps=2, max_frames=120):
    """
    Builds a GIF from a push_relabel_max_flow(..., detailed_trace=True)
    result, one frame per sampled operation, every node colored by its
    current height (darker = taller) and labeled with its current
    excess, the node being pushed or relabeled that frame outlined in
    orange. `network` should be the same Network the result was produced
    on. Detailed traces can get long on anything but a small graph, so
    frames are subsampled down to at most max_frames evenly spaced
    operations if the trace is longer than that.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    trace = result["trace"]
    if not trace:
        raise ValueError(
            "push_relabel result has no detailed trace, re-run "
            "push_relabel_max_flow(..., detailed_trace=True) first"
        )

    if len(trace) > max_frames:
        step = len(trace) / max_frames
        trace = [trace[int(i * step)] for i in range(max_frames)]

    graph, pos = _layout_for(network)
    fig, ax = plt.subplots(figsize=(8, 6))

    max_height = max(max(snap["heights"].values()) for snap in trace) or 1

    def draw_frame(i):
        ax.clear()
        snapshot = trace[i]
        heights = snapshot["heights"]
        excess = snapshot["excess"]

        colors = [plt.cm.viridis(heights.get(n, 0) / max_height) for n in graph.nodes()]
        edge_colors = ["tab:orange" if n == snapshot["node"] else "none" for n in graph.nodes()]

        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_color=colors, node_size=500,
            edgecolors=edge_colors, linewidths=3,
        )
        labels = {n: f"{n}\nh={heights.get(n, 0)} e={excess.get(n, 0):.1f}" for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=6, font_color="white")
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="lightgray", arrowsize=10)

        ax.set_title(f"Push-Relabel, step {i + 1}/{len(trace)}, {snapshot['op']} at {snapshot['node']}")
        ax.axis("off")

    anim = FuncAnimation(fig, draw_frame, frames=len(trace), interval=1000 // max(1, fps))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
