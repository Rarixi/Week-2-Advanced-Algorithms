"""
main.py
----------
Part 5 of the flow network project. Command line interface tying every
other piece of the project together, pick a solver (--ek, --pr, --mcf),
optionally pick an application that uses it (--match, --segment,
--schedule), point it at an input file, and get results printed plus a
visualization saved out.

With no application flag, --input is treated as a raw flow network
description and the chosen solver runs directly on it. Input file
formats:

  raw network (--ek/--pr/--mcf alone)
      {"edges": [[u, v, capacity, cost], ...], "source": "...", "sink": "..."}
      cost is optional per edge (defaults to 0).

  --match
      {"left": [...], "right": [...], "allowed_pairs": [[left, right], ...]}

  --schedule
      {"jobs": [...], "machines": [...], "capacities": {
          "job_demand": {...}, "machine_capacity": {...},
          "allowed": [[job, machine], ...] or {"job,machine": capacity}
      }}

  --segment
      a path to an image file (loaded as grayscale). Accepts
      --fg-value / --bg-value to seed the foreground/background models,
      and --sigma for the similarity term's falloff.

Run with:
    python3 main.py --ek --match --input graph.json
    python3 main.py --pr --input network.json
    python3 main.py --mcf --input network_with_costs.json
    python3 main.py --ek --segment --input photo.png --fg-value 40 --bg-value 210
    python3 main.py --pr --schedule --input schedule.json
"""

import argparse
import json
import os

from core.edmonds_karp import edmonds_karp_max_flow
from core.min_cost_flow import min_cost_max_flow
from core.network import Network
from core.push_relabel import push_relabel_max_flow

SOLVERS = {
    "ek": edmonds_karp_max_flow,
    "pr": push_relabel_max_flow,
    "mcf": min_cost_max_flow,
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Flow network solver CLI")

    solver_group = parser.add_mutually_exclusive_group()
    solver_group.add_argument("--ek", action="store_const", dest="solver", const="ek")
    solver_group.add_argument("--pr", action="store_const", dest="solver", const="pr")
    solver_group.add_argument("--mcf", action="store_const", dest="solver", const="mcf")

    app_group = parser.add_mutually_exclusive_group()
    app_group.add_argument("--match", action="store_const", dest="application", const="match")
    app_group.add_argument("--segment", action="store_const", dest="application", const="segment")
    app_group.add_argument("--schedule", action="store_const", dest="application", const="schedule")

    parser.add_argument("--input", required=True, help="input graph/schedule JSON file, or image path for --segment")
    parser.add_argument("--output-dir", default="results/cli_output", help="where to save the visualization")
    parser.add_argument("--fg-value", type=float, default=0.0, help="--segment only, representative foreground pixel value")
    parser.add_argument("--bg-value", type=float, default=255.0, help="--segment only, representative background pixel value")
    parser.add_argument("--sigma", type=float, default=30.0, help="--segment only, similarity term falloff")

    parser.set_defaults(solver="ek", application=None)
    return parser


def _load_raw_network(path):
    with open(path) as f:
        data = json.load(f)
    network = Network()
    for edge in data["edges"]:
        u, v, capacity = edge[0], edge[1], edge[2]
        cost = edge[3] if len(edge) > 3 else 0.0
        network.add_edge(u, v, capacity, cost)
    return network, data["source"], data["sink"]


def _run_raw_network(args):
    from visualization.plot_network import plot_network

    network, source, sink = _load_raw_network(args.input)
    solver = SOLVERS[args.solver]
    result = solver(network, source, sink)

    print(f"solver: {args.solver}")
    print(f"max-flow value: {result['value']}")
    if "cost" in result:
        print(f"min cost: {result['cost']}")
    print("flow assignment:")
    for (u, v), flow in sorted(network.flow_matrix().items(), key=lambda kv: str(kv[0])):
        print(f"  {u} -> {v}: {flow}")

    out_path = os.path.join(args.output_dir, "network.png")
    plot_network(network, source=source, sink=sink, out_path=out_path)
    print(f"visualization saved to {out_path}")


def _run_match(args):
    from applications.matching import bipartite_matching, visualize_matching

    with open(args.input) as f:
        data = json.load(f)
    left, right, allowed = data["left"], data["right"], [tuple(p) for p in data["allowed_pairs"]]

    solver = SOLVERS[args.solver]
    matching = bipartite_matching(left, right, allowed, solver=solver)

    print(f"solver: {args.solver}")
    print(f"matched {len(matching)} pairs out of {min(len(left), len(right))} possible:")
    for left_node, right_node in matching:
        print(f"  {left_node} <-> {right_node}")

    out_path = os.path.join(args.output_dir, "matching.png")
    visualize_matching(left, right, matching, allowed_pairs=allowed, out_path=out_path)
    print(f"visualization saved to {out_path}")


def _run_schedule(args):
    with open(args.input) as f:
        data = json.load(f)
    jobs, machines, capacities = data["jobs"], data["machines"], data["capacities"]
    if isinstance(capacities.get("allowed"), dict):
        capacities["allowed"] = {
            tuple(key.split(",")): value for key, value in capacities["allowed"].items()
        }
    elif isinstance(capacities.get("allowed"), list):
        capacities["allowed"] = [tuple(pair) for pair in capacities["allowed"]]

    from applications.scheduling import solve_schedule

    solver = SOLVERS[args.solver]
    result = solve_schedule(jobs, machines, capacities, solver=solver)

    print(f"solver: {args.solver}")
    print(f"total assigned: {result['total_assigned']} / {result['total_demand']}")
    print(f"fully feasible: {result['fully_feasible']}")
    print("schedule:")
    for job, machine, units in result["schedule"]:
        print(f"  {job} -> {machine}: {units}")


def _run_segment(args):
    from PIL import Image

    from applications.image_seg import default_likelihoods, default_similarity, segment_image

    image = Image.open(args.input).convert("L")
    width, height = image.size
    pixels = list(image.getdata())
    grid = [[pixels[y * width + x] for x in range(width)] for y in range(height)]

    similarity_fn = default_similarity(sigma=args.sigma)
    fg_fn, bg_fn = default_likelihoods(args.fg_value, args.bg_value)

    solver = SOLVERS[args.solver]
    mask = segment_image(grid, similarity_fn, fg_fn, bg_fn, solver=solver)

    print(f"solver: {args.solver}")
    print(f"segmented a {width}x{height} image")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "segmentation_mask.png")
    Image.fromarray(__import__("numpy").array(mask, dtype="uint8")).save(out_path)
    print(f"mask saved to {out_path}")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.application == "match":
        _run_match(args)
    elif args.application == "schedule":
        _run_schedule(args)
    elif args.application == "segment":
        _run_segment(args)
    else:
        _run_raw_network(args)


if __name__ == "__main__":
    main()
