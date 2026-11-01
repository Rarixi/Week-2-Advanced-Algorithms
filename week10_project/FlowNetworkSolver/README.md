# FlowNetworkSolver

## Description
A full max-flow and min-cost-flow toolkit: three core algorithms, three
applications built on top of them, visualization and animation, a command
line interface, a benchmark suite comparing algorithm performance at scale,
and a written report on the results. Fully implemented, tested, and
benchmarked.

- **Edmonds-Karp** -- the polynomial time max-flow algorithm, BFS augmenting
  paths over a residual graph. O(V * E) iterations.
- **Push-Relabel** -- a higher performance max-flow algorithm using height
  labels and excess flow, with the gap heuristic implemented so it actually
  scales (see "A note on Push-Relabel" below). Reaches 10k-50k node networks.
- **Min-Cost Max-Flow** -- successive shortest augmenting paths with node
  potentials (Johnson's trick), for flow problems where each unit routed
  also has a cost.
- **Bipartite Matching** -- a max-flow reduction, source to left partition,
  left to right capacity-1 edges, right partition to sink.
- **Image Segmentation** -- pixels as nodes, similarity and
  foreground/background likelihood as edges, segmented via a min s-t cut
  (Boykov-Jolly construction).
- **Scheduling and Resource Allocation** -- jobs, machines, and staffing
  modeled as flow networks and solved via max-flow.

## Structure
```
FlowNetworkSolver/
├── core/              # network.py, edmonds_karp.py, push_relabel.py, min_cost_flow.py
├── applications/      # matching.py, image_seg.py, scheduling.py
├── visualization/     # plot_network.py, animate_flow.py
├── utils/             # generators.py, validators.py
├── tests/             # 50 tests, cross-validated against brute force / DFS reference solvers
├── results/           # benchmark figures, CSV, and demo GIFs
├── analysis/          # ch9_report.md, the written technical report
├── main.py            # CLI entry point
└── week10_flow_benchmark.py
```

## Setup
```bash
pip install matplotlib networkx numpy pillow
```

## Running Tests
From inside `FlowNetworkSolver/`:
```bash
pytest tests/
```
All 50 tests pass, covering the core algorithms (cross-validated against
each other and an independent DFS reference solver), the applications
(matching against a brute force matcher, segmentation on a clean
separability case, scheduling against hand-built feasible/infeasible
instances), and the CLI end-to-end.

## Running the CLI
```bash
# raw network, pick a solver
python3 main.py --ek --input path/to/network.json
python3 main.py --pr --input path/to/network.json
python3 main.py --mcf --input path/to/network_with_costs.json

# applications
python3 main.py --ek --match --input graph.json
python3 main.py --pr --schedule --input schedule.json
python3 main.py --ek --segment --input photo.png --fg-value 40 --bg-value 210
```
See the docstring at the top of `main.py` for the exact input JSON formats.

## Running the Benchmark
```bash
python3 week10_flow_benchmark.py
```
Writes `runtime_scaling.png`, `memory_use.png`, `augmentation_counts.png`,
`convergence_profiles.png`, `comparison_table.csv`, and two demo GIFs
(`edmonds_karp_demo.gif`, `push_relabel_demo.gif`) to `results/`. The full
run sweeps grid graphs up to 14,400 nodes / 57,120 edges, a sparse vs. dense
150-node comparison, and a three-way EK/PR/MCF comparison up to 1,500 nodes.
Every max-flow value and min-cost value matched across every solver on
every instance in the run.

## A note on Push-Relabel
My first working version used plain FIFO discharge with only a current-arc
optimization, and it was unusably slow on anything resembling a real graph,
a 45x45 grid (2,025 nodes) took over a minute and about 7 million push and
relabel operations. The fix is the gap heuristic: track how many nodes sit
at each height, and whenever a relabel empties out a height band entirely,
bulk-relabel every node above that band in one pass, since they're provably
disconnected from the sink for the rest of the run. After adding it, the
same grid dropped to about 1.2 seconds, and the algorithm scaled to 14,400
nodes in under 25 seconds in the full benchmark. Full writeup in
`analysis/ch9_report.md` and in the module docstring of
`core/push_relabel.py`.

## Report
`analysis/ch9_report.md` covers the theory (max-flow, min-cut, residual
graphs, optimality conditions), a comparison of the three algorithms, the
three applications, the real benchmark results, and the engineering lessons
from the Push-Relabel fix above.
