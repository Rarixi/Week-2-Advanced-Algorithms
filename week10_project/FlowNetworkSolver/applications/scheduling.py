"""
scheduling.py
----------------
Part 2 of the flow network project. Scheduling and resource allocation
modeled as a flow network, the same three layer reduction matching.py
uses, generalized to non unit capacities. A source connects to every job
with capacity equal to that job's demand (how many units of work it
needs placed), a capacity-limited edge connects a job to every machine
it's allowed to run on, and every machine connects to a sink with
capacity equal to how much work that machine can take on. Max-flow
through this network is the largest amount of total work that can
actually be placed given every job's demand and every machine's
capacity, and the flow assignment on the job-machine edges is the
schedule itself, how many units of each job landed on each machine.
This same construction covers project staffing (jobs become roles,
machines become staff members) and tournament bracket style feasibility
(jobs become matches needing a slot, machines become time slots) just by
relabeling what the nodes represent.

`capacities` bundles the three pieces of per-instance configuration:
  job_demand          {job: units_needed}, defaults to 1 unit per job
                       for any job not listed
  machine_capacity     {machine: units_available}, defaults to 1 unit per
                       machine for any machine not listed
  allowed              either a list of (job, machine) pairs (each
                       allowed at up to that job's full demand) or a
                       dict {(job, machine): capacity} for a tighter
                       per-pair limit, defaults to every job allowed on
                       every machine if not given
"""

from core.edmonds_karp import edmonds_karp_max_flow
from core.network import Network

_SOURCE = "__source__"
_SINK = "__sink__"


def _normalize_capacities(jobs, machines, capacities):
    job_demand = capacities.get("job_demand", {})
    machine_capacity = capacities.get("machine_capacity", {})
    allowed = capacities.get("allowed")

    job_demand = {job: job_demand.get(job, 1.0) for job in jobs}
    machine_capacity = {machine: machine_capacity.get(machine, 1.0) for machine in machines}

    if allowed is None:
        allowed = {(job, machine): job_demand[job] for job in jobs for machine in machines}
    elif isinstance(allowed, dict):
        allowed = dict(allowed)
    else:
        allowed = {(job, machine): job_demand[job] for job, machine in allowed}

    return job_demand, machine_capacity, allowed


def build_scheduling_network(jobs, machines, capacities):
    """Builds the flow network reduction described above. Returns the
    Network, source label, and sink label."""
    job_demand, machine_capacity, allowed = _normalize_capacities(jobs, machines, capacities)

    network = Network()
    for job in jobs:
        network.add_edge(_SOURCE, job, job_demand[job])
    for machine in machines:
        network.add_edge(machine, _SINK, machine_capacity[machine])
    for (job, machine), cap in allowed.items():
        network.add_edge(job, machine, cap)

    return network, _SOURCE, _SINK


def solve_schedule(jobs, machines, capacities, solver=edmonds_karp_max_flow):
    """
    Solves the scheduling network's max-flow with `solver`
    (edmonds_karp_max_flow by default, push_relabel_max_flow also works)
    and returns a dict with:
      schedule          list of (job, machine, units) for every
                        job-machine pair that actually carries flow
      total_assigned    total units of work successfully placed
      total_demand      total units of work requested across every job
      fully_feasible    True if every job's full demand got placed
    """
    job_demand, _machine_capacity, _allowed = _normalize_capacities(jobs, machines, capacities)
    network, source, sink = build_scheduling_network(jobs, machines, capacities)
    result = solver(network, source, sink)

    job_set = set(jobs)
    machine_set = set(machines)
    schedule = [
        (u, v, flow)
        for (u, v), flow in network.flow_matrix().items()
        if u in job_set and v in machine_set and flow > 1e-9
    ]

    total_demand = sum(job_demand.values())
    return {
        "schedule": schedule,
        "total_assigned": result["value"],
        "total_demand": total_demand,
        "fully_feasible": abs(result["value"] - total_demand) < 1e-6,
    }
