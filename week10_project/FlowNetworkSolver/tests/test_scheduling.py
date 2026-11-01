"""
test_scheduling.py
---------------------
Tests for applications/scheduling.py. Checks a fully feasible instance
assigns every unit of demand, an over demanded instance correctly
reports infeasibility while still returning the best partial schedule,
and that the returned schedule never exceeds any job's demand or any
machine's capacity.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from applications.scheduling import solve_schedule


class TestFeasibleInstance:
    def test_every_job_gets_fully_placed_when_capacity_allows(self):
        jobs = ["J0", "J1", "J2"]
        machines = ["M0", "M1"]
        capacities = {
            "job_demand": {"J0": 1.0, "J1": 1.0, "J2": 1.0},
            "machine_capacity": {"M0": 2.0, "M1": 1.0},
            "allowed": [("J0", "M0"), ("J1", "M0"), ("J2", "M1")],
        }

        result = solve_schedule(jobs, machines, capacities)

        assert result["fully_feasible"]
        assert result["total_assigned"] == 3.0


class TestInfeasibleInstance:
    def test_over_demand_reports_infeasible_but_still_maximizes_placement(self):
        jobs = ["J0", "J1", "J2"]
        machines = ["M0"]
        capacities = {
            "job_demand": {"J0": 1.0, "J1": 1.0, "J2": 1.0},
            "machine_capacity": {"M0": 2.0},
            "allowed": [("J0", "M0"), ("J1", "M0"), ("J2", "M0")],
        }

        result = solve_schedule(jobs, machines, capacities)

        assert not result["fully_feasible"]
        assert result["total_assigned"] == 2.0  # machine capacity bottleneck
        assert result["total_demand"] == 3.0


class TestScheduleRespectsConstraints:
    def test_no_job_or_machine_exceeds_its_limit(self):
        jobs = ["J0", "J1", "J2", "J3"]
        machines = ["M0", "M1"]
        capacities = {
            "job_demand": {"J0": 2.0, "J1": 1.0, "J2": 1.0, "J3": 2.0},
            "machine_capacity": {"M0": 3.0, "M1": 3.0},
            "allowed": {
                ("J0", "M0"): 2.0, ("J0", "M1"): 2.0,
                ("J1", "M0"): 1.0, ("J1", "M1"): 1.0,
                ("J2", "M0"): 1.0,
                ("J3", "M1"): 2.0,
            },
        }

        result = solve_schedule(jobs, machines, capacities)

        per_job = {}
        per_machine = {}
        for job, machine, units in result["schedule"]:
            per_job[job] = per_job.get(job, 0.0) + units
            per_machine[machine] = per_machine.get(machine, 0.0) + units

        for job, demand in capacities["job_demand"].items():
            assert per_job.get(job, 0.0) <= demand + 1e-9
        for machine, capacity in capacities["machine_capacity"].items():
            assert per_machine.get(machine, 0.0) <= capacity + 1e-9
