"""
test_main_cli.py
-------------------
Tests for main.py. Writes small input files to a temp directory and
drives the CLI through main.main(argv), checking it runs end to end
(exit cleanly, produce the expected visualization file) for each solver
and application combination, rather than re-testing the algorithms or
applications themselves.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as cli


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


class TestRawNetworkMode:
    def test_ek_solves_and_saves_a_plot(self, tmp_path, capsys):
        input_path = tmp_path / "network.json"
        _write_json(input_path, {
            "edges": [["s", "a", 3, 0], ["a", "t", 10, 0], ["s", "b", 10, 0], ["b", "t", 2, 0]],
            "source": "s", "sink": "t",
        })
        out_dir = tmp_path / "out"

        cli.main(["--ek", "--input", str(input_path), "--output-dir", str(out_dir)])

        captured = capsys.readouterr()
        assert "max-flow value: 5.0" in captured.out
        assert (out_dir / "network.png").exists()

    def test_mcf_reports_a_cost(self, tmp_path, capsys):
        input_path = tmp_path / "network.json"
        _write_json(input_path, {
            "edges": [["s", "a", 3, 2], ["a", "t", 3, 1]],
            "source": "s", "sink": "t",
        })
        out_dir = tmp_path / "out"

        cli.main(["--mcf", "--input", str(input_path), "--output-dir", str(out_dir)])

        captured = capsys.readouterr()
        assert "min cost: 9.0" in captured.out


class TestMatchMode:
    def test_runs_and_reports_a_matching(self, tmp_path, capsys):
        input_path = tmp_path / "match.json"
        _write_json(input_path, {
            "left": ["L0", "L1"], "right": ["R0", "R1"],
            "allowed_pairs": [["L0", "R0"], ["L1", "R1"]],
        })
        out_dir = tmp_path / "out"

        cli.main(["--pr", "--match", "--input", str(input_path), "--output-dir", str(out_dir)])

        captured = capsys.readouterr()
        assert "matched 2 pairs" in captured.out
        assert (out_dir / "matching.png").exists()


class TestScheduleMode:
    def test_runs_and_reports_a_schedule(self, tmp_path, capsys):
        input_path = tmp_path / "schedule.json"
        _write_json(input_path, {
            "jobs": ["J0"], "machines": ["M0"],
            "capacities": {
                "job_demand": {"J0": 1}, "machine_capacity": {"M0": 1},
                "allowed": [["J0", "M0"]],
            },
        })
        out_dir = tmp_path / "out"

        cli.main(["--ek", "--schedule", "--input", str(input_path), "--output-dir", str(out_dir)])

        captured = capsys.readouterr()
        assert "fully feasible: True" in captured.out
