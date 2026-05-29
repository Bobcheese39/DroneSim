from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dronesim.models import RunResult, RunSummary
from dronesim.services.terrain import MapSpec, TerrainService
from dronesim.storage import RunStore


def _sample_run(*, run_id: str = "run_test001", scenario_id: str = "scenario_abc") -> RunResult:
    time_s = [0.0, 0.1, 0.2]
    return RunResult(
        run_id=run_id,
        scenario_id=scenario_id,
        backend_id="inhouse_mpc_quad",
        model_id="inhouse_mpc_quad",
        status="success",
        time_s=time_s,
        position_m=[[0.0, 0.0, 5.0], [0.5, 0.1, 5.0], [1.0, 0.2, 5.0]],
        velocity_mps=[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        acceleration_mps2=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        attitude_rad=[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        angular_rate_rad_s=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        controls=[[10.0, 0.1, 0.0, 0.0], [10.1, 0.0, 0.0, 0.0], [10.0, -0.1, 0.0, 0.0]],
        reference_position_m=[[0.0, 0.0, 5.0], [0.5, 0.0, 5.0], [1.0, 0.0, 5.0]],
        tracking_error_m=[0.0, 0.1, 0.2],
        summary=RunSummary(success=True, miss_distance_m=0.05, duration_s=0.2),
        metadata={"cfg_summary": {"dt": 0.1, "seed": 42}},
        created_utc="2026-01-01T00:00:00+00:00",
    )


class RunStoreTest(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        run = _sample_run()
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(runs_root=Path(tmp) / "runs")
            saved_dir = store.save(run)
            loaded = store.load(saved_dir)
            self.assertEqual(loaded.run_id, run.run_id)
            self.assertEqual(loaded.position_m, run.position_m)
            self.assertEqual(loaded.reference_position_m, run.reference_position_m)
            self.assertEqual(loaded.controls, run.controls)
            self.assertEqual(loaded.summary.miss_distance_m, 0.05)

    def test_trajectory_csv_columns(self) -> None:
        run = _sample_run()
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(runs_root=Path(tmp) / "runs")
            saved_dir = store.save(run)
            csv_path = saved_dir / "trajectory.csv"
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            self.assertEqual(len(rows), 3)
            for col in (
                "time_s", "x_m", "y_m", "z_m",
                "ref_x_m", "ref_y_m", "ref_z_m",
                "ft_N", "tx_Nm", "ty_Nm", "tz_Nm",
                "tracking_error_m",
            ):
                self.assertIn(col, fieldnames)
            self.assertAlmostEqual(float(rows[1]["ref_y_m"]), 0.0)
            self.assertAlmostEqual(float(rows[1]["ft_N"]), 10.1)

    def test_list_runs_and_summaries_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(runs_root=Path(tmp) / "runs")
            store.save(_sample_run(run_id="run_a", scenario_id="scenario_x"))
            store.save(_sample_run(run_id="run_b", scenario_id="scenario_x"))
            store.save(_sample_run(run_id="run_c", scenario_id="scenario_y"))

            all_paths = store.list_runs()
            self.assertEqual(len(all_paths), 3)

            filtered = store.list_runs("scenario_x")
            self.assertEqual(len(filtered), 2)

            summaries = store.list_run_summaries("scenario_x")
            self.assertEqual(len(summaries), 2)
            run_ids = {str(s["run_id"]) for s in summaries}
            self.assertEqual(run_ids, {"run_a", "run_b"})
            self.assertIn("path", summaries[0])
            self.assertFalse(summaries[0].get("time_s"))

    def test_load_scenario_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(runs_root=Path(tmp) / "runs")
            store.save(_sample_run(run_id="run_a", scenario_id="scenario_x"))
            store.save(_sample_run(run_id="run_b", scenario_id="scenario_x"))
            runs = store.load_scenario_runs("scenario_x")
            self.assertEqual(len(runs), 2)
            self.assertEqual({r.run_id for r in runs}, {"run_a", "run_b"})


class ClearanceTest(unittest.TestCase):
    def test_compute_run_clearance_m(self) -> None:
        from dronesim.services.terrain import compute_run_clearance_m

        run = _sample_run()
        service = TerrainService()
        asset = service.build_blank_asset(MapSpec(resolution=32))
        clearance = compute_run_clearance_m(run, asset)
        self.assertEqual(len(clearance), len(run.time_s))
        self.assertTrue(all(c >= 0 for c in clearance))


if __name__ == "__main__":
    unittest.main()
