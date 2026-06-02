"""Tests for Monte Carlo analysis serialization."""
from __future__ import annotations

import unittest

from dronesim.models import RunResult, RunSummary
from dronesim.web import analysis as analysis_mod


def _synthetic_run(
    trial_index: int,
    *,
    miss: float,
    success: bool = True,
    n_steps: int = 5,
    seed: int | None = None,
) -> RunResult:
    if seed is None:
        seed = 10 + trial_index
    t = [float(i) * 0.1 for i in range(n_steps)]
    err = [miss * (i + 1) / n_steps for i in range(n_steps)]
    vel = [[1.0 + trial_index * 0.1, 0.0, 0.0] for _ in t]
    att = [[0.01 * trial_index, 0.0, 0.0] for _ in t]
    return RunResult(
        run_id=f"run_{trial_index}",
        scenario_id="scenario_test",
        backend_id="inhouse_mpc_quad",
        model_id="quad",
        status="completed",
        time_s=t,
        velocity_mps=vel,
        attitude_rad=att,
        tracking_error_m=err,
        summary=RunSummary(
            success=success,
            miss_distance_m=miss,
            duration_s=t[-1] if t else 0.0,
            settle_steps=n_steps,
            wallclock_s=0.5 + trial_index * 0.1,
        ),
        metadata={
            "cfg_summary": {
                "seed": seed,
                "monte_carlo": {"trial_index": trial_index},
            }
        },
    )


class McAnalysisBlockTest(unittest.TestCase):
    def test_empty_runs(self) -> None:
        block = analysis_mod.mc_analysis_block([])
        self.assertEqual(block["mode"], "monte_carlo")
        self.assertEqual(block["trials"], [])
        self.assertEqual(block["histogram"]["counts"], [])

    def test_batch_summary_and_histogram(self) -> None:
        runs = [
            _synthetic_run(0, miss=1.0),
            _synthetic_run(1, miss=2.0),
            _synthetic_run(2, miss=3.0, success=False),
        ]
        block = analysis_mod.mc_analysis_block(runs)

        self.assertEqual(len(block["trials"]), 3)
        self.assertEqual(block["trials"][0]["trial_index"], 0)
        self.assertEqual(block["trials"][2]["success"], False)

        summary = {row["metric"]: row["value"] for row in block["summary"]}
        self.assertEqual(summary["n_trials"], 3)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["miss_mean_m"], 2.0)
        self.assertEqual(len(block["histogram"]["counts"]), len(block["histogram"]["bins"]) - 1)

    def test_envelope_series(self) -> None:
        runs = [_synthetic_run(i, miss=float(i)) for i in range(4)]
        block = analysis_mod.mc_analysis_block(runs)
        env = block["envelopes"]["tracking_error"]
        self.assertEqual(len(env["time_s"]), len(env["mean"]))
        self.assertEqual(len(env["trials"]), 4)
        vx = block["envelopes"]["velocity"]["vx"]
        self.assertEqual(len(vx["trials"]), 4)
        self.assertEqual(len(vx["mean"]), len(vx["time_s"]))


class McReplayBlockTest(unittest.TestCase):
    def test_empty_runs(self) -> None:
        block = analysis_mod.mc_replay_block([])
        self.assertEqual(block["mode"], "monte_carlo_replay")
        self.assertEqual(block["trials"], [])
        self.assertEqual(block["reference_position_m"], [])

    def test_payload_shape_and_ordering(self) -> None:
        runs = [
            _synthetic_run(2, miss=3.0),
            _synthetic_run(0, miss=1.0),
            _synthetic_run(1, miss=2.0),
        ]
        for run in runs:
            run.position_m = [[float(i), 0.0, 1.0] for i in range(len(run.time_s))]
            run.reference_position_m = [[float(i), 0.0, 0.0] for i in range(len(run.time_s))]

        block = analysis_mod.mc_replay_block(runs, center={"lat": 40.0, "lon": -105.0})

        self.assertEqual(block["mode"], "monte_carlo_replay")
        self.assertEqual(block["center"]["lat"], 40.0)
        self.assertEqual(len(block["reference_position_m"]), 5)
        self.assertEqual(len(block["trials"]), 3)
        self.assertEqual([t["trial_index"] for t in block["trials"]], [0, 1, 2])
        self.assertEqual(block["trials"][0]["run_id"], "run_0")
        self.assertEqual(len(block["trials"][1]["position_m"]), 5)
        self.assertTrue(block["trials"][2]["success"])


if __name__ == "__main__":
    unittest.main()
