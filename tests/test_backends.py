from __future__ import annotations

import threading
import time
import unittest

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.sim import (
    BackendUnavailable,
    DroneFactory,
    InHouseMpcQuadBackend,
    PlaceholderBackend,
    RunManager,
    SimulationBackend,
)
from dronesim.sim.run_manager import wait_for
from dronesim.storage import RunStore


def _make_result(scenario: ScenarioSpec, run_config: RunConfig, *, status: str = "success") -> RunResult:
    return RunResult(
        run_id=run_config.run_id,
        scenario_id=scenario.scenario_id,
        backend_id=run_config.backend_id,
        model_id=scenario.vehicle.model_id,
        status=status,
        summary=RunSummary(success=status == "success", duration_s=0.0, wallclock_s=0.0),
    )


class _StubBackend(SimulationBackend):
    backend_id = "stub_backend"
    display_name = "Stub Backend"

    def __init__(self, sleep_s: float = 0.0) -> None:
        self.calls: list[tuple[str, int | None, int]] = []
        self.sleep_s = sleep_s
        self._lock = threading.Lock()

    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress=None,
    ) -> RunResult:
        cfg = run_config or scenario.run_config
        trial_index = int(cfg.monte_carlo.get("trial_index", 0))
        with self._lock:
            self.calls.append((cfg.run_id, cfg.seed, trial_index))
        if on_step_progress is not None:
            for step in (1, 2, 3):
                on_step_progress(step, 3)
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        return _make_result(scenario, cfg)


def _make_scenario() -> ScenarioSpec:
    scenario = ScenarioSpec()
    scenario.validate()
    return scenario


class DroneFactoryTest(unittest.TestCase):
    def test_register_get_available(self) -> None:
        factory = DroneFactory(backends=[])
        stub = _StubBackend()
        factory.register(stub)
        self.assertIs(factory.get("stub_backend"), stub)
        ids = {entry["backend_id"] for entry in factory.available()}
        self.assertIn("stub_backend", ids)

    def test_get_unknown_raises(self) -> None:
        factory = DroneFactory(backends=[])
        with self.assertRaises(KeyError):
            factory.get("does_not_exist")

    def test_placeholder_backend_raises_backend_unavailable(self) -> None:
        backend = PlaceholderBackend("plan_x", "Planned Backend X", "Install plan_x first.")
        with self.assertRaises(BackendUnavailable):
            backend.run(_make_scenario())


class RunManagerSingleTest(unittest.TestCase):
    def test_single_run_streams_result_and_done(self) -> None:
        stub = _StubBackend()
        factory = DroneFactory(backends=[stub])
        manager = RunManager(factory=factory)
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="stub_backend")

        order: list[str] = []
        results: list[RunResult] = []
        progress: list[tuple[int, int]] = []
        errors: list[BaseException] = []
        done_event = threading.Event()

        def on_result(idx: int, run: RunResult) -> None:
            order.append("result")
            results.append(run)

        def on_progress(done: int, total: int) -> None:
            order.append("progress")
            progress.append((done, total))

        def on_error(idx: int, exc: BaseException) -> None:
            errors.append(exc)

        def on_done() -> None:
            order.append("done")
            done_event.set()

        manager.start_single(
            scenario,
            cfg,
            on_result=on_result,
            on_progress=on_progress,
            on_error=on_error,
            on_done=on_done,
        )
        self.assertTrue(done_event.wait(timeout=5.0))
        self.assertTrue(wait_for(manager, timeout=5.0))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].scenario_id, scenario.scenario_id)
        self.assertEqual(progress, [(1, 1)])
        self.assertEqual(errors, [])
        self.assertEqual(order, ["result", "progress", "done"])

    def test_single_run_streams_step_progress(self) -> None:
        stub = _StubBackend()
        factory = DroneFactory(backends=[stub])
        manager = RunManager(factory=factory)
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="stub_backend")

        order: list[str] = []
        step_progress: list[tuple[int, int]] = []
        done_event = threading.Event()

        def on_step_progress(step: int, total: int) -> None:
            order.append("step")
            step_progress.append((step, total))

        manager.start_single(
            scenario,
            cfg,
            on_step_progress=on_step_progress,
            on_result=lambda idx, run: order.append("result"),
            on_progress=lambda done, total: order.append("progress"),
            on_done=lambda: (order.append("done"), done_event.set()),
        )
        self.assertTrue(done_event.wait(timeout=5.0))
        self.assertEqual(step_progress, [(1, 3), (2, 3), (3, 3)])
        self.assertLess(order.index("result"), order.index("done"))
        self.assertLess(order.index("step"), order.index("result"))

    def test_single_run_surfaces_backend_error(self) -> None:
        class _BoomBackend(SimulationBackend):
            backend_id = "boom"
            display_name = "Boom"

            def run(self, scenario, run_config=None, *, on_step_progress=None):  # type: ignore[override]
                del on_step_progress
                raise RuntimeError("explode")

        factory = DroneFactory(backends=[_BoomBackend()])
        manager = RunManager(factory=factory)
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="boom")

        errors: list[BaseException] = []
        done_event = threading.Event()

        manager.start_single(
            scenario,
            cfg,
            on_error=lambda idx, exc: errors.append(exc),
            on_done=lambda: done_event.set(),
        )
        self.assertTrue(done_event.wait(timeout=5.0))
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)


class RunManagerMonteCarloTest(unittest.TestCase):
    def test_monte_carlo_assigns_seeds_and_trial_index(self) -> None:
        stub = _StubBackend()
        factory = DroneFactory(backends=[stub])
        manager = RunManager(factory=factory)
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="stub_backend")

        results: list[RunResult] = []
        progress: list[tuple[int, int]] = []
        done_event = threading.Event()

        manager.start_monte_carlo(
            scenario,
            cfg,
            n_trials=4,
            workers=2,
            base_seed=10,
            executor="thread",
            on_result=lambda idx, run: results.append(run),
            on_progress=lambda done, total: progress.append((done, total)),
            on_done=lambda: done_event.set(),
        )
        self.assertTrue(done_event.wait(timeout=10.0))
        self.assertTrue(wait_for(manager, timeout=10.0))

        self.assertEqual(len(results), 4)
        run_ids = {r.run_id for r in results}
        self.assertEqual(len(run_ids), 4, "Each MC trial should have a unique run_id")

        seeds = sorted(call[1] for call in stub.calls)
        trial_indices = sorted(call[2] for call in stub.calls)
        self.assertEqual(seeds, [10, 11, 12, 13])
        self.assertEqual(trial_indices, [0, 1, 2, 3])
        self.assertEqual(progress[-1], (4, 4))

    def test_cannot_start_while_running(self) -> None:
        stub = _StubBackend(sleep_s=0.2)
        manager = RunManager(factory=DroneFactory(backends=[stub]))
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="stub_backend")

        manager.start_monte_carlo(
            scenario,
            cfg,
            n_trials=2,
            workers=1,
            executor="thread",
        )
        try:
            with self.assertRaises(RuntimeError):
                manager.start_single(scenario, cfg)
        finally:
            self.assertTrue(wait_for(manager, timeout=5.0))


class RunManagerCancelTest(unittest.TestCase):
    def test_cancel_stops_streaming_results(self) -> None:
        stub = _StubBackend(sleep_s=0.1)
        manager = RunManager(factory=DroneFactory(backends=[stub]))
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="stub_backend")

        results: list[RunResult] = []
        done_event = threading.Event()

        manager.start_monte_carlo(
            scenario,
            cfg,
            n_trials=20,
            workers=2,
            executor="thread",
            on_result=lambda idx, run: results.append(run),
            on_done=lambda: done_event.set(),
        )
        time.sleep(0.15)
        manager.cancel()
        self.assertTrue(done_event.wait(timeout=5.0))
        self.assertTrue(wait_for(manager, timeout=5.0))
        self.assertLess(len(results), 20, "Cancel should prevent some trials from streaming")


class InHouseBackendSmokeTest(unittest.TestCase):
    """Headless end-to-end check: init, step loop, normalized result, save."""

    def test_inhouse_backend_runs_and_saves(self) -> None:
        scenario = ScenarioSpec(name="smoke-test")
        scenario.run_config = RunConfig(
            backend_id="inhouse_mpc_quad",
            max_steps=30,
            dt_s=0.1,
            target_altitude_m=5.0,
        )
        scenario.validate()

        backend = InHouseMpcQuadBackend()
        step_updates: list[tuple[int, int]] = []

        result = backend.run(
            scenario,
            scenario.run_config,
            on_step_progress=lambda step, total: step_updates.append((step, total)),
        )

        self.assertGreater(len(result.time_s), 0)
        self.assertGreater(result.summary.wallclock_s or 0.0, 0.0)
        self.assertEqual(len(result.position_m), len(result.time_s))
        self.assertTrue(step_updates, "on_step_progress should fire during the run")
        self.assertEqual(step_updates[-1][0], step_updates[-1][1])

        store = RunStore(runs_root=self._temp_runs_root())
        path = store.save(result)
        self.assertTrue(path.joinpath("run_result.json").is_file())
        self.assertTrue(path.joinpath("trajectory.csv").is_file())
        loaded = store.load(path)
        self.assertEqual(loaded.run_id, result.run_id)
        self.assertEqual(len(loaded.time_s), len(result.time_s))

    @staticmethod
    def _temp_runs_root() -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="dronesim_smoke_")


if __name__ == "__main__":
    unittest.main()
