from __future__ import annotations

import math
import sys
import threading
import time
import types
import unittest

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.sim import (
    BackendUnavailable,
    DroneFactory,
    InHouseMpcQuadBackend,
    JSBSimCessnaBackend,
    PlaceholderBackend,
    RunManager,
    SimulationBackend,
)
from dronesim.sim.backends_jsbsim import (
    _clamp_cruise_speed,
    _normalize_altitude_reference,
    _resolve_waypoints_3d,
    _state_is_finite,
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

    def test_default_factory_includes_jsbsim_cessna_backend(self) -> None:
        factory = DroneFactory()
        ids = {entry["backend_id"] for entry in factory.available()}
        self.assertIn("jsbsim_cessna", ids)


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


class _FakeJSBSimFDM:
    def __init__(self, root=None, _unused=None) -> None:
        self.root = root
        self.dt = 0.1
        self.props: dict[str, float] = {}
        self.loaded_model: str | None = None

    def load_model(self, model: str, add_model_to_path: bool = True) -> bool:
        del add_model_to_path
        self.loaded_model = model
        return True

    def set_dt(self, dt: float) -> None:
        self.dt = float(dt)

    def set_property_value(self, name: str, value: float) -> None:
        self.props[name] = float(value)

    def get_property_value(self, name: str) -> float:
        return self.props.get(name, 0.0)

    def run_ic(self) -> bool:
        self.props["position/lat-gc-deg"] = self.props.get("ic/lat-gc-deg", 37.6188056)
        self.props["position/long-gc-deg"] = self.props.get("ic/long-gc-deg", -122.3754167)
        self.props["position/h-sl-ft"] = self.props.get("ic/h-sl-ft", 5.0 * 3.280839895)
        self.props["attitude/psi-rad"] = self.props.get("ic/psi-true-deg", 0.0) * 3.141592653589793 / 180.0
        vc_kts = self.props.get("ic/vc-kts", 77.67)
        self.props["velocities/vt-fps"] = vc_kts * 1.68781
        self.props["velocities/u-fps"] = self.props["velocities/vt-fps"]
        self.props["velocities/v-east-fps"] = 0.0
        self.props["velocities/v-north-fps"] = self.props["velocities/vt-fps"]
        self.props["velocities/v-down-fps"] = 0.0
        return True

    def run(self) -> bool:
        dt = self.dt
        throttle = self.props.get("fcs/throttle-cmd-norm", 0.65)
        aileron = self.props.get("fcs/aileron-cmd-norm", 0.0)
        elevator = self.props.get("fcs/elevator-cmd-norm", 0.0)
        speed_fps = max(20.0, self.props.get("velocities/vt-fps", 120.0) + (throttle - 0.5) * 4.0)
        psi = self.props.get("attitude/psi-rad", 0.0) + aileron * 0.02
        speed_mps = speed_fps * 0.3048
        east_m = speed_mps * math_sin(psi) * dt
        north_m = speed_mps * math_cos(psi) * dt
        lat = self.props.get("position/lat-gc-deg", 37.6188056) + north_m / 110540.0
        lon = self.props.get("position/long-gc-deg", -122.3754167) + east_m / (
            111320.0 * math_cos(lat * 3.141592653589793 / 180.0)
        )
        alt_ft = self.props.get("position/h-sl-ft", 5.0 * 3.280839895) + elevator * dt * 12.0
        self.props["position/lat-gc-deg"] = lat
        self.props["position/long-gc-deg"] = lon
        self.props["position/h-sl-ft"] = alt_ft
        self.props["velocities/vt-fps"] = speed_fps
        self.props["velocities/v-east-fps"] = east_m / dt * 3.280839895
        self.props["velocities/v-north-fps"] = north_m / dt * 3.280839895
        self.props["velocities/v-down-fps"] = -elevator * 12.0
        self.props["attitude/phi-rad"] = aileron * 0.2
        self.props["attitude/theta-rad"] = elevator * 0.1
        self.props["attitude/psi-rad"] = psi
        self.props["velocities/p-rad_sec"] = aileron * 0.1
        self.props["velocities/q-rad_sec"] = elevator * 0.1
        self.props["velocities/r-rad_sec"] = aileron * 0.05
        return True


def math_sin(value: float) -> float:
    import math

    return math.sin(value)


def math_cos(value: float) -> float:
    import math

    return math.cos(value)


class JSBSimHelpersTest(unittest.TestCase):
    def test_clamp_cruise_speed(self) -> None:
        speed, clamped = _clamp_cruise_speed(150.0)
        self.assertTrue(clamped)
        self.assertEqual(speed, 70.0)
        speed, clamped = _clamp_cruise_speed(40.0)
        self.assertFalse(clamped)
        self.assertEqual(speed, 40.0)

    def test_normalize_altitude_reference(self) -> None:
        self.assertEqual(_normalize_altitude_reference("AGL"), "agl")
        self.assertEqual(_normalize_altitude_reference("msl"), "msl")
        self.assertEqual(_normalize_altitude_reference(None), "agl")

    def test_state_is_finite(self) -> None:
        self.assertTrue(_state_is_finite([0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
        self.assertFalse(_state_is_finite([float("nan"), 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]))

    def test_resolve_waypoints_agl_adds_terrain(self) -> None:
        scenario = _make_scenario()
        scenario.waypoints.waypoints[0].x_m = 0.0
        scenario.waypoints.waypoints[0].y_m = 0.0
        scenario.waypoints.waypoints[0].z_m = 100.0
        scenario.waypoints.waypoints[1].x_m = 100.0
        scenario.waypoints.waypoints[1].y_m = 0.0
        scenario.waypoints.waypoints[1].z_m = 100.0

        class _TerrainStub:
            def waypoint_to_local(self, wp, _map):
                return wp

            def fetch_map(self, _map, fetch_remote=False):
                del fetch_remote

                class _Asset:
                    @staticmethod
                    def elevation_at(x_m: float, y_m: float) -> float:
                        return 50.0 if x_m < 50.0 else 20.0

                return _Asset()

        resolved, alt_ref, warnings = _resolve_waypoints_3d(
            scenario,
            RunConfig(target_altitude_m=5.0),
            _TerrainStub(),  # type: ignore[arg-type]
            {"altitude_reference": "agl"},
        )
        self.assertEqual(alt_ref, "agl")
        self.assertEqual(len(warnings), 0)
        self.assertAlmostEqual(resolved[0][2], 150.0)
        self.assertAlmostEqual(resolved[1][2], 120.0)


class JSBSimCessnaBackendTest(unittest.TestCase):
    def test_unavailable_dependency_raises_helpful_error(self) -> None:
        scenario = _make_scenario()
        cfg = RunConfig(backend_id="jsbsim_cessna", max_steps=1)
        backend = JSBSimCessnaBackend()
        backend._jsbsim_available = False
        backend._import_error = "jsbsim import failed: missing"

        with self.assertRaises(BackendUnavailable) as ctx:
            backend.run(scenario, cfg)
        self.assertIn("requires jsbsim", str(ctx.exception))
        self.assertIn("requirements-jsbsim.txt", str(ctx.exception))

    def test_mocked_jsbsim_run_returns_normalized_result(self) -> None:
        fake_module = types.SimpleNamespace(FGFDMExec=_FakeJSBSimFDM, __version__="fake-test")
        prior = sys.modules.get("jsbsim")
        sys.modules["jsbsim"] = fake_module
        try:
            scenario = _make_scenario()
            scenario.vehicle.model_id = "jsbsim_c172"
            scenario.vehicle.model_type = "fixed_wing"
            scenario.vehicle.backend_id = "jsbsim_cessna"
            scenario.vehicle.display_name = "JSBSim Cessna 172"
            scenario.vehicle.parameters = {
                "aircraft": "c172p",
                "altitude_reference": "msl",
                "ic_settle_steps": 0,
            }
            scenario.waypoints.waypoints[0].z_m = 200.0
            scenario.waypoints.waypoints[1].z_m = 200.0
            scenario.vehicle.controller = {
                "type": "waypoint_autopilot",
                "cruise_speed_mps": 35.0,
                "max_bank_deg": 25.0,
                "waypoint_capture_radius_m": 10.0,
                "min_agl_m": 0.0,
            }
            cfg = RunConfig(
                backend_id="jsbsim_cessna",
                max_steps=5,
                dt_s=0.1,
                target_altitude_m=5.0,
                waypoint_threshold_m=0.1,
            )
            backend = JSBSimCessnaBackend()
            progress: list[tuple[int, int]] = []

            result = backend.run(
                scenario,
                cfg,
                on_step_progress=lambda step, total: progress.append((step, total)),
            )
        finally:
            if prior is None:
                sys.modules.pop("jsbsim", None)
            else:
                sys.modules["jsbsim"] = prior

        self.assertEqual(result.backend_id, "jsbsim_cessna")
        self.assertEqual(result.model_id, "jsbsim_c172")
        self.assertEqual(len(result.time_s), 5)
        self.assertEqual(len(result.position_m), len(result.time_s))
        self.assertEqual(len(result.velocity_mps), len(result.time_s))
        self.assertEqual(len(result.acceleration_mps2), len(result.time_s))
        self.assertEqual(len(result.controls[0]), 4)
        self.assertEqual(result.metadata["control_channels"], [
            "aileron_norm",
            "elevator_norm",
            "rudder_norm",
            "throttle_norm",
        ])
        self.assertEqual(progress[-1], (5, 5))
        self.assertEqual(result.metadata.get("altitude_reference"), "msl")
        self.assertIn("waypoints_local_xyz_msl", result.metadata)

    def test_live_jsbsim_short_run_stays_finite(self) -> None:
        try:
            import jsbsim  # noqa: F401
        except Exception:
            self.skipTest("jsbsim not installed")

        scenario = _make_scenario()
        scenario.vehicle.model_id = "jsbsim_c172"
        scenario.vehicle.model_type = "fixed_wing"
        scenario.vehicle.backend_id = "jsbsim_cessna"
        scenario.vehicle.parameters = {
            "aircraft": "c172p",
            "altitude_reference": "msl",
            "initial_ias_mps": 45.0,
            "ic_settle_steps": 10,
        }
        scenario.vehicle.controller = {
            "type": "waypoint_autopilot",
            "cruise_speed_mps": 45.0,
            "max_bank_deg": 20.0,
            "waypoint_capture_radius_m": 50.0,
            "heading_gain": 1.0,
            "altitude_gain": 0.008,
        }
        scenario.waypoints.waypoints[0].z_m = 500.0
        scenario.waypoints.waypoints[0].alt_m = 500.0
        scenario.waypoints.waypoints[1].x_m = 800.0
        scenario.waypoints.waypoints[1].y_m = 0.0
        scenario.waypoints.waypoints[1].z_m = 500.0
        scenario.waypoints.waypoints[1].alt_m = 500.0
        cfg = RunConfig(
            backend_id="jsbsim_cessna",
            max_steps=400,
            dt_s=0.05,
            target_altitude_m=500.0,
            waypoint_threshold_m=80.0,
        )
        result = JSBSimCessnaBackend().run(scenario, cfg)
        z_vals = [p[2] for p in result.position_m if len(p) > 2]
        self.assertGreater(len(z_vals), 150)
        self.assertTrue(all(math.isfinite(v) for v in z_vals))
        self.assertGreater(max(z_vals), 200.0)
        self.assertGreater(result.summary.duration_s, 7.0)
        self.assertNotEqual(result.status, "unstable")


if __name__ == "__main__":
    unittest.main()
