"""Phase 6 fidelity tests.

Covers:

* Legacy parity (default scenario produces identical numbers via the
  vendor path; forced ``fidelity_mode="extended"`` with no knobs also
  matches because ExtendedQuadcopter delegates to vendor Euler.)
* Aerodynamic drag reduces achievable horizontal speed.
* Wind biases hover position when drag is non-zero.
* RK4 stays within a sane tolerance of Euler on a nominal scenario.
* Terrain collision terminates the run with the expected status.
* PyBullet backend raises BackendUnavailable when PyFlyt is not installed.
* PyBullet backend produces a RunResult when PyFlyt is available.

These tests exercise the real cvxpy/MPC pipeline through the vendored
``drone_mc`` package, so they are slower than the pure-unit tests in
``test_backends.py``. Each scenario uses a short ``max_steps`` to keep
runtimes reasonable on developer machines.
"""
from __future__ import annotations

import unittest

import numpy as np

from dronesim.models import (
    DroneModelSpec,
    EnvironmentSpec,
    RunConfig,
    ScenarioSpec,
    WaypointSet,
    Waypoint,
)
from dronesim.sim import BackendUnavailable, InHouseMpcQuadBackend, PyBulletQuadBackend
from dronesim.sim.fidelity import (
    AeroParams,
    ExtendedQuadcopter,
    TerrainCollision,
    WindField,
    _ensure_vendor_path,
)
from dronesim.sim.inhouse_quad_runtime import run_simulation_local


def _short_scenario(
    *,
    environment: EnvironmentSpec | None = None,
    vehicle: DroneModelSpec | None = None,
    integration_method: str = "euler",
    fidelity_mode: str = "auto",
    max_steps: int = 60,
    seed: int = 0,
) -> ScenarioSpec:
    """Build a small, fast scenario suitable for fidelity tests."""
    scenario = ScenarioSpec(
        name="fidelity-test",
        waypoints=WaypointSet.from_local_xy(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], altitude_m=5.0,
        ),
        environment=environment or EnvironmentSpec(),
        vehicle=vehicle or DroneModelSpec(),
        run_config=RunConfig(
            dt_s=0.1,
            max_steps=max_steps,
            target_altitude_m=5.0,
            integration_method=integration_method,
            fidelity_mode=fidelity_mode,
            seed=seed,
        ),
    )
    scenario.validate()
    return scenario


def _max_horizontal_speed(result) -> float:
    vel = np.asarray(result.velocity_mps, dtype=float)
    if vel.size == 0:
        return 0.0
    horizontal = np.linalg.norm(vel[:, :2], axis=1)
    return float(np.nanmax(horizontal))


def _mean_x(result) -> float:
    pos = np.asarray(result.position_m, dtype=float)
    if pos.size == 0:
        return 0.0
    return float(np.nanmean(pos[:, 0]))


def _build_sim_cfg(backend: InHouseMpcQuadBackend, scenario: ScenarioSpec):
    """Helper that walks the same _build_sim_config path the backend uses."""
    return backend._build_sim_config(scenario, scenario.run_config)  # noqa: SLF001


class LegacyParityTest(unittest.TestCase):
    """Default scenarios must keep producing bit-identical traces."""

    def test_default_scenario_uses_vendor_path(self) -> None:
        scenario = _short_scenario()
        backend = InHouseMpcQuadBackend()
        self.assertFalse(backend._wants_extended(scenario, scenario.run_config))  # noqa: SLF001

        result = backend.run(scenario, scenario.run_config)
        # ``fidelity_path`` is recorded in metadata so external observers (and
        # tests) can verify which integration loop ran.
        self.assertEqual(result.metadata.get("fidelity_path"), "vendor")
        self.assertGreater(len(result.position_m), 0)

    def test_forced_extended_with_no_knobs_matches_vendor(self) -> None:
        """ExtendedQuadcopter should delegate to vendor Euler when no knobs are set."""
        scenario_vendor = _short_scenario(fidelity_mode="legacy", seed=11)
        scenario_extended = _short_scenario(fidelity_mode="extended", seed=11)

        backend = InHouseMpcQuadBackend()
        vendor_result = backend.run(scenario_vendor, scenario_vendor.run_config)
        extended_result = backend.run(scenario_extended, scenario_extended.run_config)

        self.assertEqual(vendor_result.metadata.get("fidelity_path"), "vendor")
        self.assertEqual(extended_result.metadata.get("fidelity_path"), "extended")

        vendor_pos = np.asarray(vendor_result.position_m, dtype=float)
        extended_pos = np.asarray(extended_result.position_m, dtype=float)
        self.assertEqual(vendor_pos.shape, extended_pos.shape)
        np.testing.assert_allclose(extended_pos, vendor_pos, atol=1e-9, rtol=0.0)

        vendor_vel = np.asarray(vendor_result.velocity_mps, dtype=float)
        extended_vel = np.asarray(extended_result.velocity_mps, dtype=float)
        np.testing.assert_allclose(extended_vel, vendor_vel, atol=1e-9, rtol=0.0)


class WindAndDragTest(unittest.TestCase):
    """High drag must damp velocity; wind plus drag must bias trajectory."""

    def test_drag_reduces_max_horizontal_speed(self) -> None:
        baseline = _short_scenario(
            vehicle=DroneModelSpec(parameters={
                "mass": 5.0, "Ix": 1.0, "Iy": 1.0, "Iz": 1.5,
                "aero": {"cd_linear": 0.0, "cd_quadratic": 0.0, "reference_area_m2": 0.1},
            }),
        )
        dragged = _short_scenario(
            vehicle=DroneModelSpec(parameters={
                "mass": 5.0, "Ix": 1.0, "Iy": 1.0, "Iz": 1.5,
                # Aggressive drag so the effect is unambiguous within ~60 steps.
                "aero": {"cd_linear": 2.0, "cd_quadratic": 5.0, "reference_area_m2": 0.1},
            }),
        )

        backend = InHouseMpcQuadBackend()
        baseline_result = backend.run(baseline, baseline.run_config)
        drag_result = backend.run(dragged, dragged.run_config)

        baseline_speed = _max_horizontal_speed(baseline_result)
        drag_speed = _max_horizontal_speed(drag_result)
        self.assertGreater(baseline_speed, 0.0)
        self.assertLess(
            drag_speed,
            baseline_speed,
            msg=f"Drag run ({drag_speed:.3f} m/s) should be slower than baseline "
            f"({baseline_speed:.3f} m/s)",
        )

    def test_wind_biases_trajectory_east(self) -> None:
        # Wind needs drag to couple to the airframe (momentum transfer through
        # the airstream); both must be enabled for the bias to appear.
        drag_block = {"cd_linear": 1.0, "cd_quadratic": 4.0, "reference_area_m2": 0.1}
        no_wind = _short_scenario(
            environment=EnvironmentSpec(wind_mps=[0.0, 0.0, 0.0]),
            vehicle=DroneModelSpec(parameters={
                "mass": 5.0, "Ix": 1.0, "Iy": 1.0, "Iz": 1.5, "aero": drag_block,
            }),
        )
        east_wind = _short_scenario(
            environment=EnvironmentSpec(wind_mps=[5.0, 0.0, 0.0]),
            vehicle=DroneModelSpec(parameters={
                "mass": 5.0, "Ix": 1.0, "Iy": 1.0, "Iz": 1.5, "aero": drag_block,
            }),
        )

        backend = InHouseMpcQuadBackend()
        no_wind_result = backend.run(no_wind, no_wind.run_config)
        east_result = backend.run(east_wind, east_wind.run_config)

        # The MPC will fight the wind, but the controller is unaware of the
        # disturbance so a measurable east bias should accumulate.
        self.assertGreater(
            _mean_x(east_result),
            _mean_x(no_wind_result),
            msg="Easterly wind should bias mean x position east of no-wind run",
        )


class IntegrationMethodTest(unittest.TestCase):
    """RK4 should track Euler closely on a tame scenario."""

    def test_rk4_close_to_euler(self) -> None:
        euler_scenario = _short_scenario(
            integration_method="euler", fidelity_mode="extended", seed=7
        )
        rk4_scenario = _short_scenario(
            integration_method="rk4", fidelity_mode="extended", seed=7
        )

        backend = InHouseMpcQuadBackend()
        euler_result = backend.run(euler_scenario, euler_scenario.run_config)
        rk4_result = backend.run(rk4_scenario, rk4_scenario.run_config)

        euler_final = np.asarray(euler_result.position_m, dtype=float)[-1]
        rk4_final = np.asarray(rk4_result.position_m, dtype=float)[-1]
        diff = float(np.linalg.norm(rk4_final - euler_final))
        # Loose sanity bound: methods should agree within meters on a
        # 60-step scenario with no disturbances. Sharper bounds depend on
        # MPC re-solve noise.
        self.assertLess(
            diff, 1.0,
            msg=f"RK4 final position diverged from Euler by {diff:.3f} m",
        )


class TerrainCollisionTest(unittest.TestCase):
    """Terrain collision must terminate the run with the right status."""

    def test_local_runtime_terminates_on_collision(self) -> None:
        _ensure_vendor_path()
        from drone_mc.config import SimConfig

        sim_cfg = SimConfig(
            waypoints=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float),
            altitude=5.0,
            dt=0.1,
            horizon=10,
            max_steps=40,
            lookahead=20,
        )

        # Synthetic flat terrain at 4.6 m. With default 0.5 m offset and the
        # quad starting at z=5.0 the clearance is 0.4 m -> immediate collision.
        def terrain_query(x_m: float, y_m: float) -> float:
            del x_m, y_m
            return 4.6

        aero = AeroParams()
        wind = WindField()
        collision = TerrainCollision(query=terrain_query, offset_m=0.5)

        def factory(cfg):
            init_kwargs = dict(cfg.init_state)
            init_kwargs.update({"mass": cfg.mass, "Ix": cfg.Ix, "Iy": cfg.Iy, "Iz": cfg.Iz})
            return ExtendedQuadcopter(
                dt=cfg.dt,
                aero=aero,
                wind=wind,
                integration_method="euler",
                collision=collision,
                **init_kwargs,
            )

        result = run_simulation_local(
            sim_cfg,
            dynamics_factory=factory,
            terrain_query=terrain_query,
            terrain_offset_m=0.5,
        )

        self.assertEqual(result.cfg_summary.get("terminated_by"), "terrain_collision")
        self.assertIn("collision", result.cfg_summary)
        # Early termination => fewer than max_steps samples recorded.
        self.assertLess(len(result.time), sim_cfg.max_steps)


class PyBulletBackendTest(unittest.TestCase):
    """PyFlyt backend availability and optional integration."""

    def test_run_raises_backend_unavailable_without_pyflyt(self) -> None:
        backend = PyBulletQuadBackend()
        if backend.availability_summary()["pyflyt"]:
            self.skipTest("PyFlyt is installed; unavailable-path test not applicable")
        scenario = _short_scenario()
        with self.assertRaises(BackendUnavailable):
            backend.run(scenario)

    def test_availability_summary_has_expected_keys(self) -> None:
        summary = PyBulletQuadBackend().availability_summary()
        for key in ("pyflyt", "pybullet", "import_error"):
            self.assertIn(key, summary)

    @unittest.skipUnless(PyBulletQuadBackend().availability_summary()["pyflyt"], "PyFlyt not installed")
    def test_run_with_pyflyt_produces_run_result(self) -> None:
        scenario = ScenarioSpec(
            name="pyflyt-smoke",
            waypoints=WaypointSet(
                waypoints=[
                    Waypoint.local(0.0, 0.0, 1.0, label="WP1"),
                    Waypoint.local(2.0, 0.0, 1.0, label="WP2"),
                ]
            ),
            run_config=RunConfig(
                backend_id="pybullet_quad",
                dt_s=0.1,
                max_steps=80,
                target_altitude_m=1.0,
                waypoint_threshold_m=0.5,
            ),
        )
        scenario.validate()
        result = PyBulletQuadBackend().run(scenario)
        self.assertEqual(result.backend_id, "pybullet_quad")
        self.assertGreater(len(result.time_s), 0)
        self.assertGreater(len(result.position_m), 0)
        self.assertIn(result.status, ("success", "completed_with_miss"))
        self.assertTrue(result.metadata.get("pyflyt_backend"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
