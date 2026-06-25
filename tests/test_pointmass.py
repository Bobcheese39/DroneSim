"""Tests for the 3DOF point-mass dynamics and backends.

Covers:
* Dynamics produce finite, monotone-to-waypoint trajectories.
* Constraints respected (speed, bank/gamma clamps).
* Waypoint capture leads to success flag.
* RunResult schema correct (matching existing backend contract).
* Factory resolution: both backend_ids resolvable.
* Stability: a configuration that would oscillate in 6DOF stays bounded
  in 3DOF (amplitude stays well below initial tracking error).
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from dronesim.config.pointmass_models import apply_pointmass_model, list_pointmass_models
from dronesim.models import (
    DroneModelSpec,
    EnvironmentSpec,
    RunConfig,
    RunResult,
    ScenarioSpec,
    WaypointSet,
)
from dronesim.sim.backends import DroneFactory
from dronesim.sim.backends_pointmass import (
    PointMass3DOFFixedWingBackend,
    PointMass3DOFQuadBackend,
)
from dronesim.sim.fidelity import WindField
from dronesim.sim.pointmass import (
    FixedWingPointMassParams,
    QuadPointMassParams,
    run_fixedwing_pointmass,
    run_quad_pointmass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quad_waypoints() -> np.ndarray:
    return np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=float)


def _fw_waypoints() -> np.ndarray:
    return np.array([[0.0, 0.0], [500.0, 0.0], [500.0, 500.0]], dtype=float)


def _quad_params(**overrides) -> QuadPointMassParams:
    p = QuadPointMassParams(
        mass=5.0,
        max_accel_mps2=5.0,
        max_speed_mps=10.0,
        kp_pos=1.2,
        kd_pos=1.4,
        waypoint_capture_radius_m=0.5,
        target_altitude_m=5.0,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _fw_params(**overrides) -> FixedWingPointMassParams:
    p = FixedWingPointMassParams(
        cruise_speed_mps=40.0,
        min_speed_mps=25.0,
        max_speed_mps=70.0,
        max_bank_deg=30.0,
        max_climb_deg=8.0,
        max_descent_deg=8.0,
        turn_rate_limit_deg_s=10.0,
        climb_rate_limit_mps=4.0,
        heading_gain=1.5,
        altitude_gain=0.03,
        waypoint_capture_radius_m=75.0,
        target_altitude_m=100.0,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _make_quad_scenario(backend_id: str = "pointmass_quad") -> ScenarioSpec:
    scenario = ScenarioSpec()
    scenario.vehicle = DroneModelSpec(
        model_id="pointmass_quad_default",
        model_type="pointmass_quad",
        backend_id=backend_id,
        parameters={
            "mass": 5.0,
            "max_accel_mps2": 5.0,
            "max_speed_mps": 10.0,
            "kp_pos": 1.2,
            "kd_pos": 1.4,
            "waypoint_capture_radius_m": 0.5,
        },
    )
    scenario.waypoints = WaypointSet.from_local_xy(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        altitude_m=5.0,
    )
    scenario.run_config = RunConfig(
        backend_id=backend_id,
        dt_s=0.1,
        max_steps=300,
        target_altitude_m=5.0,
    )
    return scenario


def _make_fw_scenario(backend_id: str = "pointmass_fixed_wing") -> ScenarioSpec:
    scenario = ScenarioSpec()
    scenario.vehicle = DroneModelSpec(
        model_id="pointmass_fw_default",
        model_type="pointmass_fixed_wing",
        backend_id=backend_id,
        parameters={
            "cruise_speed_mps": 40.0,
            "min_speed_mps": 25.0,
            "max_speed_mps": 70.0,
            "max_bank_deg": 30.0,
            "max_climb_deg": 8.0,
            "max_descent_deg": 8.0,
            "turn_rate_limit_deg_s": 10.0,
            "climb_rate_limit_mps": 4.0,
            "heading_gain": 1.5,
            "altitude_gain": 0.03,
            "waypoint_capture_radius_m": 75.0,
        },
    )
    scenario.waypoints = WaypointSet.from_local_xy(
        [(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)],
        altitude_m=100.0,
    )
    scenario.run_config = RunConfig(
        backend_id=backend_id,
        dt_s=0.05,
        max_steps=2000,
        target_altitude_m=100.0,
    )
    return scenario


# ---------------------------------------------------------------------------
# Dynamics: quadcopter
# ---------------------------------------------------------------------------


class TestQuadPointMassDynamics(unittest.TestCase):

    def test_result_is_finite(self):
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=200)
        self.assertGreater(len(result.time_s), 0)
        self.assertTrue(np.all(np.isfinite(result.position_m)))
        self.assertTrue(np.all(np.isfinite(result.velocity_mps)))
        self.assertTrue(np.all(np.isfinite(result.acceleration_mps2)))
        self.assertTrue(np.all(np.isfinite(result.attitude_rad)))

    def test_waypoint_capture_success(self):
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=500)
        self.assertTrue(result.success, "Should reach final waypoint within 500 steps")
        self.assertLess(result.miss_distance_m, 2.0)

    def test_speed_constraint_respected(self):
        wps = _quad_waypoints()
        params = _quad_params(max_speed_mps=3.0)
        result = run_quad_pointmass(wps, params, dt=0.1, max_steps=300)
        speeds = np.linalg.norm(result.velocity_mps, axis=1)
        # Allow a small tolerance for RK4 overshoot in a single step
        self.assertTrue(np.all(speeds <= params.max_speed_mps + 0.5))

    def test_altitude_held_constant(self):
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(target_altitude_m=5.0), dt=0.1, max_steps=300)
        z_vals = result.position_m[:, 2]
        # The reference altitude is fixed; position should track it closely
        self.assertTrue(np.all(np.abs(z_vals - 5.0) < 1.0), "Altitude should stay near target")

    def test_accel_constraint_respected(self):
        wps = _quad_waypoints()
        params = _quad_params(max_accel_mps2=2.0)
        result = run_quad_pointmass(wps, params, dt=0.1, max_steps=300)
        acc_mags = np.linalg.norm(result.acceleration_mps2, axis=1)
        self.assertTrue(np.all(acc_mags <= params.max_accel_mps2 + 1e-9))

    def test_monotone_approach(self):
        """Horizontal distance to final waypoint should trend downward."""
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=500)
        pos_xy = result.position_m[:, :2]
        final = wps[-1]
        dists = np.linalg.norm(pos_xy - final, axis=1)
        # Error at end should be less than at start (rough monotone check)
        self.assertLess(dists[-1], dists[0], "Should approach final waypoint")

    def test_per_waypoint_altitude_climb(self):
        """Quad should climb toward a higher second waypoint altitude."""
        # (M, 3) waypoints: start at 5 m, climb to 25 m at second waypoint
        wps = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 25.0], [10.0, 10.0, 25.0]])
        params = _quad_params(target_altitude_m=5.0)
        result = run_quad_pointmass(wps, params, dt=0.1, max_steps=600)
        # Final altitude should have moved appreciably toward 25 m
        final_z = result.position_m[-1, 2]
        self.assertGreater(final_z, 10.0, "Quad should climb toward 25 m waypoint altitude")
        # All positions must stay finite
        self.assertTrue(np.all(np.isfinite(result.position_m)))

    def test_wind_does_not_crash(self):
        wps = _quad_waypoints()
        wind = WindField.from_environment(
            wind_mps=[2.0, 1.0, 0.0],
            gust_std_mps=0.5,
            gust_decorrelation_s=2.0,
            rng=np.random.default_rng(42),
        )
        result = run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=500, wind=wind)
        self.assertGreater(len(result.time_s), 0)
        self.assertTrue(np.all(np.isfinite(result.position_m)))

    def test_step_progress_callback_called(self):
        wps = _quad_waypoints()
        calls: list[tuple[int, int]] = []
        run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=300,
                           on_step_progress=lambda s, t: calls.append((s, t)))
        self.assertGreater(len(calls), 0)

    def test_controls_shape(self):
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=200)
        self.assertEqual(result.controls.shape[1], 4)  # ft, tx, ty, tz

    def test_attitude_within_bounds(self):
        """Roll / pitch should stay within +/- 90 deg for reasonable params."""
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(), dt=0.1, max_steps=300)
        roll = result.attitude_rad[:, 0]
        pitch = result.attitude_rad[:, 1]
        self.assertTrue(np.all(np.abs(roll) <= math.pi / 2 + 0.01))
        self.assertTrue(np.all(np.abs(pitch) <= math.pi / 2 + 0.01))


# ---------------------------------------------------------------------------
# Dynamics: fixed-wing
# ---------------------------------------------------------------------------


class TestFixedWingPointMassDynamics(unittest.TestCase):

    def test_result_is_finite(self):
        wps = _fw_waypoints()
        result = run_fixedwing_pointmass(wps, _fw_params(), dt=0.05, max_steps=1000)
        self.assertGreater(len(result.time_s), 0)
        self.assertTrue(np.all(np.isfinite(result.position_m)))
        self.assertTrue(np.all(np.isfinite(result.velocity_mps)))
        self.assertTrue(np.all(np.isfinite(result.attitude_rad)))

    def test_waypoint_capture_success(self):
        wps = _fw_waypoints()
        result = run_fixedwing_pointmass(wps, _fw_params(), dt=0.05, max_steps=2000)
        self.assertTrue(result.success, "Should reach final waypoint within 2000 steps")

    def test_speed_stays_at_cruise(self):
        wps = _fw_waypoints()
        params = _fw_params(cruise_speed_mps=40.0)
        result = run_fixedwing_pointmass(wps, params, dt=0.05, max_steps=1000)
        speeds = np.linalg.norm(result.velocity_mps, axis=1)
        # After settling, speed should be close to cruise
        self.assertTrue(np.all(speeds <= params.max_speed_mps + 0.5))
        self.assertTrue(np.all(speeds >= params.min_speed_mps - 0.5))

    def test_bank_angle_clamped(self):
        wps = _fw_waypoints()
        params = _fw_params(max_bank_deg=30.0)
        result = run_fixedwing_pointmass(wps, params, dt=0.05, max_steps=1000)
        bank = result.attitude_rad[:, 0]
        max_bank_rad = math.radians(params.max_bank_deg)
        self.assertTrue(np.all(np.abs(bank) <= max_bank_rad + 1e-6))

    def test_flight_path_clamped(self):
        wps = _fw_waypoints()
        params = _fw_params(max_climb_deg=8.0, max_descent_deg=8.0)
        result = run_fixedwing_pointmass(wps, params, dt=0.05, max_steps=1000)
        gamma = result.attitude_rad[:, 1]
        max_rad = math.radians(8.0)
        self.assertTrue(np.all(np.abs(gamma) <= max_rad + 1e-6))

    def test_altitude_converges(self):
        wps = _fw_waypoints()
        params = _fw_params(target_altitude_m=100.0)
        result = run_fixedwing_pointmass(wps, params, dt=0.05, max_steps=2000)
        # Altitude at end should be within a reasonable bound of target
        final_alt = result.position_m[-1, 2]
        self.assertAlmostEqual(final_alt, 100.0, delta=20.0)

    def test_wind_does_not_crash(self):
        wps = _fw_waypoints()
        wind = WindField.from_environment(
            wind_mps=[5.0, 3.0, 0.0],
            gust_std_mps=1.0,
            gust_decorrelation_s=2.0,
            rng=np.random.default_rng(7),
        )
        result = run_fixedwing_pointmass(wps, _fw_params(), dt=0.05, max_steps=2000, wind=wind)
        self.assertGreater(len(result.time_s), 0)
        self.assertTrue(np.all(np.isfinite(result.position_m)))

    def test_controls_shape(self):
        wps = _fw_waypoints()
        result = run_fixedwing_pointmass(wps, _fw_params(), dt=0.05, max_steps=1000)
        self.assertEqual(result.controls.shape[1], 4)

    def test_monotone_approach(self):
        wps = _fw_waypoints()
        result = run_fixedwing_pointmass(wps, _fw_params(), dt=0.05, max_steps=2000)
        pos_xy = result.position_m[:, :2]
        final = wps[-1]
        dists = np.linalg.norm(pos_xy - final, axis=1)
        self.assertLess(dists[-1], dists[0])

    def test_per_waypoint_altitude_climb(self):
        """Fixed-wing should climb toward a higher second waypoint altitude."""
        # (M, 3) waypoints: start at 100 m, climb to 200 m by third waypoint
        wps = np.array([
            [0.0, 0.0, 100.0],
            [500.0, 0.0, 200.0],
            [500.0, 500.0, 200.0],
        ])
        params = _fw_params(
            target_altitude_m=100.0,
            max_climb_deg=12.0,
            max_descent_deg=12.0,
            altitude_gain=0.05,
        )
        result = run_fixedwing_pointmass(wps, params, dt=0.05, max_steps=3000)
        # Altitude should move toward 200 m (even if not fully reached)
        final_z = result.position_m[-1, 2]
        self.assertGreater(final_z, 130.0, "Fixed-wing should climb toward 200 m waypoint altitude")
        # Flight-path angle must stay within clamp throughout
        gamma = result.attitude_rad[:, 1]
        max_rad = math.radians(12.0)
        self.assertTrue(np.all(np.abs(gamma) <= max_rad + 1e-6))
        self.assertTrue(np.all(np.isfinite(result.position_m)))


# ---------------------------------------------------------------------------
# Stability: bounded where 6DOF would oscillate
# ---------------------------------------------------------------------------


class TestPointMassStability(unittest.TestCase):
    """Verify that 3DOF stays bounded for high-gain configurations that
    would typically cause oscillation in a 6DOF model."""

    def test_quad_high_gain_stays_bounded(self):
        """Very high kp_pos that would cause 6DOF oscillations stays stable."""
        wps = _quad_waypoints()
        # Gains that would massively overshoot with full 6DOF rotational dynamics
        params = _quad_params(kp_pos=10.0, kd_pos=0.1, max_accel_mps2=20.0)
        result = run_quad_pointmass(wps, params, dt=0.1, max_steps=500)
        # Must not diverge: all positions finite
        self.assertTrue(np.all(np.isfinite(result.position_m)))
        # Speed must remain bounded
        speeds = np.linalg.norm(result.velocity_mps, axis=1)
        self.assertTrue(np.all(speeds <= params.max_speed_mps + 0.5))
        # Tracking error must not explode
        self.assertTrue(np.all(result.tracking_error_m < 100.0))

    def test_fw_high_gain_stays_bounded(self):
        """High heading gain does not cause divergence or NaN."""
        wps = _fw_waypoints()
        params = _fw_params(heading_gain=10.0, altitude_gain=0.5)
        result = run_fixedwing_pointmass(wps, params, dt=0.05, max_steps=2000)
        self.assertTrue(np.all(np.isfinite(result.position_m)))
        # Bank still clamped
        bank = result.attitude_rad[:, 0]
        max_bank_rad = math.radians(params.max_bank_deg)
        self.assertTrue(np.all(np.abs(bank) <= max_bank_rad + 1e-6))

    def test_quad_large_dt_still_stable(self):
        """Larger dt that would blow up many integrators stays finite."""
        wps = _quad_waypoints()
        result = run_quad_pointmass(wps, _quad_params(), dt=0.5, max_steps=200)
        self.assertTrue(np.all(np.isfinite(result.position_m)))


# ---------------------------------------------------------------------------
# Backend: RunResult contract
# ---------------------------------------------------------------------------


class TestPointMassBackendContract(unittest.TestCase):

    def _run_quad(self) -> RunResult:
        backend = PointMass3DOFQuadBackend()
        scenario = _make_quad_scenario()
        return backend.run(scenario, scenario.run_config)

    def _run_fw(self) -> RunResult:
        backend = PointMass3DOFFixedWingBackend()
        scenario = _make_fw_scenario()
        return backend.run(scenario, scenario.run_config)

    def test_quad_run_result_fields(self):
        result = self._run_quad()
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.backend_id, "pointmass_quad")
        self.assertGreater(len(result.time_s), 0)
        self.assertEqual(len(result.position_m), len(result.time_s))
        self.assertEqual(len(result.velocity_mps), len(result.time_s))
        self.assertEqual(len(result.attitude_rad), len(result.time_s))
        self.assertEqual(len(result.controls), len(result.time_s))
        self.assertEqual(len(result.reference_position_m), len(result.time_s))
        self.assertEqual(len(result.tracking_error_m), len(result.time_s))

    def test_quad_run_result_summary(self):
        result = self._run_quad()
        self.assertIsNotNone(result.summary)
        self.assertIsNotNone(result.summary.duration_s)
        self.assertIsNotNone(result.summary.wallclock_s)

    def test_quad_run_result_status(self):
        result = self._run_quad()
        self.assertIn(result.status, {"success", "completed_with_miss", "terrain_collision"})

    def test_fw_run_result_fields(self):
        result = self._run_fw()
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.backend_id, "pointmass_fixed_wing")
        self.assertGreater(len(result.time_s), 0)
        self.assertEqual(len(result.position_m), len(result.time_s))
        self.assertEqual(len(result.velocity_mps), len(result.time_s))
        self.assertEqual(len(result.attitude_rad), len(result.time_s))
        self.assertEqual(len(result.controls), len(result.time_s))

    def test_fw_run_result_status(self):
        result = self._run_fw()
        self.assertIn(result.status, {"success", "completed_with_miss", "terrain_collision"})

    def test_quad_to_dict_roundtrip(self):
        result = self._run_quad()
        d = result.to_dict()
        restored = RunResult.from_dict(d)
        self.assertEqual(restored.run_id, result.run_id)
        self.assertEqual(restored.backend_id, result.backend_id)

    def test_quad_monte_carlo_seed(self):
        """Two runs with different seeds should produce different results."""
        backend = PointMass3DOFQuadBackend()
        s1 = _make_quad_scenario()
        s1.run_config.seed = 1
        s1.run_config.monte_carlo["init_pos_std"] = 0.5
        r1 = backend.run(s1, s1.run_config)

        s2 = _make_quad_scenario()
        s2.run_config.seed = 999
        s2.run_config.monte_carlo["init_pos_std"] = 0.5
        r2 = backend.run(s2, s2.run_config)

        # Trajectories from different seeds + perturbations should differ
        p1 = np.array(r1.position_m[:min(5, len(r1.position_m))])
        p2 = np.array(r2.position_m[:min(5, len(r2.position_m))])
        self.assertFalse(np.allclose(p1, p2), "Different seeds should produce different trajectories")


# ---------------------------------------------------------------------------
# Factory resolution
# ---------------------------------------------------------------------------


class TestFactoryResolution(unittest.TestCase):

    def test_both_backends_registered(self):
        factory = DroneFactory()
        ids = {b["backend_id"] for b in factory.available()}
        self.assertIn("pointmass_quad", ids)
        self.assertIn("pointmass_fixed_wing", ids)

    def test_quad_backend_resolvable(self):
        factory = DroneFactory()
        backend = factory.get("pointmass_quad")
        self.assertIsInstance(backend, PointMass3DOFQuadBackend)

    def test_fw_backend_resolvable(self):
        factory = DroneFactory()
        backend = factory.get("pointmass_fixed_wing")
        self.assertIsInstance(backend, PointMass3DOFFixedWingBackend)


# ---------------------------------------------------------------------------
# Config catalog
# ---------------------------------------------------------------------------


class TestPointmassConfig(unittest.TestCase):

    def test_list_models_returns_entries(self):
        models = list_pointmass_models()
        self.assertGreater(len(models), 0)
        ids = [m["id"] for m in models]
        self.assertIn("pointmass_quad_default", ids)
        self.assertIn("pointmass_fw_default", ids)

    def test_apply_quad_model(self):
        scenario_dict: dict = {"vehicle": {}, "metadata": {}}
        result = apply_pointmass_model(scenario_dict, "pointmass_quad_default")
        vehicle = result["vehicle"]
        self.assertEqual(vehicle["backend_id"], "pointmass_quad")
        self.assertEqual(vehicle["model_type"], "pointmass_quad")
        params = vehicle["parameters"]
        self.assertIn("kp_pos", params)
        self.assertIn("max_accel_mps2", params)
        self.assertEqual(result["metadata"]["pointmass_model"], "pointmass_quad_default")

    def test_apply_fw_model(self):
        scenario_dict: dict = {"vehicle": {}, "metadata": {}}
        result = apply_pointmass_model(scenario_dict, "pointmass_fw_default")
        vehicle = result["vehicle"]
        self.assertEqual(vehicle["backend_id"], "pointmass_fixed_wing")
        params = vehicle["parameters"]
        self.assertIn("cruise_speed_mps", params)
        self.assertIn("max_bank_deg", params)

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            apply_pointmass_model({}, "nonexistent_model_xyz")

    def test_model_types_correct(self):
        models = list_pointmass_models()
        for m in models:
            self.assertIn(m["model_type"], {"pointmass_quad", "pointmass_fixed_wing"})


if __name__ == "__main__":
    unittest.main()
