"""PyBullet / PyFlyt quadcopter backend.

Uses PyFlyt's :class:`PyFlyt.core.Aviary` with QuadX position-control mode
(``set_mode(7)``) to follow scenario waypoints and return a normalized
:class:`dronesim.models.RunResult` compatible with replay and analysis.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.services.terrain import TerrainService
from dronesim.sim.backends import (
    BackendUnavailable,
    SimulationBackend,
    StepProgressCb,
)
from dronesim.sim.debug_log import get_sim_logger

logger = get_sim_logger(__name__)

_CONTROL_HZ = 120
_MIN_SPAWN_Z_M = 0.5

_INSTALL_HINT = (
    "Install PyFlyt (preferred) or gym-pybullet-drones to enable this backend. "
    "Try `pip install -r requirements-pybullet.txt` or follow "
    "https://github.com/bmabsout/gym-pybullet-drones."
)


def _detect_libs() -> tuple[bool, bool, str | None]:
    """Return ``(pyflyt_available, pybullet_available, error_message)``.

    Soft import only -- never raises. The detection runs once per
    backend instance because importing pybullet has measurable side
    effects (spawns a shared library and prints a banner on some
    platforms).
    """
    pyflyt_ok = False
    pybullet_ok = False
    err: str | None = None
    try:
        import PyFlyt.core  # noqa: F401

        pyflyt_ok = True
    except Exception as exc:  # noqa: BLE001
        err = f"PyFlyt import failed: {exc}"
    try:
        import pybullet  # noqa: F401

        pybullet_ok = True
    except Exception as exc:  # noqa: BLE001
        if err is None:
            err = f"pybullet import failed: {exc}"
        else:
            err = f"{err}; pybullet import failed: {exc}"
    return pyflyt_ok, pybullet_ok, err


def _resolve_waypoints_3d(
    scenario: ScenarioSpec,
    run_config: RunConfig,
    terrain_service: TerrainService,
) -> tuple[list[tuple[float, float, float]], float]:
    """Convert scenario waypoints to local ENU (x, y, z) and a PyFlyt z offset.

    PyFlyt uses a flat ground plane at z=0. Negative terrain-relative
    altitudes are lifted uniformly so spawn height and setpoints stay above
    the ground plane.
    """
    scenario.validate()
    default_z = float(run_config.target_altitude_m)
    resolved: list[tuple[float, float, float]] = []
    for wp in scenario.waypoints.waypoints:
        if not wp.has_local_xy():
            wp = terrain_service.waypoint_to_local(wp, scenario.map)
        x_m, y_m, _z_m = wp.local_xyz()
        if wp.z_m is not None:
            z_m = float(wp.z_m)
        elif wp.alt_m != 0.0 or default_z == 0.0:
            z_m = float(wp.alt_m)
        else:
            z_m = default_z
        resolved.append((float(x_m), float(y_m), z_m))

    if len(resolved) < 2:
        raise ValueError("PyBullet backend requires at least 2 waypoints")

    min_z = min(z for _, _, z in resolved)
    z_offset = max(_MIN_SPAWN_Z_M - min_z, 0.0)
    if z_offset > 0.0:
        resolved = [(x, y, z + z_offset) for x, y, z in resolved]
    return resolved, z_offset


def _yaw_toward(
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
    fallback_yaw: float,
) -> float:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback_yaw
    return math.atan2(dy, dx)


def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    try:
        return np.gradient(values, time_s, axis=0)
    except Exception:
        dt = float(np.nanmean(np.diff(time_s))) if len(time_s) > 1 else 1.0
        return np.gradient(values, axis=0) / max(dt, 1e-9)


class PyBulletQuadBackend(SimulationBackend):
    """PyFlyt Aviary adapter for waypoint-driven quadcopter simulation."""

    backend_id = "pybullet_quad"
    display_name = "PyBullet / PyFlyt Quadcopter"

    def __init__(self, terrain_service: TerrainService | None = None) -> None:
        self.terrain_service = terrain_service or TerrainService()
        self._pyflyt_available, self._pybullet_available, self._import_error = _detect_libs()
        if self._pyflyt_available:
            logger.info("PyBulletQuadBackend ready (PyFlyt=%s, pybullet=%s)", True, self._pybullet_available)
        elif self._pybullet_available:
            logger.info(
                "PyBulletQuadBackend: pybullet detected but PyFlyt missing -- install PyFlyt to run."
            )
        else:
            logger.debug("PyBulletQuadBackend libs not available: %s", self._import_error)

    # ------------------------------------------------------------------
    @property
    def libs_available(self) -> bool:
        return self._pyflyt_available or self._pybullet_available

    def availability_summary(self) -> dict[str, object]:
        return {
            "pyflyt": self._pyflyt_available,
            "pybullet": self._pybullet_available,
            "import_error": self._import_error,
        }

    # ------------------------------------------------------------------
    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        cfg = run_config or scenario.run_config
        if not self.libs_available:
            logger.warning(
                "PyBulletQuadBackend.run() invoked without supported libs "
                "(scenario=%s run_id=%s)",
                scenario.scenario_id,
                cfg.run_id,
            )
            raise BackendUnavailable(
                f"{self.display_name} requires PyFlyt or pybullet. {_INSTALL_HINT}"
            )
        if not self._pyflyt_available:
            raise BackendUnavailable(
                f"{self.display_name} requires PyFlyt for waypoint simulation. {_INSTALL_HINT}"
            )
        return self._run_with_pyflyt(scenario, cfg, on_step_progress=on_step_progress)

    def _run_with_pyflyt(
        self,
        scenario: ScenarioSpec,
        cfg: RunConfig,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        from PyFlyt.core import Aviary

        waypoints, z_offset = _resolve_waypoints_3d(scenario, cfg, self.terrain_service)
        x0, y0, z0 = waypoints[0]
        threshold = float(cfg.waypoint_threshold_m)
        max_steps = int(cfg.max_steps)
        dt_s = float(cfg.dt_s)
        steps_per_sample = max(1, round(dt_s * _CONTROL_HZ))

        logger.info(
            "Starting PyFlyt run scenario=%s run_id=%s waypoints=%d dt=%.4f max_steps=%d z_offset=%.3f",
            scenario.scenario_id,
            cfg.run_id,
            len(waypoints),
            dt_s,
            max_steps,
            z_offset,
        )

        t0 = time.perf_counter()
        env = Aviary(
            start_pos=np.array([[x0, y0, z0]], dtype=float),
            start_orn=np.array([[0.0, 0.0, 0.0]], dtype=float),
            render=False,
            drone_type="quadx",
            physics_hz=240,
        )
        env.set_mode(7)

        time_s: list[float] = []
        position_m: list[list[float]] = []
        velocity_mps: list[list[float]] = []
        attitude_rad: list[list[float]] = []
        angular_rate_rad_s: list[list[float]] = []
        controls: list[list[float]] = []
        reference_position_m: list[list[float]] = []

        wp_idx = 0
        final_wp = waypoints[-1]
        status = "completed_with_miss"
        success = False

        try:
            for step in range(max_steps):
                state = env.state(0)
                pos = np.asarray(state[3, :], dtype=float)
                if wp_idx < len(waypoints) - 1:
                    target_xy = waypoints[wp_idx][:2]
                    dist_xy = math.hypot(pos[0] - target_xy[0], pos[1] - target_xy[1])
                    if dist_xy < threshold:
                        wp_idx += 1

                target = waypoints[wp_idx]
                if wp_idx < len(waypoints) - 1:
                    yaw = _yaw_toward(target[:2], waypoints[wp_idx + 1][:2], float(state[1, 2]))
                else:
                    yaw = float(state[1, 2])

                setpoint = np.array([target[0], target[1], yaw, target[2]], dtype=float)
                env.set_setpoint(0, setpoint)
                for _ in range(steps_per_sample):
                    env.step()

                state = env.state(0)
                pos = np.asarray(state[3, :], dtype=float)
                vel = np.asarray(state[2, :], dtype=float)
                att = np.asarray(state[1, :], dtype=float)
                att_rate = np.asarray(state[0, :], dtype=float)

                sample_t = (step + 1) * dt_s
                time_s.append(sample_t)
                position_m.append(pos.tolist())
                velocity_mps.append(vel.tolist())
                attitude_rad.append(att.tolist())
                angular_rate_rad_s.append(att_rate.tolist())
                controls.append(setpoint.tolist())
                reference_position_m.append([target[0], target[1], target[2]])

                if on_step_progress is not None:
                    try:
                        on_step_progress(step + 1, max_steps)
                    except Exception:
                        pass

                miss = math.hypot(pos[0] - final_wp[0], pos[1] - final_wp[1])
                miss_z = abs(pos[2] - final_wp[2])
                if wp_idx >= len(waypoints) - 1 and miss < threshold and miss_z < threshold:
                    success = True
                    status = "success"
                    break
        finally:
            env.disconnect()

        wallclock_s = time.perf_counter() - t0
        time_arr = np.asarray(time_s, dtype=float)
        pos_arr = np.asarray(position_m, dtype=float)
        vel_arr = np.asarray(velocity_mps, dtype=float)
        acc_arr = _gradient(vel_arr, time_arr) if len(vel_arr) else np.zeros((0, 3))
        ref_arr = np.asarray(reference_position_m, dtype=float)
        tracking_error = (
            np.linalg.norm(pos_arr[:, :3] - ref_arr[:, :3], axis=1) if len(pos_arr) else np.array([])
        )

        final_pos = pos_arr[-1] if len(pos_arr) else np.array([x0, y0, z0])
        miss_distance = float(
            np.linalg.norm(final_pos[:3] - np.asarray(final_wp, dtype=float))
        )

        summary = RunSummary(
            success=success,
            miss_distance_m=miss_distance,
            settle_steps=len(time_s),
            duration_s=float(time_arr[-1]) if len(time_arr) else 0.0,
            max_tracking_error_m=float(np.nanmax(tracking_error)) if len(tracking_error) else None,
            mean_tracking_error_m=float(np.nanmean(tracking_error)) if len(tracking_error) else None,
            max_altitude_m=float(np.nanmax(pos_arr[:, 2])) if len(pos_arr) else None,
            min_altitude_m=float(np.nanmin(pos_arr[:, 2])) if len(pos_arr) else None,
            wallclock_s=wallclock_s,
        )

        logger.info(
            "Completed %s run run_id=%s status=%s success=%s miss=%.3fm duration=%.2fs wallclock=%.3fs",
            self.backend_id,
            cfg.run_id,
            status,
            success,
            miss_distance,
            summary.duration_s,
            wallclock_s,
        )

        metadata: dict[str, object] = {
            "backend_display_name": self.display_name,
            "source": "PyFlyt.core.Aviary",
            "pyflyt_backend": True,
            "pyflyt_z_offset_m": z_offset,
            "waypoints_local_xyz": [[x, y, z] for x, y, z in waypoints],
            "control_hz": _CONTROL_HZ,
            "steps_per_sample": steps_per_sample,
            "fidelity_path": "pyflyt_quadx_mode7",
        }
        if z_offset > 0.0:
            metadata["note"] = (
                "PyFlyt z coordinates include a uniform offset above the flat ground plane."
            )

        return RunResult(
            run_id=cfg.run_id,
            scenario_id=scenario.scenario_id,
            backend_id=self.backend_id,
            model_id=scenario.vehicle.model_id,
            status=status,
            time_s=time_s,
            position_m=position_m,
            velocity_mps=velocity_mps,
            acceleration_mps2=acc_arr.tolist(),
            attitude_rad=attitude_rad,
            angular_rate_rad_s=angular_rate_rad_s,
            controls=controls,
            reference_position_m=reference_position_m,
            tracking_error_m=tracking_error.tolist(),
            summary=summary,
            metadata=metadata,
        )


def factory_default() -> PyBulletQuadBackend:
    """Constructor wrapper kept for symmetry with other backend modules."""
    return PyBulletQuadBackend()


__all__: list[str] = ["PyBulletQuadBackend", "factory_default"]

LogFn = Callable[[str], None]
