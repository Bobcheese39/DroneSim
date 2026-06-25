"""3DOF point-mass simulation backends.

Provides two :class:`~dronesim.sim.backends.SimulationBackend` implementations:

* :class:`PointMass3DOFQuadBackend` (``backend_id="pointmass_quad"``) –
  Cartesian PD point-mass for quadrotors.
* :class:`PointMass3DOFFixedWingBackend` (``backend_id="pointmass_fixed_wing"``) –
  Speed-gamma-track point-mass for fixed-wing aircraft.

Both backends normalise their output to :class:`~dronesim.models.RunResult`
using the same field mapping as :class:`InHouseMpcQuadBackend` so downstream
visualisation and analysis code is backend-agnostic.

Monte Carlo support
-------------------
Both backends honour ``run_config.monte_carlo`` / ``run_config.seed`` for
initial position and velocity perturbation, matching the naming conventions
used by the in-house 6DOF quad backend.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.sim.backends import SimulationBackend, StepProgressCb
from dronesim.sim.debug_log import get_sim_logger
from dronesim.sim.fidelity import TerrainCollision, WindField
from dronesim.sim.pointmass import (
    FixedWingPointMassParams,
    PointMassResult,
    QuadPointMassParams,
    run_fixedwing_pointmass,
    run_quad_pointmass,
)
from dronesim.services.terrain import MapCacheMiss, TerrainService

logger = get_sim_logger(__name__)


def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    try:
        return np.gradient(values, time_s, axis=0)
    except Exception:
        dt = float(np.nanmean(np.diff(time_s))) if len(time_s) > 1 else 1.0
        return np.gradient(values, axis=0) / max(dt, 1e-9)


def _resolve_waypoints(
    scenario: ScenarioSpec,
    terrain_service: TerrainService,
    default_alt_m: float = 5.0,
) -> np.ndarray:
    """Return an (M, 3) local-ENU array of waypoint x/y/z coords.

    Per-waypoint z priority: ``z_m`` (if set) → ``alt_m`` (if non-zero) →
    ``default_alt_m`` (the run's ``target_altitude_m``).
    """
    local_waypoints: list[list[float]] = []
    for wp in scenario.waypoints.waypoints:
        if not wp.has_local_xy():
            wp = terrain_service.waypoint_to_local(wp, scenario.map)
        x_m, y_m, _ = wp.local_xyz()
        if wp.z_m is not None:
            z_m = float(wp.z_m)
        elif wp.alt_m != 0.0:
            z_m = float(wp.alt_m)
        else:
            z_m = default_alt_m
        local_waypoints.append([x_m, y_m, z_m])
    return np.asarray(local_waypoints, dtype=float)


def _build_wind(scenario: ScenarioSpec, rng: np.random.Generator) -> WindField:
    env = scenario.environment
    return WindField.from_environment(
        wind_mps=env.wind_mps,
        gust_std_mps=float(getattr(env, "gust_std_mps", 0.0)),
        gust_decorrelation_s=float(getattr(env, "gust_decorrelation_s", 2.0)),
        rng=rng,
    )


def _resolve_terrain_collision(
    scenario: ScenarioSpec,
    terrain_service: TerrainService,
) -> TerrainCollision:
    env = scenario.environment
    offset_m = float(getattr(env, "terrain_collision_offset_m", 0.5))
    if not getattr(env, "terrain_collision_enabled", False):
        return TerrainCollision(query=None, offset_m=offset_m)
    try:
        asset = terrain_service.fetch_map(scenario.map, fetch_remote=False)
        return TerrainCollision(query=asset.elevation_at, offset_m=offset_m)
    except MapCacheMiss as exc:
        logger.warning("Terrain collision requested but no cached map: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load terrain for collision check: %s", exc)
    return TerrainCollision(query=None, offset_m=offset_m)


def _perturb_init_pos(
    pos: np.ndarray,
    rng: np.random.Generator,
    init_pos_std: float,
) -> np.ndarray:
    if init_pos_std <= 0.0:
        return pos
    return pos + rng.normal(0.0, init_pos_std, size=3)


def _perturb_init_vel(
    vel: np.ndarray,
    rng: np.random.Generator,
    init_vel_std: float,
) -> np.ndarray:
    if init_vel_std <= 0.0:
        return vel
    return vel + rng.normal(0.0, init_vel_std, size=3)


def _pointmass_result_to_run_result(
    pm: PointMassResult,
    *,
    scenario: ScenarioSpec,
    cfg: RunConfig,
    backend_id: str,
    model_id: str,
    display_name: str,
) -> RunResult:
    """Convert a PointMassResult into the normalised RunResult contract."""
    time_s = pm.time_s
    pos = pm.position_m
    vel = pm.velocity_mps
    acc = pm.acceleration_mps2
    if acc is None or len(acc) == 0:
        acc = _gradient(vel, time_s) if len(vel) > 1 else np.zeros((len(pos), 3))
    att = pm.attitude_rad
    att_rate = pm.angular_rate_rad_s
    controls = pm.controls
    ref = pm.reference_position_m
    tracking_error = pm.tracking_error_m

    if len(pos) and len(ref):
        tracking_error_arr = np.linalg.norm(pos - ref, axis=1)
    else:
        tracking_error_arr = np.array([])

    if pm.terminated_by == "terrain_collision":
        status = "terrain_collision"
    elif pm.success:
        status = "success"
    else:
        status = "completed_with_miss"

    summary = RunSummary(
        success=pm.success,
        miss_distance_m=pm.miss_distance_m,
        settle_steps=pm.settle_steps,
        duration_s=float(time_s[-1]) if len(time_s) else 0.0,
        max_tracking_error_m=float(np.nanmax(tracking_error_arr)) if len(tracking_error_arr) else None,
        mean_tracking_error_m=float(np.nanmean(tracking_error_arr)) if len(tracking_error_arr) else None,
        max_altitude_m=float(np.nanmax(pos[:, 2])) if len(pos) else None,
        min_altitude_m=float(np.nanmin(pos[:, 2])) if len(pos) else None,
        wallclock_s=pm.wallclock_s,
    )

    metadata: dict[str, Any] = {
        "backend_display_name": display_name,
        "source": "dronesim.sim.pointmass",
        "model": pm.metadata.get("model", "unknown"),
        "fidelity": "3dof_pointmass",
    }
    if pm.terminated_by:
        metadata["terminated_by"] = pm.terminated_by

    return RunResult(
        run_id=cfg.run_id,
        scenario_id=scenario.scenario_id,
        backend_id=backend_id,
        model_id=model_id,
        status=status,
        time_s=time_s.tolist(),
        position_m=pos.tolist(),
        velocity_mps=vel.tolist(),
        acceleration_mps2=acc.tolist(),
        attitude_rad=att.tolist(),
        angular_rate_rad_s=att_rate.tolist(),
        controls=controls.tolist(),
        reference_position_m=ref.tolist(),
        tracking_error_m=tracking_error_arr.tolist(),
        summary=summary,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Quadcopter backend
# ---------------------------------------------------------------------------


class PointMass3DOFQuadBackend(SimulationBackend):
    """3DOF point-mass quadcopter backend.

    Implements the same ``SimulationBackend`` contract as the in-house MPC
    quad but uses a kinematic PD plant that is unconditionally stable for any
    positive ``kp_pos`` / ``kd_pos`` / ``max_accel_mps2`` settings.
    """

    backend_id = "pointmass_quad"
    display_name = "3DOF Point-Mass Quadcopter"

    def __init__(self, terrain_service: TerrainService | None = None) -> None:
        self.terrain_service = terrain_service or TerrainService()
        logger.debug("Initialized %s backend", self.backend_id)

    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        cfg = run_config or scenario.run_config
        logger.info(
            "Starting %s run scenario=%s run_id=%s",
            self.display_name,
            scenario.scenario_id,
            cfg.run_id,
        )
        t0 = time.perf_counter()

        # Monte Carlo seed
        mc = cfg.monte_carlo
        seed = cfg.seed
        if seed is None:
            seed = int(mc.get("trial_index", 0))
        rng = np.random.default_rng(seed)
        wind_rng = np.random.default_rng((seed or 0) + 0xC0FFEE)

        waypoints_xyz = _resolve_waypoints(
            scenario, self.terrain_service, default_alt_m=float(cfg.target_altitude_m)
        )
        params = QuadPointMassParams.from_dict(
            scenario.vehicle.parameters,
            target_altitude_m=float(cfg.target_altitude_m),
        )

        # Initial position with optional perturbation — z from first waypoint
        init_pos = np.array([waypoints_xyz[0, 0], waypoints_xyz[0, 1], waypoints_xyz[0, 2]])
        init_vel = np.zeros(3)
        init_pos_std = float(mc.get("init_pos_std", 0.0))
        init_vel_std = float(mc.get("init_vel_std", 0.0))
        init_pos = _perturb_init_pos(init_pos, rng, init_pos_std)
        init_vel = _perturb_init_vel(init_vel, rng, init_vel_std)

        wind = _build_wind(scenario, wind_rng)
        collision = _resolve_terrain_collision(scenario, self.terrain_service)

        logger.debug(
            "Quad 3DOF: waypoints=%d dt=%.3f max_steps=%d kp=%.3f kd=%.3f a_max=%.2f",
            len(waypoints_xyz),
            float(cfg.dt_s),
            int(cfg.max_steps),
            params.kp_pos,
            params.kd_pos,
            params.max_accel_mps2,
        )

        pm_result = run_quad_pointmass(
            waypoints_xy=waypoints_xyz,
            params=params,
            dt=float(cfg.dt_s),
            max_steps=int(cfg.max_steps),
            wind=wind,
            collision=collision,
            init_pos=init_pos,
            init_vel=init_vel,
            on_step_progress=on_step_progress,
        )

        logger.info(
            "Completed %s run_id=%s status=%s success=%s miss=%.3fm wallclock=%.3fs",
            self.backend_id,
            cfg.run_id,
            "success" if pm_result.success else "miss",
            pm_result.success,
            pm_result.miss_distance_m,
            time.perf_counter() - t0,
        )

        return _pointmass_result_to_run_result(
            pm_result,
            scenario=scenario,
            cfg=cfg,
            backend_id=self.backend_id,
            model_id=scenario.vehicle.model_id,
            display_name=self.display_name,
        )


# ---------------------------------------------------------------------------
# Fixed-wing backend
# ---------------------------------------------------------------------------


class PointMass3DOFFixedWingBackend(SimulationBackend):
    """3DOF point-mass fixed-wing backend.

    Provides a smooth, always-stable alternative to the JSBSim backend for
    preliminary route analysis and parameter tuning.  Speed is held at cruise;
    bank and flight-path angle are guidance-commanded and rate-limited.
    """

    backend_id = "pointmass_fixed_wing"
    display_name = "3DOF Point-Mass Fixed-Wing"

    def __init__(self, terrain_service: TerrainService | None = None) -> None:
        self.terrain_service = terrain_service or TerrainService()
        logger.debug("Initialized %s backend", self.backend_id)

    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        cfg = run_config or scenario.run_config
        logger.info(
            "Starting %s run scenario=%s run_id=%s",
            self.display_name,
            scenario.scenario_id,
            cfg.run_id,
        )
        t0 = time.perf_counter()

        mc = cfg.monte_carlo
        seed = cfg.seed
        if seed is None:
            seed = int(mc.get("trial_index", 0))
        rng = np.random.default_rng(seed)
        wind_rng = np.random.default_rng((seed or 0) + 0xC0FFEE)

        waypoints_xyz = _resolve_waypoints(
            scenario, self.terrain_service, default_alt_m=float(cfg.target_altitude_m)
        )

        # Fixed-wing params can live in vehicle.parameters or vehicle.controller
        vehicle_params: dict = dict(scenario.vehicle.parameters)
        controller: dict = dict(scenario.vehicle.controller or {})
        merged_params = {**controller, **vehicle_params}

        params = FixedWingPointMassParams.from_dict(
            merged_params,
            target_altitude_m=float(cfg.target_altitude_m),
        )

        # z from first resolved waypoint
        init_pos = np.array([waypoints_xyz[0, 0], waypoints_xyz[0, 1], waypoints_xyz[0, 2]])
        init_pos_std = float(mc.get("init_pos_std", 0.0))
        init_pos = _perturb_init_pos(init_pos, rng, init_pos_std)

        wind = _build_wind(scenario, wind_rng)
        collision = _resolve_terrain_collision(scenario, self.terrain_service)

        logger.debug(
            "FixedWing 3DOF: waypoints=%d dt=%.3f max_steps=%d V=%.1f bank_max=%.1fdeg",
            len(waypoints_xyz),
            float(cfg.dt_s),
            int(cfg.max_steps),
            params.cruise_speed_mps,
            params.max_bank_deg,
        )

        pm_result = run_fixedwing_pointmass(
            waypoints_xy=waypoints_xyz,
            params=params,
            dt=float(cfg.dt_s),
            max_steps=int(cfg.max_steps),
            wind=wind,
            collision=collision,
            init_pos=init_pos,
            on_step_progress=on_step_progress,
        )

        logger.info(
            "Completed %s run_id=%s status=%s success=%s miss=%.3fm wallclock=%.3fs",
            self.backend_id,
            cfg.run_id,
            "success" if pm_result.success else "miss",
            pm_result.success,
            pm_result.miss_distance_m,
            time.perf_counter() - t0,
        )

        return _pointmass_result_to_run_result(
            pm_result,
            scenario=scenario,
            cfg=cfg,
            backend_id=self.backend_id,
            model_id=scenario.vehicle.model_id,
            display_name=self.display_name,
        )


__all__ = [
    "PointMass3DOFFixedWingBackend",
    "PointMass3DOFQuadBackend",
]
