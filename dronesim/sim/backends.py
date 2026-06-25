"""Simulation backend interfaces and adapters."""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable

StepProgressCb = Callable[[int, int], None]

import numpy as np

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.services.terrain import MapCacheMiss, TerrainService
from dronesim.sim.debug_log import get_sim_logger

logger = get_sim_logger(__name__)


class SimulationBackend(ABC):
    """Common interface every vehicle simulator must implement."""

    backend_id: str
    display_name: str

    @abstractmethod
    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        """Run one scenario and return a normalized result."""


class BackendUnavailable(RuntimeError):
    """Raised when an optional backend dependency is not installed."""


class PlaceholderBackend(SimulationBackend):
    """Explicit placeholder for planned high-fidelity backends."""

    def __init__(self, backend_id: str, display_name: str, install_hint: str) -> None:
        self.backend_id = backend_id
        self.display_name = display_name
        self.install_hint = install_hint

    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        del on_step_progress
        cfg = run_config or scenario.run_config
        logger.warning(
            "Placeholder backend invoked: backend_id=%s scenario_id=%s run_id=%s",
            self.backend_id,
            scenario.scenario_id,
            cfg.run_id,
        )
        raise BackendUnavailable(
            f"{self.display_name} is planned but not wired yet. {self.install_hint}"
        )


class InHouseMpcQuadBackend(SimulationBackend):
    """Adapter around simulations/6DOF_Quadcopter_MPC-main/drone_mc."""

    backend_id = "inhouse_mpc_quad"
    display_name = "In-house MPC Quadcopter"

    def __init__(self, terrain_service: TerrainService | None = None) -> None:
        self.terrain_service = terrain_service or TerrainService()
        self._ensure_vendor_path()
        logger.debug("Initialized %s backend", self.backend_id)

    @staticmethod
    def _ensure_vendor_path() -> None:
        root = Path(__file__).resolve().parents[2]
        vendor_root = root / "simulations" / "6DOF_Quadcopter_MPC-main"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
            logger.debug("Prepended vendor path: %s", vendor_root)
        else:
            logger.debug("Vendor path already present: %s", vendor_root)

    _PERTURBATION_FIELDS = (
        "init_pos_std",
        "init_vel_std",
        "init_att_std",
        "mass_jitter_pct",
        "inertia_jitter_pct",
    )

    def _build_sim_config(self, scenario: ScenarioSpec, run_config: RunConfig):
        from drone_mc.config import SimConfig

        scenario.validate()
        local_waypoints = []
        for wp in scenario.waypoints.waypoints:
            if not wp.has_local_xy():
                wp = self.terrain_service.waypoint_to_local(wp, scenario.map)
            x_m, y_m, _z_m = wp.local_xyz()
            local_waypoints.append([x_m, y_m])

        vehicle_params = scenario.vehicle.parameters
        mc = run_config.monte_carlo
        trial_index = int(mc.get("trial_index", 0))
        first_x, first_y = local_waypoints[0]
        logger.debug(
            "Built SimConfig for scenario=%s run_id=%s trial=%s waypoints=%d "
            "dt=%.4f max_steps=%d seed=%s",
            scenario.scenario_id,
            run_config.run_id,
            trial_index,
            len(local_waypoints),
            float(run_config.dt_s),
            int(run_config.max_steps),
            run_config.seed,
        )
        logger.debug(
            "Vehicle params mass=%.3f Ix=%.3f Iy=%.3f Iz=%.3f | MC knobs=%s",
            float(vehicle_params.get("mass", 5.0)),
            float(vehicle_params.get("Ix", 1.0)),
            float(vehicle_params.get("Iy", 1.0)),
            float(vehicle_params.get("Iz", 1.5)),
            {key: mc.get(key, 0.0) for key in self._PERTURBATION_FIELDS},
        )
        return SimConfig(
            waypoints=np.asarray(local_waypoints, dtype=float),
            altitude=float(run_config.target_altitude_m),
            dt=float(run_config.dt_s),
            horizon=int(run_config.horizon),
            max_steps=int(run_config.max_steps),
            waypt_thresh=float(run_config.waypoint_threshold_m),
            lookahead=int(run_config.lookahead),
            mass=float(vehicle_params.get("mass", 5.0)),
            Ix=float(vehicle_params.get("Ix", 1.0)),
            Iy=float(vehicle_params.get("Iy", 1.0)),
            Iz=float(vehicle_params.get("Iz", 1.5)),
            init_state={
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll_dot": 0.0,
                "pitch_dot": 0.0,
                "yaw_dot": 0.0,
                "x_dot": 0.0,
                "y_dot": 0.0,
                "z_dot": 0.0,
                "x": float(first_x),
                "y": float(first_y),
                "z": float(run_config.target_altitude_m),
            },
            init_pos_std=float(mc.get("init_pos_std", 0.0)),
            init_vel_std=float(mc.get("init_vel_std", 0.0)),
            init_att_std=float(mc.get("init_att_std", 0.0)),
            force_noise_std=float(mc.get("force_noise_std", 0.0)),
            mass_jitter_pct=float(mc.get("mass_jitter_pct", 0.0)),
            inertia_jitter_pct=float(mc.get("inertia_jitter_pct", 0.0)),
            seed=run_config.seed,
            trial_index=trial_index,
        )

    @classmethod
    def _needs_perturbation(cls, sim_cfg) -> bool:
        return any(float(getattr(sim_cfg, field, 0.0)) > 0.0 for field in cls._PERTURBATION_FIELDS)

    @staticmethod
    def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        if len(values) < 2:
            return np.zeros_like(values)
        try:
            return np.gradient(values, time_s, axis=0)
        except Exception:
            dt = float(np.nanmean(np.diff(time_s))) if len(time_s) > 1 else 1.0
            return np.gradient(values, axis=0) / max(dt, 1e-9)

    @staticmethod
    def _wants_extended(scenario: ScenarioSpec, run_config: RunConfig) -> bool:
        """Auto-detect whether Phase 6 fidelity is requested for this run.

        Returns True if any of the user-facing knobs has been moved off the
        legacy default. ``fidelity_mode`` of ``legacy`` / ``extended`` forces
        the decision regardless of the other knobs.
        """
        mode = (getattr(run_config, "fidelity_mode", "auto") or "auto").lower()
        if mode == "legacy":
            return False
        if mode == "extended":
            return True
        if getattr(run_config, "integration_method", "euler").lower() == "rk4":
            return True
        env = scenario.environment
        if any(float(v) != 0.0 for v in (env.wind_mps or [0.0, 0.0, 0.0])):
            return True
        if getattr(env, "gust_std_mps", 0.0) and float(env.gust_std_mps) > 0.0:
            return True
        if getattr(env, "terrain_collision_enabled", False):
            return True
        aero = scenario.vehicle.parameters.get("aero") or {}
        if float(aero.get("cd_linear", 0.0)) > 0.0:
            return True
        if float(aero.get("cd_quadratic", 0.0)) > 0.0:
            return True
        return False

    def _resolve_terrain_query(self, scenario: ScenarioSpec):
        """Return ``(query, offset_m)`` for terrain collision or ``(None, _)``."""
        env = scenario.environment
        if not getattr(env, "terrain_collision_enabled", False):
            return None, float(getattr(env, "terrain_collision_offset_m", 0.5))
        try:
            asset = self.terrain_service.fetch_map(scenario.map, fetch_remote=False)
        except MapCacheMiss as exc:
            logger.warning(
                "Terrain collision requested but no cached map for scenario=%s: %s",
                scenario.scenario_id,
                exc,
            )
            return None, float(getattr(env, "terrain_collision_offset_m", 0.5))
        except Exception as exc:  # noqa: BLE001 - never fail the run for a terrain miss
            logger.warning(
                "Failed to load terrain for collision check (scenario=%s): %s",
                scenario.scenario_id,
                exc,
            )
            return None, float(getattr(env, "terrain_collision_offset_m", 0.5))
        return asset.elevation_at, float(getattr(env, "terrain_collision_offset_m", 0.5))

    def _build_dynamics_factory(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig,
        terrain_query,
        terrain_offset_m: float,
        rng: np.random.Generator,
    ):
        """Construct a ``dynamics_factory(sim_cfg) -> ExtendedQuadcopter``."""
        from dronesim.sim.fidelity import (
            AeroParams,
            TerrainCollision,
            WindField,
            build_extended_quad_from_sim_cfg,
        )

        env = scenario.environment
        aero = AeroParams.from_vehicle_params(
            scenario.vehicle.parameters,
            air_density_kg_m3=float(getattr(env, "air_density_kg_m3", 1.225)),
        )
        wind = WindField.from_environment(
            wind_mps=env.wind_mps,
            gust_std_mps=float(getattr(env, "gust_std_mps", 0.0)),
            gust_decorrelation_s=float(getattr(env, "gust_decorrelation_s", 2.0)),
            rng=rng,
        )
        collision = TerrainCollision(query=terrain_query, offset_m=terrain_offset_m)
        integration_method = getattr(run_config, "integration_method", "euler") or "euler"

        def factory(sim_cfg):
            return build_extended_quad_from_sim_cfg(
                sim_cfg,
                aero=aero,
                wind=wind,
                integration_method=integration_method,
                collision=collision,
            )

        return factory

    def run(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        cfg = run_config or scenario.run_config
        logger.info(
            "Starting %s run scenario=%s run_id=%s backend=%s",
            self.display_name,
            scenario.scenario_id,
            cfg.run_id,
            self.backend_id,
        )
        sim_cfg = self._build_sim_config(scenario, cfg)
        if self._needs_perturbation(sim_cfg):
            seed = sim_cfg.seed if sim_cfg.seed is not None else sim_cfg.trial_index
            logger.debug("Applying Monte Carlo perturbation with seed=%s trial=%s", seed, sim_cfg.trial_index)
            sim_cfg = sim_cfg.sample(np.random.default_rng(seed))
        else:
            logger.debug("No perturbation requested for run_id=%s", cfg.run_id)

        extended = self._wants_extended(scenario, cfg)
        terrain_query = None
        terrain_offset_m = float(getattr(scenario.environment, "terrain_collision_offset_m", 0.5))
        path_label = "vendor"
        t0 = time.perf_counter()
        if extended:
            terrain_query, terrain_offset_m = self._resolve_terrain_query(scenario)
            wind_seed = sim_cfg.seed if sim_cfg.seed is not None else sim_cfg.trial_index
            wind_rng = np.random.default_rng((wind_seed or 0) + 0xC0FFEE)
            dynamics_factory = self._build_dynamics_factory(
                scenario, cfg, terrain_query, terrain_offset_m, wind_rng
            )
            from dronesim.sim.inhouse_quad_runtime import run_simulation_local

            sim_result = run_simulation_local(
                sim_cfg,
                dynamics_factory=dynamics_factory,
                terrain_query=terrain_query,
                terrain_offset_m=terrain_offset_m,
                on_step_progress=on_step_progress,
            )
            path_label = "extended"
        else:
            from drone_mc.simulator import run_simulation

            sim_result = run_simulation(sim_cfg, on_step_progress=on_step_progress)
        logger.debug(
            "%s simulation finished in %.3fs success=%s steps=%d path=%s",
            self.display_name,
            time.perf_counter() - t0,
            bool(sim_result.success),
            len(sim_result.time),
            path_label,
        )

        time_s = np.asarray(sim_result.time, dtype=float)
        pos = np.asarray(sim_result.pos, dtype=float)
        vel = np.asarray(sim_result.vel, dtype=float)
        acc = self._gradient(vel, time_s) if len(vel) else np.zeros((0, 3))
        att = np.asarray(sim_result.att, dtype=float)
        att_rate = np.asarray(sim_result.att_rate, dtype=float)
        controls = np.asarray(sim_result.u, dtype=float)
        ref = np.column_stack([
            np.asarray(sim_result.ref_xy, dtype=float),
            np.full(len(sim_result.ref_xy), float(cfg.target_altitude_m)),
        ])
        tracking_error = np.linalg.norm(pos[:, :3] - ref[:, :3], axis=1) if len(pos) else np.array([])

        summary = RunSummary(
            success=bool(sim_result.success),
            miss_distance_m=float(sim_result.miss_distance),
            settle_steps=int(sim_result.settle_steps),
            duration_s=float(time_s[-1]) if len(time_s) else 0.0,
            max_tracking_error_m=float(np.nanmax(tracking_error)) if len(tracking_error) else None,
            mean_tracking_error_m=float(np.nanmean(tracking_error)) if len(tracking_error) else None,
            max_altitude_m=float(np.nanmax(pos[:, 2])) if len(pos) else None,
            min_altitude_m=float(np.nanmin(pos[:, 2])) if len(pos) else None,
            wallclock_s=float(sim_result.wallclock_s),
        )

        terminated_by = (sim_result.cfg_summary or {}).get("terminated_by")
        if terminated_by == "terrain_collision":
            status = "terrain_collision"
        elif sim_result.success:
            status = "success"
        else:
            status = "completed_with_miss"

        logger.info(
            "Completed %s run run_id=%s status=%s success=%s miss=%.3fm duration=%.2fs wallclock=%.3fs",
            self.backend_id,
            cfg.run_id,
            status,
            bool(sim_result.success),
            float(sim_result.miss_distance),
            summary.duration_s,
            summary.wallclock_s,
        )
        logger.debug(
            "Run metrics run_id=%s max_track_err=%s mean_track_err=%s alt=[%s, %s] settle_steps=%s",
            cfg.run_id,
            summary.max_tracking_error_m,
            summary.mean_tracking_error_m,
            summary.min_altitude_m,
            summary.max_altitude_m,
            summary.settle_steps,
        )

        metadata = {
            "backend_display_name": self.display_name,
            "source": "simulations/6DOF_Quadcopter_MPC-main/drone_mc",
            "cfg_summary": sim_result.cfg_summary,
            "spline_points": np.asarray(sim_result.spline).tolist(),
            "waypoints_local_xy": np.asarray(sim_result.waypoints).tolist(),
            "fidelity_path": path_label,
            "integration_method": getattr(cfg, "integration_method", "euler"),
        }
        if terminated_by is not None:
            metadata["terminated_by"] = terminated_by

        return RunResult(
            run_id=cfg.run_id,
            scenario_id=scenario.scenario_id,
            backend_id=self.backend_id,
            model_id=scenario.vehicle.model_id,
            status=status,
            time_s=time_s.tolist(),
            position_m=pos.tolist(),
            velocity_mps=vel.tolist(),
            acceleration_mps2=acc.tolist(),
            attitude_rad=att.tolist(),
            angular_rate_rad_s=att_rate.tolist(),
            controls=controls.tolist(),
            reference_position_m=ref.tolist(),
            tracking_error_m=tracking_error.tolist(),
            summary=summary,
            metadata=metadata,
        )


class DroneFactory:
    """Registry/factory for simulation backends."""

    def __init__(self, backends: Iterable[SimulationBackend] | None = None) -> None:
        self._backends: dict[str, SimulationBackend] = {}
        if backends is None:
            # Import locally to avoid an import cycle with backends_pybullet
            # and backends_jsbsim (which import SimulationBackend /
            # BackendUnavailable from here).
            from dronesim.sim.backends_jsbsim import JSBSimCessnaBackend
            from dronesim.sim.backends_pointmass import (
                PointMass3DOFFixedWingBackend,
                PointMass3DOFQuadBackend,
            )
            from dronesim.sim.backends_pybullet import PyBulletQuadBackend

            backends = [
                InHouseMpcQuadBackend(),
                PyBulletQuadBackend(),
                JSBSimCessnaBackend(),
                PointMass3DOFQuadBackend(),
                PointMass3DOFFixedWingBackend(),
            ]
        for backend in backends:
            self.register(backend)
        logger.debug("DroneFactory initialized with %d backend(s)", len(self._backends))

    def register(self, backend: SimulationBackend) -> None:
        self._backends[backend.backend_id] = backend
        logger.debug("Registered backend %s (%s)", backend.backend_id, backend.display_name)

    def get(self, backend_id: str) -> SimulationBackend:
        try:
            backend = self._backends[backend_id]
            logger.debug("Resolved backend %s", backend_id)
            return backend
        except KeyError as exc:
            logger.error("Unknown simulation backend requested: %s", backend_id)
            raise KeyError(f"Unknown simulation backend: {backend_id}") from exc

    def available(self) -> list[dict[str, str]]:
        return [
            {"backend_id": backend.backend_id, "display_name": backend.display_name}
            for backend in self._backends.values()
        ]


class SimulationManager:
    """Small orchestration layer between GUI/scenario services and backends."""

    def __init__(self, factory: DroneFactory | None = None) -> None:
        self.factory = factory or DroneFactory()

    def run(self, scenario: ScenarioSpec, run_config: RunConfig | None = None) -> RunResult:
        cfg = run_config or scenario.run_config
        logger.info(
            "SimulationManager dispatch scenario=%s run_id=%s backend=%s",
            scenario.scenario_id,
            cfg.run_id,
            cfg.backend_id,
        )
        backend = self.factory.get(cfg.backend_id)
        return backend.run(scenario, cfg)
