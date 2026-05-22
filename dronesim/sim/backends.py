"""Simulation backend interfaces and adapters."""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.services.terrain import TerrainService


class SimulationBackend(ABC):
    """Common interface every vehicle simulator must implement."""

    backend_id: str
    display_name: str

    @abstractmethod
    def run(self, scenario: ScenarioSpec, run_config: RunConfig | None = None) -> RunResult:
        """Run one scenario and return a normalized result."""


class BackendUnavailable(RuntimeError):
    """Raised when an optional backend dependency is not installed."""


class PlaceholderBackend(SimulationBackend):
    """Explicit placeholder for planned high-fidelity backends."""

    def __init__(self, backend_id: str, display_name: str, install_hint: str) -> None:
        self.backend_id = backend_id
        self.display_name = display_name
        self.install_hint = install_hint

    def run(self, scenario: ScenarioSpec, run_config: RunConfig | None = None) -> RunResult:
        del scenario, run_config
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

    @staticmethod
    def _ensure_vendor_path() -> None:
        root = Path(__file__).resolve().parents[2]
        vendor_root = root / "simulations" / "6DOF_Quadcopter_MPC-main"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))

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
        first_x, first_y = local_waypoints[0]
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
            trial_index=0,
        )

    @staticmethod
    def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        if len(values) < 2:
            return np.zeros_like(values)
        try:
            return np.gradient(values, time_s, axis=0)
        except Exception:
            dt = float(np.nanmean(np.diff(time_s))) if len(time_s) > 1 else 1.0
            return np.gradient(values, axis=0) / max(dt, 1e-9)

    def run(self, scenario: ScenarioSpec, run_config: RunConfig | None = None) -> RunResult:
        from drone_mc.simulator import run_simulation

        cfg = run_config or scenario.run_config
        sim_cfg = self._build_sim_config(scenario, cfg)
        sim_result = run_simulation(sim_cfg)

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

        return RunResult(
            run_id=cfg.run_id,
            scenario_id=scenario.scenario_id,
            backend_id=self.backend_id,
            model_id=scenario.vehicle.model_id,
            status="success" if sim_result.success else "completed_with_miss",
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
            metadata={
                "backend_display_name": self.display_name,
                "source": "simulations/6DOF_Quadcopter_MPC-main/drone_mc",
                "cfg_summary": sim_result.cfg_summary,
                "spline_points": np.asarray(sim_result.spline).tolist(),
                "waypoints_local_xy": np.asarray(sim_result.waypoints).tolist(),
            },
        )


class DroneFactory:
    """Registry/factory for simulation backends."""

    def __init__(self, backends: Iterable[SimulationBackend] | None = None) -> None:
        self._backends: dict[str, SimulationBackend] = {}
        if backends is None:
            backends = [
                InHouseMpcQuadBackend(),
                PlaceholderBackend(
                    "pybullet_quad",
                    "PyBullet/PyFlyt Quadcopter",
                    "Install and configure gym-pybullet-drones or PyFlyt before enabling this backend.",
                ),
                PlaceholderBackend(
                    "jsbsim_cessna",
                    "JSBSim Cessna",
                    "Install jsbsim and wire aircraft scripts/properties before enabling this backend.",
                ),
            ]
        for backend in backends:
            self.register(backend)

    def register(self, backend: SimulationBackend) -> None:
        self._backends[backend.backend_id] = backend

    def get(self, backend_id: str) -> SimulationBackend:
        try:
            return self._backends[backend_id]
        except KeyError as exc:
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
        backend = self.factory.get(cfg.backend_id)
        return backend.run(scenario, cfg)
