"""JSBSim Cessna fixed-wing backend.

This adapter keeps JSBSim behind the same normalized ``RunResult`` contract as
the quadcopter backends. A lightweight waypoint autopilot drives heading,
flight-path angle, and throttle through the JSBSim property tree.
"""
from __future__ import annotations

import importlib
import math
import time
from typing import Any, Callable

import numpy as np

from dronesim.models import RunConfig, RunResult, RunSummary, ScenarioSpec
from dronesim.services.terrain import MapCacheMiss, TerrainService, local_to_lat_lon, to_local_meters
from dronesim.sim.backends import BackendUnavailable, SimulationBackend, StepProgressCb
from dronesim.sim.debug_log import get_sim_logger

logger = get_sim_logger(__name__)

_FT_PER_M = 3.280839895
_FPS_PER_MPS = 3.280839895
_KTS_PER_MPS = 1.943844492
_M_PER_FT = 0.3048
_MPS_PER_FPS = 0.3048

_CRUISE_SPEED_MIN_MPS = 15.0
_CRUISE_SPEED_MAX_MPS = 70.0
_DEFAULT_CAPTURE_RADIUS_M = 75.0
_DEFAULT_MIN_AGL_M = 10.0
_DEFAULT_MAX_SINK_MPS = 5.0
_DEFAULT_MAX_CLIMB_DEG = 8.0
_DEFAULT_MAX_DESCENT_DEG = 8.0
_IC_SETTLE_STEPS = 30

_INSTALL_HINT = (
    "Install JSBSim to enable the Cessna backend. "
    "Try `pip install -r requirements-jsbsim.txt`; if aircraft data is outside "
    "the package defaults, set vehicle.parameters.jsbsim_root or aircraft_path."
)


def _detect_jsbsim() -> tuple[bool, str | None]:
    try:
        importlib.import_module("jsbsim")
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return False, f"jsbsim import failed: {exc}"
    return True, None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _angle_wrap_rad(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    try:
        return np.gradient(values, time_s, axis=0)
    except Exception:
        dt = float(np.nanmean(np.diff(time_s))) if len(time_s) > 1 else 1.0
        return np.gradient(values, axis=0) / max(dt, 1e-9)


def _normalize_altitude_reference(value: object) -> str:
    ref = str(value or "agl").strip().lower()
    return "msl" if ref == "msl" else "agl"


def _resolve_waypoints_3d(
    scenario: ScenarioSpec,
    run_config: RunConfig,
    terrain_service: TerrainService,
    params: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], str, list[str]]:
    """Resolve waypoints to local ENU with z in meters MSL for JSBSim."""
    scenario.validate()
    default_z = float(run_config.target_altitude_m)
    altitude_ref = _normalize_altitude_reference(params.get("altitude_reference", "agl"))
    warnings: list[str] = []

    terrain_at: Callable[[float, float], float] | None = None
    if altitude_ref == "agl":
        terrain_at, _offset = _resolve_terrain_elevation_fn(scenario, terrain_service)
        if terrain_at is None:
            warnings.append(
                "altitude_reference=agl but terrain map unavailable; using waypoint z as MSL"
            )
            altitude_ref = "msl"

    resolved: list[tuple[float, float, float]] = []
    for wp in scenario.waypoints.waypoints:
        if not wp.has_local_xy():
            wp = terrain_service.waypoint_to_local(wp, scenario.map)
        x_m, y_m, _z_m = wp.local_xyz()
        if wp.z_m is not None:
            z_value = float(wp.z_m)
        elif wp.alt_m != 0.0 or default_z == 0.0:
            z_value = float(wp.alt_m)
        else:
            z_value = default_z
        if altitude_ref == "agl" and terrain_at is not None:
            z_msl = terrain_at(float(x_m), float(y_m)) + z_value
        else:
            z_msl = z_value
        resolved.append((float(x_m), float(y_m), float(z_msl)))

    if len(resolved) < 2:
        raise ValueError("JSBSim Cessna backend requires at least 2 waypoints")
    return resolved, altitude_ref, warnings


def _resolve_terrain_elevation_fn(
    scenario: ScenarioSpec,
    terrain_service: TerrainService,
) -> tuple[Callable[[float, float], float] | None, float]:
    env = scenario.environment
    offset_m = float(getattr(env, "terrain_collision_offset_m", 0.5))
    try:
        asset = terrain_service.fetch_map(scenario.map, fetch_remote=False)
    except MapCacheMiss as exc:
        logger.warning(
            "Terrain map unavailable for scenario=%s: %s",
            scenario.scenario_id,
            exc,
        )
        return None, offset_m
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load terrain for scenario=%s: %s",
            scenario.scenario_id,
            exc,
        )
        return None, offset_m
    return asset.elevation_at, offset_m


def _bearing_rad(from_xy: tuple[float, float], to_xy: tuple[float, float], fallback: float) -> float:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback
    return math.atan2(dx, dy)


def _horizontal_distance_xy(pos: list[float], target: tuple[float, float, float]) -> float:
    return float(math.hypot(pos[0] - target[0], pos[1] - target[1]))


def _distance_xyz(pos: list[float], target: tuple[float, float, float]) -> float:
    return float(
        math.sqrt(
            (pos[0] - target[0]) ** 2
            + (pos[1] - target[1]) ** 2
            + (pos[2] - target[2]) ** 2
        )
    )


def _as_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _clamp_cruise_speed(cruise_speed_mps: float) -> tuple[float, bool]:
    clamped = _clamp(cruise_speed_mps, _CRUISE_SPEED_MIN_MPS, _CRUISE_SPEED_MAX_MPS)
    return clamped, abs(clamped - cruise_speed_mps) > 1e-6


def _state_is_finite(pos: list[float], vel: list[float], att: list[float]) -> bool:
    for component in (*pos, *vel, *att):
        if not math.isfinite(float(component)):
            return False
    return True


class JSBSimCessnaBackend(SimulationBackend):
    """JSBSim C172/Cessna adapter for waypoint-driven fixed-wing simulation."""

    backend_id = "jsbsim_cessna"
    display_name = "JSBSim Cessna"

    def __init__(self, terrain_service: TerrainService | None = None) -> None:
        self.terrain_service = terrain_service or TerrainService()
        self._jsbsim_available, self._import_error = _detect_jsbsim()
        if self._jsbsim_available:
            logger.info("JSBSimCessnaBackend ready")
        else:
            logger.debug("JSBSimCessnaBackend unavailable: %s", self._import_error)

    @property
    def libs_available(self) -> bool:
        return self._jsbsim_available

    def availability_summary(self) -> dict[str, object]:
        return {"jsbsim": self._jsbsim_available, "import_error": self._import_error}

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
                "JSBSimCessnaBackend.run() invoked without jsbsim "
                "(scenario=%s run_id=%s)",
                scenario.scenario_id,
                cfg.run_id,
            )
            detail = f" {self._import_error}" if self._import_error else ""
            raise BackendUnavailable(f"{self.display_name} requires jsbsim.{detail} {_INSTALL_HINT}")
        return self._run_with_jsbsim(scenario, cfg, on_step_progress=on_step_progress)

    def _run_with_jsbsim(
        self,
        scenario: ScenarioSpec,
        cfg: RunConfig,
        *,
        on_step_progress: StepProgressCb | None = None,
    ) -> RunResult:
        jsbsim = importlib.import_module("jsbsim")
        params = scenario.vehicle.parameters or {}
        controller = scenario.vehicle.controller or {}
        waypoints, altitude_ref, altitude_warnings = _resolve_waypoints_3d(
            scenario, cfg, self.terrain_service, params
        )
        aircraft = str(params.get("aircraft", params.get("aircraft_xml", "c172p")))
        dt_s = float(cfg.dt_s)
        max_steps = int(cfg.max_steps)
        success_threshold = float(cfg.waypoint_threshold_m)

        cruise_raw = float(controller.get("cruise_speed_mps", params.get("cruise_speed_mps", 40.0)))
        cruise_speed_mps, cruise_clamped = _clamp_cruise_speed(cruise_raw)
        heading_gain = float(controller.get("heading_gain", 1.5))
        altitude_gain = float(controller.get("altitude_gain", 0.012))
        pitch_gain = float(controller.get("pitch_gain", altitude_gain * 80.0))
        throttle_gain = float(controller.get("throttle_gain", 0.02))
        base_throttle = float(controller.get("base_throttle", params.get("base_throttle", 0.65)))
        max_bank_deg = float(controller.get("max_bank_deg", params.get("max_bank_deg", 25.0)))
        elevator_trim = float(controller.get("elevator_trim", params.get("elevator_trim", 0.0)))
        capture_radius_m = float(
            controller.get("waypoint_capture_radius_m", _DEFAULT_CAPTURE_RADIUS_M)
        )
        min_agl_m = float(controller.get("min_agl_m", _DEFAULT_MIN_AGL_M))
        max_sink_mps = float(controller.get("max_sink_rate_mps", _DEFAULT_MAX_SINK_MPS))
        max_climb_deg = float(controller.get("max_climb_deg", _DEFAULT_MAX_CLIMB_DEG))
        max_descent_deg = float(controller.get("max_descent_deg", _DEFAULT_MAX_DESCENT_DEG))
        ic_settle_steps = int(params.get("ic_settle_steps", _IC_SETTLE_STEPS))

        terrain_at, terrain_offset_m = _resolve_terrain_elevation_fn(scenario, self.terrain_service)
        terrain_collision = bool(getattr(scenario.environment, "terrain_collision_enabled", False))

        logger.info(
            "Starting JSBSim Cessna run scenario=%s run_id=%s aircraft=%s waypoints=%d "
            "dt=%.4f max_steps=%d alt_ref=%s",
            scenario.scenario_id,
            cfg.run_id,
            aircraft,
            len(waypoints),
            dt_s,
            max_steps,
            altitude_ref,
        )

        fdm = self._build_fdm(jsbsim, params)
        self._load_aircraft(fdm, aircraft, params)
        self._set_dt(fdm, dt_s)
        self._set_initial_conditions(
            fdm,
            scenario,
            waypoints,
            cruise_speed_mps,
            params,
            base_throttle=base_throttle,
        )
        self._set_environment(fdm, scenario)
        if not self._call_bool(fdm.run_ic):
            raise RuntimeError(f"JSBSim failed to initialize aircraft model '{aircraft}'")

        self._start_engine(fdm, params, base_throttle)
        self._run_ic_settle(fdm, base_throttle, ic_settle_steps)

        time_s: list[float] = []
        position_m: list[list[float]] = []
        velocity_mps: list[list[float]] = []
        attitude_rad: list[list[float]] = []
        angular_rate_rad_s: list[list[float]] = []
        controls: list[list[float]] = []
        reference_position_m: list[list[float]] = []

        wp_idx = 0
        status = "completed_with_miss"
        success = False
        commanded_gamma_deg = 0.0
        t0 = time.perf_counter()
        max_bank_rad = math.radians(max(max_bank_deg, 1.0))

        for step in range(max_steps):
            pos = self._position_local_m(fdm, scenario)
            nav_idx = min(wp_idx + 1, len(waypoints) - 1)
            target = waypoints[nav_idx]
            if wp_idx < len(waypoints) - 1:
                if _horizontal_distance_xy(pos, target) < capture_radius_m:
                    wp_idx += 1
                    nav_idx = min(wp_idx + 1, len(waypoints) - 1)
                    target = waypoints[nav_idx]
            att = self._attitude_rad(fdm)
            vel = self._velocity_mps(fdm)
            speed_mps = self._speed_mps(fdm)

            aileron, rudder, elevator, throttle, commanded_gamma_deg = self._compute_autopilot(
                pos=pos,
                vel=vel,
                att=att,
                target=target,
                speed_mps=speed_mps,
                cruise_speed_mps=cruise_speed_mps,
                heading_gain=heading_gain,
                pitch_gain=pitch_gain,
                max_bank_rad=max_bank_rad,
                elevator_trim=elevator_trim,
                base_throttle=base_throttle,
                throttle_gain=throttle_gain,
                capture_radius_m=capture_radius_m,
                max_climb_deg=max_climb_deg,
                max_descent_deg=max_descent_deg,
                max_sink_mps=max_sink_mps,
                min_agl_m=min_agl_m,
                terrain_at=terrain_at,
                commanded_gamma_deg=commanded_gamma_deg,
                dt_s=dt_s,
            )

            self._set_controls(fdm, aileron=aileron, elevator=elevator, rudder=rudder, throttle=throttle)
            fdm.run()

            sample_t = (step + 1) * dt_s
            pos = self._position_local_m(fdm, scenario)
            vel = self._velocity_mps(fdm)
            att = self._attitude_rad(fdm)
            rates = self._angular_rates_rad_s(fdm)

            if not _state_is_finite(pos, vel, att):
                status = "unstable"
                break

            agl_m = self._agl_m(pos, terrain_at)
            if terrain_collision and terrain_at is not None and agl_m < terrain_offset_m:
                status = "ground_collision"
                time_s.append(sample_t)
                position_m.append(pos)
                velocity_mps.append(vel)
                attitude_rad.append(att)
                angular_rate_rad_s.append(rates)
                controls.append([aileron, elevator, rudder, throttle])
                reference_position_m.append([target[0], target[1], target[2]])
                break

            if terrain_at is not None and agl_m < min_agl_m * 0.5:
                status = "ground_collision"
                time_s.append(sample_t)
                position_m.append(pos)
                velocity_mps.append(vel)
                attitude_rad.append(att)
                angular_rate_rad_s.append(rates)
                controls.append([aileron, elevator, rudder, throttle])
                reference_position_m.append([target[0], target[1], target[2]])
                break

            time_s.append(sample_t)
            position_m.append(pos)
            velocity_mps.append(vel)
            attitude_rad.append(att)
            angular_rate_rad_s.append(rates)
            controls.append([aileron, elevator, rudder, throttle])
            reference_position_m.append([target[0], target[1], target[2]])

            if on_step_progress is not None:
                try:
                    on_step_progress(step + 1, max_steps)
                except Exception:
                    pass

            if wp_idx >= len(waypoints) - 1:
                horiz = _horizontal_distance_xy(pos, waypoints[-1])
                vert = abs(pos[2] - waypoints[-1][2])
                if horiz < success_threshold and vert < success_threshold:
                    success = True
                    status = "success"
                    break

        wallclock_s = time.perf_counter() - t0
        time_arr = np.asarray(time_s, dtype=float)
        pos_arr = np.asarray(position_m, dtype=float)
        vel_arr = np.asarray(velocity_mps, dtype=float)
        ref_arr = np.asarray(reference_position_m, dtype=float)
        acc_arr = _gradient(vel_arr, time_arr) if len(vel_arr) else np.zeros((0, 3))
        tracking_error = (
            np.linalg.norm(pos_arr[:, :3] - ref_arr[:, :3], axis=1) if len(pos_arr) else np.array([])
        )
        final_pos = pos_arr[-1] if len(pos_arr) else np.asarray(waypoints[0], dtype=float)
        miss_distance = float(np.linalg.norm(final_pos[:3] - np.asarray(waypoints[-1], dtype=float)))

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
        metadata: dict[str, object] = {
            "backend_display_name": self.display_name,
            "source": "jsbsim.FGFDMExec",
            "aircraft": aircraft,
            "altitude_reference": altitude_ref,
            "waypoints_local_xyz_msl": [[x, y, z] for x, y, z in waypoints],
            "waypoints_local_xyz": [[x, y, z] for x, y, z in waypoints],
            "control_channels": ["aileron_norm", "elevator_norm", "rudder_norm", "throttle_norm"],
            "controller": {
                "type": "waypoint_autopilot",
                "cruise_speed_mps": cruise_speed_mps,
                "max_bank_deg": max_bank_deg,
                "waypoint_capture_radius_m": capture_radius_m,
            },
            "coordinate_frame": "local_enu_meters_from_map_center",
        }
        if cruise_clamped:
            metadata["cruise_speed_clamped"] = True
            metadata["cruise_speed_requested_mps"] = cruise_raw
        if altitude_warnings:
            metadata["altitude_warnings"] = altitude_warnings
        version = getattr(jsbsim, "__version__", None)
        if version is not None:
            metadata["jsbsim_version"] = str(version)

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

    def _compute_autopilot(
        self,
        *,
        pos: list[float],
        vel: list[float],
        att: list[float],
        target: tuple[float, float, float],
        speed_mps: float,
        cruise_speed_mps: float,
        heading_gain: float,
        pitch_gain: float,
        max_bank_rad: float,
        elevator_trim: float,
        base_throttle: float,
        throttle_gain: float,
        capture_radius_m: float,
        max_climb_deg: float,
        max_descent_deg: float,
        max_sink_mps: float,
        min_agl_m: float,
        terrain_at: Callable[[float, float], float] | None,
        commanded_gamma_deg: float,
        dt_s: float,
    ) -> tuple[float, float, float, float, float]:
        roll, pitch, yaw = att[0], att[1], att[2]
        horiz_dist = max(_horizontal_distance_xy(pos, target), 1.0)
        desired_heading = _bearing_rad((pos[0], pos[1]), (target[0], target[1]), yaw)
        heading_error = _angle_wrap_rad(desired_heading - yaw)

        dist_scale = min(1.0, capture_radius_m / horiz_dist)
        bank_cmd = heading_gain * heading_error / max_bank_rad * dist_scale
        if abs(roll) > max_bank_rad:
            bank_cmd *= 0.25
        if vel[2] < -max_sink_mps * 0.5:
            bank_cmd *= 0.4

        agl_m = self._agl_m(pos, terrain_at)
        if agl_m < min_agl_m:
            bank_cmd *= 0.3

        aileron = _clamp(bank_cmd, -1.0, 1.0)
        rudder = _clamp(0.25 * aileron, -1.0, 1.0)

        altitude_error = target[2] - pos[2]
        gamma_target_deg = _clamp(pitch_gain * altitude_error, -max_descent_deg, max_climb_deg)
        if vel[2] < -max_sink_mps:
            gamma_target_deg = max(gamma_target_deg, max_climb_deg * 0.75)
        if agl_m < min_agl_m:
            gamma_target_deg = max_climb_deg
        if pitch < math.radians(-12.0):
            gamma_target_deg = max_climb_deg
        elif pitch > math.radians(18.0):
            gamma_target_deg = min(gamma_target_deg, -max_descent_deg * 0.5)

        gamma_rate_limit = 4.0 * dt_s
        commanded_gamma_deg += _clamp(
            gamma_target_deg - commanded_gamma_deg,
            -gamma_rate_limit,
            gamma_rate_limit,
        )
        elevator = _clamp(
            elevator_trim + commanded_gamma_deg / max(max_climb_deg, 1.0),
            -1.0,
            1.0,
        )
        throttle = _clamp(base_throttle + throttle_gain * (cruise_speed_mps - speed_mps), 0.0, 1.0)
        return aileron, rudder, elevator, throttle, commanded_gamma_deg

    @staticmethod
    def _agl_m(pos: list[float], terrain_at: Callable[[float, float], float] | None) -> float:
        if terrain_at is None:
            return float("inf")
        try:
            return pos[2] - terrain_at(pos[0], pos[1])
        except (ValueError, IndexError):
            return float("inf")

    @staticmethod
    def _build_fdm(jsbsim: Any, params: dict[str, Any]):
        root = params.get("jsbsim_root") or None
        try:
            return jsbsim.FGFDMExec(root, None)
        except TypeError:
            return jsbsim.FGFDMExec(root)

    @staticmethod
    def _call_bool(fn) -> bool:
        result = fn()
        return True if result is None else bool(result)

    @staticmethod
    def _set_prop(fdm, name: str, value: float) -> None:
        if hasattr(fdm, "set_property_value"):
            fdm.set_property_value(name, float(value))
        else:
            fdm[name] = float(value)

    @staticmethod
    def _get_prop(fdm, name: str, fallback: float = 0.0) -> float:
        try:
            if hasattr(fdm, "get_property_value"):
                value = fdm.get_property_value(name)
            else:
                value = fdm[name]
        except Exception:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _load_aircraft(self, fdm, aircraft: str, params: dict[str, Any]) -> None:
        aircraft_path = params.get("aircraft_path")
        engine_path = params.get("engine_path")
        systems_path = params.get("systems_path")
        if aircraft_path and engine_path and systems_path and hasattr(fdm, "load_model_with_paths"):
            ok = fdm.load_model_with_paths(
                aircraft,
                str(aircraft_path),
                str(engine_path),
                str(systems_path),
            )
        else:
            ok = fdm.load_model(aircraft)
        if not ok:
            raise RuntimeError(f"JSBSim could not load aircraft model '{aircraft}'")

    def _set_dt(self, fdm, dt_s: float) -> None:
        if hasattr(fdm, "set_dt"):
            fdm.set_dt(float(dt_s))
            return
        self._set_prop(fdm, "simulation/dt", float(dt_s))

    def _set_initial_conditions(
        self,
        fdm,
        scenario: ScenarioSpec,
        waypoints: list[tuple[float, float, float]],
        cruise_speed_mps: float,
        params: dict[str, Any],
        *,
        base_throttle: float,
    ) -> None:
        x0, y0, z0 = waypoints[0]
        lat0, lon0 = local_to_lat_lon(x0, y0, scenario.map.center_lat, scenario.map.center_lon)
        heading = math.degrees(_bearing_rad(waypoints[0][:2], waypoints[1][:2], 0.0))
        heading_override = params.get("initial_heading_deg")
        if heading_override not in (None, ""):
            heading = float(heading_override)
        initial_speed = float(params.get("initial_ias_mps", cruise_speed_mps))
        flap_cmd = float(params.get("flap_cmd_norm", 0.0))

        self._set_prop(fdm, "ic/lat-gc-deg", float(lat0))
        self._set_prop(fdm, "ic/long-gc-deg", float(lon0))
        self._set_prop(fdm, "ic/h-sl-ft", float(z0) * _FT_PER_M)
        self._set_prop(fdm, "ic/vc-kts", initial_speed * _KTS_PER_MPS)
        self._set_prop(fdm, "ic/psi-true-deg", heading)
        self._set_prop(fdm, "ic/gamma-deg", float(params.get("initial_flight_path_deg", 0.0)))
        self._set_prop(fdm, "ic/theta-deg", float(params.get("initial_pitch_deg", 0.0)))
        self._set_prop(fdm, "ic/phi-deg", float(params.get("initial_roll_deg", 0.0)))
        self._set_prop(fdm, "fcs/flap-cmd-norm", flap_cmd)
        self._set_controls(fdm, aileron=0.0, elevator=0.0, rudder=0.0, throttle=base_throttle)

    def _start_engine(self, fdm, params: dict[str, Any], base_throttle: float) -> None:
        if not _as_bool(params.get("engine_running"), default=True):
            return
        for name in ("propulsion/engine/set-running", "propulsion/engine[0]/set-running"):
            try:
                self._set_prop(fdm, name, 1.0)
            except Exception:
                pass
        self._set_controls(fdm, aileron=0.0, elevator=0.0, rudder=0.0, throttle=base_throttle)

    def _run_ic_settle(self, fdm, base_throttle: float, steps: int) -> None:
        for _ in range(max(steps, 0)):
            self._set_controls(fdm, aileron=0.0, elevator=0.0, rudder=0.0, throttle=base_throttle)
            fdm.run()

    def _set_environment(self, fdm, scenario: ScenarioSpec) -> None:
        wind = scenario.environment.wind_mps or [0.0, 0.0, 0.0]
        padded = [0.0, 0.0, 0.0]
        for i, value in enumerate(list(wind)[:3]):
            padded[i] = float(value)
        self._set_prop(fdm, "atmosphere/wind-east-fps", padded[0] * _FPS_PER_MPS)
        self._set_prop(fdm, "atmosphere/wind-north-fps", padded[1] * _FPS_PER_MPS)
        self._set_prop(fdm, "atmosphere/wind-down-fps", -padded[2] * _FPS_PER_MPS)

    def _set_controls(
        self,
        fdm,
        *,
        aileron: float,
        elevator: float,
        rudder: float,
        throttle: float,
    ) -> None:
        self._set_prop(fdm, "fcs/aileron-cmd-norm", aileron)
        self._set_prop(fdm, "fcs/elevator-cmd-norm", elevator)
        self._set_prop(fdm, "fcs/rudder-cmd-norm", rudder)
        self._set_prop(fdm, "fcs/throttle-cmd-norm", throttle)
        self._set_prop(fdm, "fcs/throttle-cmd-norm[0]", throttle)

    def _position_local_m(self, fdm, scenario: ScenarioSpec) -> list[float]:
        lat = self._get_prop(fdm, "position/lat-gc-deg", scenario.map.center_lat)
        lon = self._get_prop(fdm, "position/long-gc-deg", scenario.map.center_lon)
        if lat == scenario.map.center_lat and lon == scenario.map.center_lon:
            lat = self._get_prop(fdm, "position/lat-geod-deg", lat)
            lon = self._get_prop(fdm, "position/long-gc-deg", lon)
        x, y = to_local_meters(
            np.array([lat], dtype=float),
            np.array([lon], dtype=float),
            scenario.map.center_lat,
            scenario.map.center_lon,
        )
        z = self._get_prop(fdm, "position/h-sl-ft", 0.0) * _M_PER_FT
        return [float(x[0]), float(y[0]), float(z)]

    def _velocity_mps(self, fdm) -> list[float]:
        north = self._get_prop(fdm, "velocities/v-north-fps", math.nan)
        east = self._get_prop(fdm, "velocities/v-east-fps", math.nan)
        down = self._get_prop(fdm, "velocities/v-down-fps", math.nan)
        if math.isfinite(north) and math.isfinite(east) and math.isfinite(down):
            return [east * _MPS_PER_FPS, north * _MPS_PER_FPS, -down * _MPS_PER_FPS]
        u = self._get_prop(fdm, "velocities/u-fps", 0.0)
        v = self._get_prop(fdm, "velocities/v-fps", 0.0)
        w = self._get_prop(fdm, "velocities/w-fps", 0.0)
        yaw = self._get_prop(fdm, "attitude/psi-rad", 0.0)
        east_mps = (u * math.sin(yaw) + v * math.cos(yaw)) * _MPS_PER_FPS
        north_mps = (u * math.cos(yaw) - v * math.sin(yaw)) * _MPS_PER_FPS
        return [east_mps, north_mps, -w * _MPS_PER_FPS]

    def _speed_mps(self, fdm) -> float:
        vt = self._get_prop(fdm, "velocities/vt-fps", math.nan)
        if math.isfinite(vt):
            return vt * _MPS_PER_FPS
        vel = self._velocity_mps(fdm)
        return float(math.sqrt(sum(component * component for component in vel)))

    def _attitude_rad(self, fdm) -> list[float]:
        return [
            self._get_prop(fdm, "attitude/phi-rad", 0.0),
            self._get_prop(fdm, "attitude/theta-rad", 0.0),
            self._get_prop(fdm, "attitude/psi-rad", 0.0),
        ]

    def _angular_rates_rad_s(self, fdm) -> list[float]:
        return [
            self._get_prop(fdm, "velocities/p-rad_sec", 0.0),
            self._get_prop(fdm, "velocities/q-rad_sec", 0.0),
            self._get_prop(fdm, "velocities/r-rad_sec", 0.0),
        ]


def factory_default() -> JSBSimCessnaBackend:
    """Constructor wrapper kept for symmetry with other backend modules."""
    return JSBSimCessnaBackend()


__all__: list[str] = ["JSBSimCessnaBackend", "factory_default"]
