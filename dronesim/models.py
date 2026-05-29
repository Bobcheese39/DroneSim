"""Shared data contracts for DroneSim.

These dataclasses keep GUI, storage, map services, and simulator backends
decoupled from the current in-house quadcopter implementation.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1.0"


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp for persisted artifacts."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_ready(value: Any) -> Any:
    """Convert nested dataclasses and numpy-ish scalar values to JSON values."""
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class MapSpec:
    center_lat: float = 37.6188056
    center_lon: float = -122.3754167
    radius_km: float = 1.0
    resolution: int = 400
    name: str = "default_map"
    imagery_source: str = "esri_world_imagery"
    elevation_source: str = "aws_terrarium"
    cache_key: str | None = None
    vertical_exaggeration: float = 1.0

    def key(self) -> str:
        if self.cache_key:
            return self.cache_key
        return (
            f"{self.center_lat:.6f}_{self.center_lon:.6f}_"
            f"r{self.radius_km:.3f}_n{self.resolution}"
        ).replace("-", "m").replace(".", "p")

    def validate(self) -> None:
        if not -90.0 <= self.center_lat <= 90.0:
            raise ValueError("Map center_lat must be between -90 and 90 degrees")
        if not -180.0 <= self.center_lon <= 180.0:
            raise ValueError("Map center_lon must be between -180 and 180 degrees")
        if self.radius_km <= 0:
            raise ValueError("Map radius_km must be positive")
        if self.resolution < 16:
            raise ValueError("Map resolution must be at least 16")


@dataclass
class Waypoint:
    lat: float | None = None
    lon: float | None = None
    alt_m: float = 0.0
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    time_s: float | None = None
    speed_mps: float | None = None
    label: str = ""

    @classmethod
    def local(cls, x_m: float, y_m: float, z_m: float = 0.0, label: str = "") -> "Waypoint":
        return cls(x_m=x_m, y_m=y_m, z_m=z_m, alt_m=z_m, label=label)

    @classmethod
    def geographic(
        cls, lat: float, lon: float, alt_m: float = 0.0, label: str = ""
    ) -> "Waypoint":
        return cls(lat=lat, lon=lon, alt_m=alt_m, z_m=alt_m, label=label)

    def has_local_xy(self) -> bool:
        return self.x_m is not None and self.y_m is not None

    def has_geographic(self) -> bool:
        return self.lat is not None and self.lon is not None

    def local_xyz(self) -> tuple[float, float, float]:
        if self.x_m is None or self.y_m is None:
            raise ValueError(f"Waypoint {self.label or '<unnamed>'} has no local x/y")
        return float(self.x_m), float(self.y_m), float(self.z_m if self.z_m is not None else self.alt_m)


@dataclass
class WaypointSet:
    waypoints: list[Waypoint] = field(default_factory=list)
    coordinate_frame: str = "local_ned_or_enu"
    smoothing: str = "spline"
    default_alt_m: float = 5.0

    @classmethod
    def from_local_xy(cls, points: list[tuple[float, float]], altitude_m: float = 5.0) -> "WaypointSet":
        return cls(
            waypoints=[
                Waypoint.local(float(x), float(y), float(altitude_m), label=f"WP{i}")
                for i, (x, y) in enumerate(points)
            ],
            default_alt_m=altitude_m,
        )

    def validate(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("A scenario needs at least two waypoints")
        for wp in self.waypoints:
            if not (wp.has_local_xy() or wp.has_geographic()):
                raise ValueError("Each waypoint must have local x/y or lat/lon coordinates")

    def local_xy_array(self) -> list[list[float]]:
        self.validate()
        return [[wp.local_xyz()[0], wp.local_xyz()[1]] for wp in self.waypoints]


@dataclass
class Marker:
    label: str
    lat: float | None = None
    lon: float | None = None
    alt_m: float = 0.0
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    color: str = "red"
    size: float = 10.0
    visible: bool = True
    role: str = "annotation"
    notes: str = ""


@dataclass
class MarkerSet:
    markers: list[Marker] = field(default_factory=list)


@dataclass
class DroneModelSpec:
    model_id: str = "inhouse_mpc_quad"
    model_type: str = "quadcopter"
    backend_id: str = "inhouse_mpc_quad"
    display_name: str = "In-house MPC Quadcopter"
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "mass": 5.0,
        "Ix": 1.0,
        "Iy": 1.0,
        "Iz": 1.5,
        # Phase 6 aero block. Zero coefficients => no drag => legacy behavior.
        # TODO Phase 6 follow-up: rotor allocation, motor lag, thrust/torque
        # coefficients, actuator saturation, battery sag, sensor noise live here.
        "aero": {
            "cd_linear": 0.0,
            "cd_quadratic": 0.0,
            "reference_area_m2": 0.1,
        },
    })
    controller: dict[str, Any] = field(default_factory=lambda: {
        "type": "mpc",
        "horizon": 20,
        "lookahead": 60,
    })
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentSpec:
    gravity_mps2: float = 9.80665
    wind_mps: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    turbulence: dict[str, Any] = field(default_factory=dict)
    terrain_collision_enabled: bool = False
    # Phase 6: gust + atmosphere. Zero std == no gusts (legacy-equivalent).
    gust_std_mps: float = 0.0
    gust_decorrelation_s: float = 2.0
    air_density_kg_m3: float = 1.225
    # Distance kept above terrain before a collision is declared. Tunable so
    # high-resolution terrain doesn't trigger spurious collisions on hover.
    terrain_collision_offset_m: float = 0.5


@dataclass
class RunConfig:
    run_id: str = field(default_factory=lambda: new_id("run"))
    backend_id: str = "inhouse_mpc_quad"
    dt_s: float = 0.1
    max_steps: int = 250
    target_altitude_m: float = 5.0
    horizon: int = 20
    lookahead: int = 60
    waypoint_threshold_m: float = 0.25
    seed: int | None = None
    # Phase 6 fidelity selectors.
    # ``integration_method``: "euler" (legacy) or "rk4".
    # ``fidelity_mode``: "auto" (pick extended path when any fidelity knob is
    # non-zero), "legacy" (force vendor path), or "extended" (force local
    # runtime even when knobs are zero, useful for parity tests).
    integration_method: str = "euler"
    fidelity_mode: str = "auto"
    monte_carlo: dict[str, Any] = field(default_factory=lambda: {
        "enabled": false_bool(),
        "n_trials": 1,
        "workers": 1,
        "base_seed": 0,
        "init_pos_std": 0.0,
        "init_vel_std": 0.0,
        "init_att_std": 0.0,
        "force_noise_std": 0.0,
        "mass_jitter_pct": 0.0,
        "inertia_jitter_pct": 0.0,
    })


def false_bool() -> bool:
    """Named helper keeps default_factory JSON-looking without a bare literal trap."""
    return False


@dataclass
class ScenarioSpec:
    scenario_id: str = field(default_factory=lambda: new_id("scenario"))
    schema_version: str = SCHEMA_VERSION
    name: str = "Untitled Scenario"
    description: str = ""
    map: MapSpec = field(default_factory=MapSpec)
    waypoints: WaypointSet = field(default_factory=lambda: WaypointSet.from_local_xy(
        [(0.0, 0.0), (1.0, 2.0), (2.0, 4.5), (3.0, 3.0)],
        altitude_m=5.0,
    ))
    markers: MarkerSet = field(default_factory=MarkerSet)
    vehicle: DroneModelSpec = field(default_factory=DroneModelSpec)
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    run_config: RunConfig = field(default_factory=RunConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=utc_now)
    updated_utc: str = field(default_factory=utc_now)

    def validate(self) -> None:
        self.map.validate()
        self.waypoints.validate()
        if self.run_config.dt_s <= 0:
            raise ValueError("RunConfig dt_s must be positive")
        if self.run_config.max_steps <= 0:
            raise ValueError("RunConfig max_steps must be positive")
        if not math.isfinite(self.run_config.target_altitude_m):
            raise ValueError("RunConfig target_altitude_m must be finite")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioSpec":
        map_spec = MapSpec(**payload.get("map", {}))
        wp_payload = payload.get("waypoints", {})
        waypoint_set = WaypointSet(
            waypoints=[Waypoint(**wp) for wp in wp_payload.get("waypoints", [])],
            coordinate_frame=wp_payload.get("coordinate_frame", "local_ned_or_enu"),
            smoothing=wp_payload.get("smoothing", "spline"),
            default_alt_m=wp_payload.get("default_alt_m", 5.0),
        )
        marker_payload = payload.get("markers", {})
        marker_set = MarkerSet(markers=[Marker(**m) for m in marker_payload.get("markers", [])])
        scenario = cls(
            scenario_id=payload.get("scenario_id", new_id("scenario")),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            name=payload.get("name", "Untitled Scenario"),
            description=payload.get("description", ""),
            map=map_spec,
            waypoints=waypoint_set,
            markers=marker_set,
            vehicle=DroneModelSpec(**payload.get("vehicle", {})),
            environment=EnvironmentSpec(**payload.get("environment", {})),
            run_config=RunConfig(**payload.get("run_config", {})),
            metadata=payload.get("metadata", {}),
            created_utc=payload.get("created_utc", utc_now()),
            updated_utc=payload.get("updated_utc", utc_now()),
        )
        scenario.validate()
        return scenario


@dataclass
class RunSummary:
    success: bool = False
    miss_distance_m: float | None = None
    settle_steps: int | None = None
    duration_s: float | None = None
    max_tracking_error_m: float | None = None
    mean_tracking_error_m: float | None = None
    max_altitude_m: float | None = None
    min_altitude_m: float | None = None
    wallclock_s: float | None = None


@dataclass
class RunResult:
    run_id: str
    scenario_id: str
    backend_id: str
    model_id: str
    status: str
    time_s: list[float] = field(default_factory=list)
    position_m: list[list[float]] = field(default_factory=list)
    velocity_mps: list[list[float]] = field(default_factory=list)
    acceleration_mps2: list[list[float]] = field(default_factory=list)
    attitude_rad: list[list[float]] = field(default_factory=list)
    angular_rate_rad_s: list[list[float]] = field(default_factory=list)
    controls: list[list[float]] = field(default_factory=list)
    reference_position_m: list[list[float]] = field(default_factory=list)
    tracking_error_m: list[float] = field(default_factory=list)
    summary: RunSummary = field(default_factory=RunSummary)
    units: dict[str, str] = field(default_factory=lambda: {
        "time_s": "s",
        "position_m": "m",
        "velocity_mps": "m/s",
        "acceleration_mps2": "m/s^2",
        "attitude_rad": "rad",
        "angular_rate_rad_s": "rad/s",
        "controls": "backend-specific SI units",
        "reference_position_m": "m",
        "tracking_error_m": "m",
    })
    metadata: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunResult":
        summary_payload = payload.get("summary", {})
        return cls(
            run_id=payload["run_id"],
            scenario_id=payload["scenario_id"],
            backend_id=payload["backend_id"],
            model_id=payload["model_id"],
            status=payload.get("status", "unknown"),
            time_s=payload.get("time_s", []),
            position_m=payload.get("position_m", []),
            velocity_mps=payload.get("velocity_mps", []),
            acceleration_mps2=payload.get("acceleration_mps2", []),
            attitude_rad=payload.get("attitude_rad", []),
            angular_rate_rad_s=payload.get("angular_rate_rad_s", []),
            controls=payload.get("controls", []),
            reference_position_m=payload.get("reference_position_m", []),
            tracking_error_m=payload.get("tracking_error_m", []),
            summary=RunSummary(**summary_payload),
            units=payload.get("units", {}),
            metadata=payload.get("metadata", {}),
            created_utc=payload.get("created_utc", utc_now()),
        )

    def trajectory_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, t in enumerate(self.time_s):
            pos = self.position_m[i] if i < len(self.position_m) else [None, None, None]
            vel = self.velocity_mps[i] if i < len(self.velocity_mps) else [None, None, None]
            acc = self.acceleration_mps2[i] if i < len(self.acceleration_mps2) else [None, None, None]
            att = self.attitude_rad[i] if i < len(self.attitude_rad) else [None, None, None]
            ref = self.reference_position_m[i] if i < len(self.reference_position_m) else [None, None, None]
            ctrl = self.controls[i] if i < len(self.controls) else [None, None, None, None]
            err = self.tracking_error_m[i] if i < len(self.tracking_error_m) else None
            rows.append({
                "time_s": t,
                "x_m": pos[0],
                "y_m": pos[1],
                "z_m": pos[2],
                "ref_x_m": ref[0],
                "ref_y_m": ref[1],
                "ref_z_m": ref[2],
                "vx_mps": vel[0],
                "vy_mps": vel[1],
                "vz_mps": vel[2],
                "ax_mps2": acc[0],
                "ay_mps2": acc[1],
                "az_mps2": acc[2],
                "roll_rad": att[0],
                "pitch_rad": att[1],
                "yaw_rad": att[2],
                "ft_N": ctrl[0] if len(ctrl) > 0 else None,
                "tx_Nm": ctrl[1] if len(ctrl) > 1 else None,
                "ty_Nm": ctrl[2] if len(ctrl) > 2 else None,
                "tz_Nm": ctrl[3] if len(ctrl) > 3 else None,
                "tracking_error_m": err,
            })
        return rows
