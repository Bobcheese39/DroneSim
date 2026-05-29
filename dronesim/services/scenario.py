"""Scenario creation, validation, and persistence."""
from __future__ import annotations

import shutil
from pathlib import Path

from dronesim.models import (
    DroneModelSpec,
    EnvironmentSpec,
    MapSpec,
    Marker,
    MarkerSet,
    RunConfig,
    ScenarioSpec,
    Waypoint,
    WaypointSet,
    new_id,
    read_json,
    utc_now,
    write_json,
)
from dronesim.services.terrain import TerrainService


class ScenarioManager:
    """Factory/repository for ScenarioSpec JSON files."""

    def __init__(self, scenario_root: str | Path = "scenarios") -> None:
        self.scenario_root = Path(scenario_root)
        self.scenario_root.mkdir(parents=True, exist_ok=True)
        self.terrain_service = TerrainService()

    def default_scenario(self, name: str = "Untitled Scenario") -> ScenarioSpec:
        return ScenarioSpec(name=name)

    def create(
        self,
        *,
        name: str,
        waypoints: list[Waypoint],
        map_spec: MapSpec | None = None,
        markers: list[Marker] | None = None,
        description: str = "",
        vehicle: DroneModelSpec | None = None,
        environment: EnvironmentSpec | None = None,
        run_config: RunConfig | None = None,
        metadata: dict | None = None,
        coordinate_frame: str | None = None,
    ) -> ScenarioSpec:
        map_spec = map_spec or MapSpec()
        normalized_waypoints = [
            self.terrain_service.waypoint_to_local(wp, map_spec) if wp.has_geographic() else wp
            for wp in waypoints
        ]
        normalized_markers = [
            self.terrain_service.marker_to_local(marker, map_spec)
            if marker.lat is not None and marker.lon is not None
            else marker
            for marker in (markers or [])
        ]
        frame = coordinate_frame or (
            "wgs84+local_enu"
            if any(wp.has_geographic() for wp in normalized_waypoints)
            else "local_enu"
        )
        scenario = ScenarioSpec(
            name=name,
            description=description,
            map=map_spec,
            waypoints=WaypointSet(waypoints=normalized_waypoints, coordinate_frame=frame),
            markers=MarkerSet(markers=normalized_markers),
            vehicle=vehicle or DroneModelSpec(),
            environment=environment or EnvironmentSpec(),
            run_config=run_config or RunConfig(),
            metadata=metadata or {},
        )
        if normalized_waypoints:
            first = normalized_waypoints[0]
            scenario.waypoints.default_alt_m = float(first.z_m if first.z_m is not None else first.alt_m)
            scenario.run_config.target_altitude_m = scenario.waypoints.default_alt_m
        scenario.validate()
        return scenario

    def create_from_local_waypoints(
        self,
        *,
        name: str,
        points_xy: list[tuple[float, float]],
        altitude_m: float = 5.0,
        map_spec: MapSpec | None = None,
        markers: list[Marker] | None = None,
    ) -> ScenarioSpec:
        scenario = self.create(
            name=name,
            map_spec=map_spec,
            waypoints=[
                Waypoint.local(float(x), float(y), float(altitude_m), label=f"WP{i}")
                for i, (x, y) in enumerate(points_xy)
            ],
            markers=markers,
            coordinate_frame="local_enu",
        )
        scenario.run_config.target_altitude_m = altitude_m
        scenario.validate()
        return scenario

    def create_from_geographic_waypoints(
        self,
        *,
        name: str,
        waypoints: list[Waypoint],
        map_spec: MapSpec,
        markers: list[Marker] | None = None,
    ) -> ScenarioSpec:
        scenario = self.create(
            name=name,
            map_spec=map_spec,
            waypoints=waypoints,
            markers=markers,
            coordinate_frame="wgs84+local_enu",
        )
        return scenario

    def validate(self, scenario: ScenarioSpec) -> list[str]:
        try:
            scenario.validate()
        except Exception as exc:
            return [str(exc)]
        return []

    def path_for(self, scenario: ScenarioSpec) -> Path:
        filename = f"{scenario.scenario_id}.json"
        return self.scenario_root / filename

    def save(self, scenario: ScenarioSpec, path: str | Path | None = None) -> Path:
        scenario.validate()
        scenario.updated_utc = utc_now()
        target = Path(path) if path is not None else self.path_for(scenario)
        return write_json(target, scenario.to_dict())

    def load(self, path_or_id: str | Path) -> ScenarioSpec:
        path = Path(path_or_id)
        if not path.suffix:
            path = self.scenario_root / f"{path_or_id}.json"
        return ScenarioSpec.from_dict(read_json(path))

    def duplicate(self, scenario: ScenarioSpec, *, name: str | None = None) -> ScenarioSpec:
        payload = scenario.to_dict()
        payload["scenario_id"] = new_id("scenario")
        payload["name"] = name or f"{scenario.name} Copy"
        payload["created_utc"] = utc_now()
        payload["updated_utc"] = utc_now()
        return ScenarioSpec.from_dict(payload)

    def export(self, scenario: ScenarioSpec, destination: str | Path) -> Path:
        saved = self.save(scenario)
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, dest)
        return dest

    def list_scenarios(self) -> list[ScenarioSpec]:
        scenarios: list[ScenarioSpec] = []
        for path in sorted(self.scenario_root.glob("*.json")):
            try:
                scenarios.append(self.load(path))
            except Exception:
                continue
        return scenarios
