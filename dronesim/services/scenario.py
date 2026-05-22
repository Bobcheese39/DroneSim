"""Scenario creation, validation, and persistence."""
from __future__ import annotations

import shutil
from pathlib import Path

from dronesim.models import (
    MapSpec,
    Marker,
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

    def create_from_local_waypoints(
        self,
        *,
        name: str,
        points_xy: list[tuple[float, float]],
        altitude_m: float = 5.0,
        map_spec: MapSpec | None = None,
        markers: list[Marker] | None = None,
    ) -> ScenarioSpec:
        scenario = ScenarioSpec(
            name=name,
            map=map_spec or MapSpec(),
            waypoints=WaypointSet.from_local_xy(points_xy, altitude_m=altitude_m),
        )
        scenario.markers.markers = markers or []
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
        normalized = [self.terrain_service.waypoint_to_local(wp, map_spec) for wp in waypoints]
        scenario = ScenarioSpec(
            name=name,
            map=map_spec,
            waypoints=WaypointSet(waypoints=normalized, coordinate_frame="wgs84+local_enu"),
        )
        scenario.markers.markers = markers or []
        if normalized:
            scenario.run_config.target_altitude_m = normalized[0].alt_m or scenario.run_config.target_altitude_m
        scenario.validate()
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
