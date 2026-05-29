"""Helpers to (re)build a :class:`ScenarioSpec` from request payloads.

``ScenarioSpec.from_dict`` always validates, which rejects in-progress edits
(e.g. a scenario momentarily holding fewer than two waypoints while the user
edits on the map). The edit endpoints need a lenient path, so this module
reconstructs the dataclass graph without the final ``validate()`` call.
"""
from __future__ import annotations

from typing import Any

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
    utc_now,
)


def scenario_from_payload(payload: dict[str, Any], *, validate: bool = True) -> ScenarioSpec:
    """Build a scenario from a plain dict, optionally skipping validation."""
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
    scenario = ScenarioSpec(
        scenario_id=payload.get("scenario_id", new_id("scenario")),
        schema_version=payload.get("schema_version", ScenarioSpec().schema_version),
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
    if validate:
        scenario.validate()
    return scenario
