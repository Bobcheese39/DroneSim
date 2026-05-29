"""ScenarioEditor service.

Centralizes add/delete/move/reorder operations for waypoints and markers
so the GUI no longer mutates list/DataFrame state in-line. Coordinates
are kept synchronized between the local ENU frame and WGS84 lat/lon via
:class:`~dronesim.services.terrain.TerrainService`.

This module is intentionally *pure* (no Panel dependency) so it can be
unit-tested without a running app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dronesim.models import MapSpec, Marker, MarkerSet, ScenarioSpec, Waypoint, WaypointSet
from dronesim.services.terrain import TerrainService, local_to_lat_lon


@dataclass
class MarkerPlacementState:
    """Two-step marker placement state machine.

    Phase 1 captures ``x_m``/``y_m`` on the first map click. Phase 2
    locks XY and only adjusts ``z_m``; clicking again (or pressing the
    commit button) finalizes the marker.
    """

    awaiting: str = "xy"  # "xy" | "z" | "commit"
    x_m: float | None = None
    y_m: float | None = None
    z_m: float = 0.0
    label: str = ""
    color: str = "#ffd166"

    def reset(self) -> None:
        self.awaiting = "xy"
        self.x_m = None
        self.y_m = None
        self.z_m = 0.0


class ScenarioEditor:
    """Mutates a :class:`ScenarioSpec` in place via high-level edits."""

    def __init__(self, terrain_service: TerrainService | None = None) -> None:
        self.terrain_service = terrain_service or TerrainService()

    # ------------------------------------------------------------------
    # Waypoint operations
    # ------------------------------------------------------------------

    def add_waypoint(
        self,
        scenario: ScenarioSpec,
        *,
        x_m: float | None = None,
        y_m: float | None = None,
        z_m: float | None = None,
        lat: float | None = None,
        lon: float | None = None,
        alt_m: float | None = None,
        label: str | None = None,
        index: int | None = None,
    ) -> int:
        """Append (or insert) a waypoint and sync hybrid coordinates.

        Returns the inserted index.
        """
        spec = scenario.map
        default_alt = scenario.waypoints.default_alt_m
        if x_m is not None and y_m is not None:
            z = float(z_m if z_m is not None else default_alt)
            new_label = label or _next_label(scenario.waypoints.waypoints, prefix="WP")
            waypoint = Waypoint.local(float(x_m), float(y_m), z, label=new_label)
            waypoint = self._sync_local_to_geographic(waypoint, spec)
        elif lat is not None and lon is not None:
            altitude = float(alt_m if alt_m is not None else default_alt)
            new_label = label or _next_label(scenario.waypoints.waypoints, prefix="WP")
            waypoint = Waypoint.geographic(float(lat), float(lon), altitude, label=new_label)
            waypoint = self.terrain_service.waypoint_to_local(waypoint, spec)
        else:
            raise ValueError("add_waypoint requires either local x/y or geographic lat/lon")

        target = scenario.waypoints.waypoints
        if index is None or index < 0 or index > len(target):
            target.append(waypoint)
            return len(target) - 1
        target.insert(index, waypoint)
        return index

    def delete_waypoint(self, scenario: ScenarioSpec, index: int) -> bool:
        waypoints = scenario.waypoints.waypoints
        if not 0 <= index < len(waypoints):
            return False
        waypoints.pop(index)
        return True

    def move_waypoint(
        self,
        scenario: ScenarioSpec,
        index: int,
        *,
        x_m: float | None = None,
        y_m: float | None = None,
        z_m: float | None = None,
    ) -> bool:
        waypoints = scenario.waypoints.waypoints
        if not 0 <= index < len(waypoints):
            return False
        wp = waypoints[index]
        if x_m is not None:
            wp.x_m = float(x_m)
        if y_m is not None:
            wp.y_m = float(y_m)
        if z_m is not None:
            wp.z_m = float(z_m)
            wp.alt_m = float(z_m)
        waypoints[index] = self._sync_local_to_geographic(wp, scenario.map)
        return True

    def reorder_waypoint(self, scenario: ScenarioSpec, index: int, new_index: int) -> bool:
        waypoints = scenario.waypoints.waypoints
        if not 0 <= index < len(waypoints):
            return False
        if not 0 <= new_index < len(waypoints):
            return False
        if index == new_index:
            return False
        wp = waypoints.pop(index)
        waypoints.insert(new_index, wp)
        return True

    def move_waypoint_up(self, scenario: ScenarioSpec, index: int) -> bool:
        if index <= 0:
            return False
        return self.reorder_waypoint(scenario, index, index - 1)

    def move_waypoint_down(self, scenario: ScenarioSpec, index: int) -> bool:
        if index >= len(scenario.waypoints.waypoints) - 1:
            return False
        return self.reorder_waypoint(scenario, index, index + 1)

    # ------------------------------------------------------------------
    # Marker operations
    # ------------------------------------------------------------------

    def add_marker(
        self,
        scenario: ScenarioSpec,
        *,
        x_m: float | None = None,
        y_m: float | None = None,
        z_m: float | None = None,
        lat: float | None = None,
        lon: float | None = None,
        alt_m: float | None = None,
        label: str | None = None,
        color: str = "yellow",
        size: float = 10.0,
        notes: str = "",
        role: str = "annotation",
        index: int | None = None,
    ) -> int:
        markers = scenario.markers.markers
        new_label = label or _next_label([m for m in markers], prefix="Marker")
        marker = Marker(label=new_label, color=color, size=float(size), notes=notes, role=role)
        if x_m is not None and y_m is not None:
            marker.x_m = float(x_m)
            marker.y_m = float(y_m)
            marker.z_m = float(z_m if z_m is not None else 0.0)
            marker.alt_m = float(marker.z_m)
            marker = self._sync_marker_local_to_geographic(marker, scenario.map)
        elif lat is not None and lon is not None:
            marker.lat = float(lat)
            marker.lon = float(lon)
            marker.alt_m = float(alt_m if alt_m is not None else 0.0)
            marker = self.terrain_service.marker_to_local(marker, scenario.map)
        else:
            raise ValueError("add_marker requires either local x/y or geographic lat/lon")

        if index is None or index < 0 or index > len(markers):
            markers.append(marker)
            return len(markers) - 1
        markers.insert(index, marker)
        return index

    def delete_marker(self, scenario: ScenarioSpec, index: int) -> bool:
        markers = scenario.markers.markers
        if not 0 <= index < len(markers):
            return False
        markers.pop(index)
        return True

    def move_marker(
        self,
        scenario: ScenarioSpec,
        index: int,
        *,
        x_m: float | None = None,
        y_m: float | None = None,
        z_m: float | None = None,
    ) -> bool:
        markers = scenario.markers.markers
        if not 0 <= index < len(markers):
            return False
        marker = markers[index]
        if x_m is not None:
            marker.x_m = float(x_m)
        if y_m is not None:
            marker.y_m = float(y_m)
        if z_m is not None:
            marker.z_m = float(z_m)
            marker.alt_m = float(z_m)
        markers[index] = self._sync_marker_local_to_geographic(marker, scenario.map)
        return True

    def reorder_marker(self, scenario: ScenarioSpec, index: int, new_index: int) -> bool:
        markers = scenario.markers.markers
        if not 0 <= index < len(markers):
            return False
        if not 0 <= new_index < len(markers):
            return False
        if index == new_index:
            return False
        marker = markers.pop(index)
        markers.insert(new_index, marker)
        return True

    def move_marker_up(self, scenario: ScenarioSpec, index: int) -> bool:
        if index <= 0:
            return False
        return self.reorder_marker(scenario, index, index - 1)

    def move_marker_down(self, scenario: ScenarioSpec, index: int) -> bool:
        if index >= len(scenario.markers.markers) - 1:
            return False
        return self.reorder_marker(scenario, index, index + 1)

    def update_marker_fields(
        self,
        scenario: ScenarioSpec,
        index: int,
        *,
        label: str | None = None,
        color: str | None = None,
        size: float | None = None,
        visible: bool | None = None,
        notes: str | None = None,
        role: str | None = None,
    ) -> bool:
        markers = scenario.markers.markers
        if not 0 <= index < len(markers):
            return False
        marker = markers[index]
        if label is not None:
            marker.label = str(label)
        if color is not None:
            marker.color = str(color)
        if size is not None:
            marker.size = float(size)
        if visible is not None:
            marker.visible = bool(visible)
        if notes is not None:
            marker.notes = str(notes)
        if role is not None:
            marker.role = str(role)
        return True

    # ------------------------------------------------------------------
    # Two-step marker placement helpers
    # ------------------------------------------------------------------

    def start_marker_placement(
        self,
        state: MarkerPlacementState,
        *,
        x_m: float,
        y_m: float,
        z_m: float = 0.0,
        label: str = "",
        color: str = "#ffd166",
    ) -> None:
        state.awaiting = "z"
        state.x_m = float(x_m)
        state.y_m = float(y_m)
        state.z_m = float(z_m)
        state.label = label
        state.color = color

    def adjust_marker_placement_z(
        self,
        state: MarkerPlacementState,
        z_m: float,
    ) -> None:
        state.z_m = float(z_m)

    def commit_marker_placement(
        self,
        scenario: ScenarioSpec,
        state: MarkerPlacementState,
        *,
        color: str = "yellow",
        size: float = 10.0,
        notes: str = "",
        role: str = "annotation",
    ) -> int | None:
        if state.x_m is None or state.y_m is None:
            return None
        label = state.label or _next_label(scenario.markers.markers, prefix="Marker")
        idx = self.add_marker(
            scenario,
            x_m=state.x_m,
            y_m=state.y_m,
            z_m=state.z_m,
            label=label,
            color=color,
            size=size,
            notes=notes,
            role=role,
        )
        state.reset()
        return idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_local_to_geographic(self, waypoint: Waypoint, spec: MapSpec) -> Waypoint:
        if waypoint.x_m is None or waypoint.y_m is None:
            return waypoint
        try:
            lat, lon = local_to_lat_lon(
                waypoint.x_m,
                waypoint.y_m,
                spec.center_lat,
                spec.center_lon,
            )
            waypoint.lat = float(lat)
            waypoint.lon = float(lon)
            if waypoint.alt_m is None:
                waypoint.alt_m = float(waypoint.z_m if waypoint.z_m is not None else 0.0)
            else:
                waypoint.alt_m = float(waypoint.z_m if waypoint.z_m is not None else waypoint.alt_m)
        except Exception:  # noqa: BLE001
            pass
        return waypoint

    def _sync_marker_local_to_geographic(self, marker: Marker, spec: MapSpec) -> Marker:
        if marker.x_m is None or marker.y_m is None:
            return marker
        try:
            lat, lon = local_to_lat_lon(
                marker.x_m,
                marker.y_m,
                spec.center_lat,
                spec.center_lon,
            )
            marker.lat = float(lat)
            marker.lon = float(lon)
            if marker.z_m is not None:
                marker.alt_m = float(marker.z_m)
        except Exception:  # noqa: BLE001
            pass
        return marker


def _next_label(items: Iterable, prefix: str) -> str:
    seq = list(items)
    return f"{prefix}{len(seq)}"


def make_scenario_default_waypoints() -> WaypointSet:
    return WaypointSet.from_local_xy(
        [(0.0, 0.0), (1.0, 2.0), (2.0, 4.5), (3.0, 3.0)],
        altitude_m=5.0,
    )


def make_scenario_default_markers() -> MarkerSet:
    return MarkerSet(
        markers=[
            Marker(label="Launch", x_m=0.0, y_m=0.0, z_m=5.0, alt_m=5.0, color="red"),
        ]
    )
