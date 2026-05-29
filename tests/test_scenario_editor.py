"""Tests for the :mod:`dronesim.services.scenario_editor` module."""
from __future__ import annotations

import unittest

from dronesim.models import MapSpec, Marker, ScenarioSpec, Waypoint, WaypointSet, MarkerSet
from dronesim.services.scenario_editor import MarkerPlacementState, ScenarioEditor


def _make_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="editor-test",
        map=MapSpec(center_lat=37.0, center_lon=-122.0, radius_km=0.5, resolution=32),
        waypoints=WaypointSet(
            waypoints=[
                Waypoint.local(0.0, 0.0, 5.0, label="WP0"),
                Waypoint.local(10.0, 0.0, 5.0, label="WP1"),
                Waypoint.local(10.0, 10.0, 5.0, label="WP2"),
            ],
            default_alt_m=5.0,
        ),
        markers=MarkerSet(
            markers=[
                Marker(label="A", x_m=1.0, y_m=1.0, z_m=2.0, alt_m=2.0),
                Marker(label="B", x_m=3.0, y_m=3.0, z_m=4.0, alt_m=4.0),
            ]
        ),
    )


class WaypointEditOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.editor = ScenarioEditor()
        self.scenario = _make_scenario()

    def test_add_waypoint_appends_and_syncs_geographic(self) -> None:
        idx = self.editor.add_waypoint(self.scenario, x_m=20.0, y_m=20.0, z_m=10.0)
        self.assertEqual(idx, 3)
        added = self.scenario.waypoints.waypoints[3]
        self.assertEqual(added.x_m, 20.0)
        self.assertEqual(added.y_m, 20.0)
        self.assertEqual(added.z_m, 10.0)
        self.assertIsNotNone(added.lat)
        self.assertIsNotNone(added.lon)
        self.assertNotEqual(added.label, "")

    def test_add_waypoint_via_geographic_coords(self) -> None:
        idx = self.editor.add_waypoint(
            self.scenario, lat=37.001, lon=-122.001, alt_m=15.0, label="Custom"
        )
        added = self.scenario.waypoints.waypoints[idx]
        self.assertEqual(added.label, "Custom")
        self.assertIsNotNone(added.x_m)
        self.assertIsNotNone(added.y_m)
        self.assertEqual(added.z_m, 15.0)

    def test_delete_waypoint_removes_by_index(self) -> None:
        before = list(self.scenario.waypoints.waypoints)
        self.assertTrue(self.editor.delete_waypoint(self.scenario, 1))
        after = self.scenario.waypoints.waypoints
        self.assertEqual(len(after), len(before) - 1)
        self.assertNotIn(before[1], after)

    def test_delete_waypoint_out_of_range_is_safe(self) -> None:
        self.assertFalse(self.editor.delete_waypoint(self.scenario, -1))
        self.assertFalse(self.editor.delete_waypoint(self.scenario, 99))

    def test_move_waypoint_updates_local_and_geographic(self) -> None:
        ok = self.editor.move_waypoint(self.scenario, 0, x_m=5.0, y_m=5.0, z_m=12.0)
        self.assertTrue(ok)
        wp = self.scenario.waypoints.waypoints[0]
        self.assertEqual(wp.x_m, 5.0)
        self.assertEqual(wp.y_m, 5.0)
        self.assertEqual(wp.z_m, 12.0)
        self.assertEqual(wp.alt_m, 12.0)
        self.assertIsNotNone(wp.lat)
        self.assertIsNotNone(wp.lon)

    def test_reorder_waypoint_up_and_down(self) -> None:
        labels_before = [wp.label for wp in self.scenario.waypoints.waypoints]
        self.assertTrue(self.editor.move_waypoint_down(self.scenario, 0))
        labels_after = [wp.label for wp in self.scenario.waypoints.waypoints]
        self.assertEqual(labels_after[0], labels_before[1])
        self.assertEqual(labels_after[1], labels_before[0])

        self.assertTrue(self.editor.move_waypoint_up(self.scenario, 1))
        labels_back = [wp.label for wp in self.scenario.waypoints.waypoints]
        self.assertEqual(labels_back, labels_before)

    def test_reorder_clamps_to_bounds(self) -> None:
        self.assertFalse(self.editor.move_waypoint_up(self.scenario, 0))
        last = len(self.scenario.waypoints.waypoints) - 1
        self.assertFalse(self.editor.move_waypoint_down(self.scenario, last))


class MarkerEditOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.editor = ScenarioEditor()
        self.scenario = _make_scenario()

    def test_add_marker_assigns_default_label(self) -> None:
        idx = self.editor.add_marker(self.scenario, x_m=5.0, y_m=5.0, z_m=3.0)
        self.assertEqual(idx, 2)
        added = self.scenario.markers.markers[2]
        self.assertEqual(added.label, "Marker2")
        self.assertEqual(added.role, "annotation")
        self.assertIsNotNone(added.lat)

    def test_delete_and_reorder_markers(self) -> None:
        self.assertTrue(self.editor.delete_marker(self.scenario, 0))
        self.assertEqual(len(self.scenario.markers.markers), 1)
        self.scenario.markers.markers.append(Marker(label="C", x_m=7.0, y_m=7.0, z_m=1.0))
        self.assertTrue(self.editor.move_marker_up(self.scenario, 1))
        self.assertEqual(self.scenario.markers.markers[0].label, "C")

    def test_update_marker_fields_preserves_role(self) -> None:
        self.assertTrue(
            self.editor.update_marker_fields(
                self.scenario, 0, label="X", color="blue", role="hazard"
            )
        )
        marker = self.scenario.markers.markers[0]
        self.assertEqual(marker.label, "X")
        self.assertEqual(marker.color, "blue")
        self.assertEqual(marker.role, "hazard")


class MarkerTwoStepPlacementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.editor = ScenarioEditor()
        self.scenario = _make_scenario()
        self.state = MarkerPlacementState()

    def test_initial_state_is_awaiting_xy(self) -> None:
        self.assertEqual(self.state.awaiting, "xy")
        self.assertIsNone(self.state.x_m)
        self.assertIsNone(self.state.y_m)

    def test_first_click_locks_xy_and_awaits_z(self) -> None:
        self.editor.start_marker_placement(
            self.state, x_m=2.0, y_m=3.0, z_m=4.0, label="Goal"
        )
        self.assertEqual(self.state.awaiting, "z")
        self.assertEqual(self.state.x_m, 2.0)
        self.assertEqual(self.state.y_m, 3.0)
        self.assertEqual(self.state.z_m, 4.0)
        self.assertEqual(self.state.label, "Goal")

    def test_z_adjustment_keeps_xy_locked(self) -> None:
        self.editor.start_marker_placement(self.state, x_m=1.0, y_m=1.0, z_m=0.0)
        self.editor.adjust_marker_placement_z(self.state, 12.5)
        self.assertEqual(self.state.x_m, 1.0)
        self.assertEqual(self.state.y_m, 1.0)
        self.assertEqual(self.state.z_m, 12.5)

    def test_commit_adds_marker_and_resets(self) -> None:
        self.editor.start_marker_placement(self.state, x_m=4.0, y_m=5.0, z_m=2.0, label="Goal")
        self.editor.adjust_marker_placement_z(self.state, 8.0)
        idx = self.editor.commit_marker_placement(self.scenario, self.state)
        self.assertIsNotNone(idx)
        added = self.scenario.markers.markers[idx]
        self.assertEqual(added.x_m, 4.0)
        self.assertEqual(added.y_m, 5.0)
        self.assertEqual(added.z_m, 8.0)
        self.assertEqual(added.label, "Goal")
        self.assertEqual(self.state.awaiting, "xy")
        self.assertIsNone(self.state.x_m)
        self.assertIsNone(self.state.y_m)

    def test_reset_clears_state(self) -> None:
        self.editor.start_marker_placement(self.state, x_m=1.0, y_m=2.0, z_m=3.0)
        self.state.reset()
        self.assertEqual(self.state.awaiting, "xy")
        self.assertIsNone(self.state.x_m)
        self.assertIsNone(self.state.y_m)


if __name__ == "__main__":
    unittest.main()
