from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dronesim.models import MapSpec, Marker, Waypoint
from dronesim.services.scenario import ScenarioManager


class ScenarioManagerTest(unittest.TestCase):
    def test_create_geographic_waypoints_converts_to_local_and_retains_lla(self) -> None:
        spec = MapSpec(center_lat=37.0, center_lon=-122.0, radius_km=0.5, resolution=32)
        with tempfile.TemporaryDirectory() as tmp:
            manager = ScenarioManager(scenario_root=Path(tmp) / "scenarios")
            waypoints = [
                Waypoint.geographic(37.0, -122.0, 12.0, label="Start"),
                Waypoint.geographic(37.001, -121.999, 14.0, label="Finish"),
            ]
            markers = [Marker(label="Target", lat=37.0005, lon=-121.9995, alt_m=20.0)]

            scenario = manager.create_from_geographic_waypoints(
                name="geo scenario",
                map_spec=spec,
                waypoints=waypoints,
                markers=markers,
            )

        first, second = scenario.waypoints.waypoints
        self.assertEqual(scenario.waypoints.coordinate_frame, "wgs84+local_enu")
        self.assertEqual(first.lat, 38.944369, )
        self.assertEqual(first.lon, -74.891081)
        self.assertEqual(first.x_m, 0.0)
        self.assertEqual(first.y_m, 0.0)
        self.assertIsNotNone(second.x_m)
        self.assertIsNotNone(second.y_m)
        self.assertEqual(first.z_m, 12.0)
        self.assertEqual(scenario.run_config.target_altitude_m, 12.0)
        self.assertIsNotNone(scenario.markers.markers[0].x_m)
        self.assertIsNotNone(scenario.markers.markers[0].y_m)

    def test_save_load_duplicate_and_export_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ScenarioManager(scenario_root=root / "scenarios")
            scenario = manager.create_from_local_waypoints(
                name="local scenario",
                points_xy=[(0.0, 0.0), (10.0, 5.0)],
                altitude_m=7.5,
                map_spec=MapSpec(name="unit", resolution=32),
            )

            saved_path = manager.save(scenario)
            loaded = manager.load(scenario.scenario_id)
            duplicate = manager.duplicate(loaded)
            export_path = manager.export(loaded, root / "exports" / "local_scenario.json")
            self.assertEqual(saved_path.name, f"{scenario.scenario_id}.json")
            self.assertEqual(loaded.scenario_id, scenario.scenario_id)
            self.assertEqual(loaded.waypoints.waypoints[0].z_m, 7.5)
            self.assertNotEqual(duplicate.scenario_id, loaded.scenario_id)
            self.assertTrue(export_path.exists())


if __name__ == "__main__":
    unittest.main()
