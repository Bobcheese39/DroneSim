"""Tests for the FastAPI web layer (:mod:`dronesim.web`).

These exercise the REST surface against temp-rooted services so they never
touch the repo's real ``scenarios``/``runs``/``maps`` directories or the
network. Map building is redirected to the offline blank-asset builder.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dronesim.models import MapSpec, RunResult, RunSummary, ScenarioSpec, Waypoint, WaypointSet
from dronesim.services.scenario import ScenarioManager
from dronesim.services.scenario_editor import ScenarioEditor
from dronesim.services.terrain import TerrainService
from dronesim.storage import RunStore
from dronesim.web import state as web_state
from dronesim.web.server import app


def _demo_scenario_dict() -> dict:
    scenario = ScenarioSpec(
        name="web-test",
        map=MapSpec(center_lat=37.0, center_lon=-122.0, radius_km=0.5, resolution=32),
        waypoints=WaypointSet(
            waypoints=[
                Waypoint.local(0.0, 0.0, 5.0, label="WP0"),
                Waypoint.local(10.0, 0.0, 5.0, label="WP1"),
            ],
            default_alt_m=5.0,
        ),
    )
    return scenario.to_dict()


class WebApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        st = web_state.AppState()
        st.scenario_manager = ScenarioManager(self.tmp / "scenarios")
        st.terrain_service = TerrainService(self.tmp / "maps")
        st.scenario_editor = ScenarioEditor(terrain_service=st.terrain_service)
        st.run_store = RunStore(self.tmp / "runs")
        # Offline map build: blank asset (no network), still a real MapAsset.
        st.terrain_service.fetch_map = lambda spec, **kw: st.terrain_service.build_blank_asset(spec)
        web_state._STATE = st
        self.state = st
        self.client = TestClient(app)

    def tearDown(self) -> None:
        web_state._STATE = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- basics ---------------------------------------------------------
    def test_index_and_static(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/css/app.css").status_code, 200)
        self.assertEqual(self.client.get("/js/main.js").status_code, 200)

    def test_backends(self) -> None:
        rows = self.client.get("/api/backends").json()
        self.assertTrue(any(r["backend_id"] == "inhouse_mpc_quad" for r in rows))

    # -- scenario CRUD + validate --------------------------------------
    def test_validate_and_save_and_list(self) -> None:
        d = _demo_scenario_dict()
        v = self.client.post("/api/scenarios/validate", json=d).json()
        self.assertTrue(v["ok"])
        self.assertEqual(v["messages"], [])

        saved = self.client.post("/api/scenarios", json=d)
        self.assertEqual(saved.status_code, 200)
        sid = saved.json()["scenario"]["scenario_id"]

        listing = self.client.get("/api/scenarios").json()
        self.assertTrue(any(s["scenario_id"] == sid for s in listing))

        fetched = self.client.get(f"/api/scenarios/{sid}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["scenario_id"], sid)

    def test_validate_reports_errors(self) -> None:
        d = _demo_scenario_dict()
        d["waypoints"]["waypoints"] = d["waypoints"]["waypoints"][:1]  # too few
        v = self.client.post("/api/scenarios/validate", json=d).json()
        self.assertFalse(v["ok"])
        self.assertTrue(v["messages"])

    # -- map build + imagery -------------------------------------------
    def test_map_build_and_imagery(self) -> None:
        d = _demo_scenario_dict()
        resp = self.client.post("/api/map/build", json={"map": d["map"], "fetch_remote": False})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("key", body)
        self.assertTrue(body["imagery_url"].startswith("/api/map/imagery/"))
        self.assertTrue(body["heightmap_url"].startswith("/api/map/heightmap/"))
        self.assertIn("heightmap", body)
        hm = body["heightmap"]
        self.assertIn("width", hm)
        self.assertIn("height", hm)
        self.assertIn("height_offset", hm)
        self.assertIn("height_scale", hm)

        img = self.client.get(body["imagery_url"])
        self.assertEqual(img.status_code, 200)
        self.assertEqual(img.headers["content-type"], "image/png")

        hm_resp = self.client.get(body["heightmap_url"])
        self.assertEqual(hm_resp.status_code, 200)
        self.assertEqual(hm_resp.headers["content-type"], "application/octet-stream")
        self.assertEqual(len(hm_resp.content), hm["width"] * hm["height"] * 2)

    # -- editing --------------------------------------------------------
    def test_edit_add_move_delete_reorder(self) -> None:
        d = _demo_scenario_dict()

        added = self.client.post(
            "/api/edit/add-waypoint", json={"scenario": d, "x_m": 5.0, "y_m": 5.0, "z_m": 6.0}
        ).json()
        self.assertEqual(added["index"], 2)
        scen = added["scenario"]
        self.assertEqual(len(scen["waypoints"]["waypoints"]), 3)
        # local->geo sync happened server-side
        self.assertIsNotNone(scen["waypoints"]["waypoints"][2]["lat"])

        moved = self.client.post(
            "/api/edit/move",
            json={"scenario": scen, "kind": "waypoint", "index": 2, "x_m": 9.0, "y_m": 9.0, "z_m": 7.0},
        ).json()
        self.assertTrue(moved["ok"])
        self.assertEqual(moved["scenario"]["waypoints"]["waypoints"][2]["x_m"], 9.0)

        reordered = self.client.post(
            "/api/edit/reorder",
            json={"scenario": moved["scenario"], "kind": "waypoint", "index": 2, "direction": "up"},
        ).json()
        self.assertTrue(reordered["ok"])
        self.assertEqual(reordered["new_index"], 1)

        deleted = self.client.post(
            "/api/edit/delete",
            json={"scenario": reordered["scenario"], "kind": "waypoint", "index": 1},
        ).json()
        self.assertTrue(deleted["ok"])
        self.assertEqual(len(deleted["scenario"]["waypoints"]["waypoints"]), 2)

    def test_add_marker_geographic(self) -> None:
        d = _demo_scenario_dict()
        resp = self.client.post(
            "/api/edit/add-marker",
            json={"scenario": d, "lat": 37.001, "lon": -122.001, "alt_m": 3.0, "label": "T"},
        ).json()
        marker = resp["scenario"]["markers"]["markers"][resp["index"]]
        self.assertIsNotNone(marker["x_m"])  # geo->local sync happened
        self.assertEqual(marker["label"], "T")

    # -- run listing + result serialization ----------------------------
    def test_run_result_serialization(self) -> None:
        # Save a scenario so the result endpoint can resolve its map center.
        d = _demo_scenario_dict()
        saved = self.client.post("/api/scenarios", json=d).json()
        sid = saved["scenario"]["scenario_id"]

        run = RunResult(
            run_id="run_testabc",
            scenario_id=sid,
            backend_id="inhouse_mpc_quad",
            model_id="m",
            status="completed",
            time_s=[0.0, 0.1, 0.2],
            position_m=[[0, 0, 5], [1, 0, 5], [2, 0, 5]],
            velocity_mps=[[0, 0, 0], [1, 0, 0], [1, 0, 0]],
            reference_position_m=[[0, 0, 5], [1, 0, 5], [2, 0, 5]],
            tracking_error_m=[0.0, 0.1, 0.05],
            summary=RunSummary(success=True, miss_distance_m=0.05, duration_s=0.2),
        )
        run_dir = self.state.run_store.save(run)

        runs = self.client.get("/api/runs").json()
        self.assertTrue(any(r["run_id"] == "run_testabc" for r in runs))

        result = self.client.get("/api/runs/result", params={"path": str(run_dir)})
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["run"]["run_id"], "run_testabc")
        analysis = body["analysis"]
        self.assertEqual(analysis["time_s"], [0.0, 0.1, 0.2])
        self.assertEqual(len(analysis["error_decomposition"]["ex"]), 3)
        self.assertTrue(any(row["metric"] == "miss_distance_m" for row in analysis["summary"]))
        self.assertEqual(body["center"]["lat"], 37.0)


if __name__ == "__main__":
    unittest.main()
