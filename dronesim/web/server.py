"""FastAPI application for DroneSim.

Wraps the existing domain services in REST + WebSocket endpoints and serves
the static vanilla-JS frontend.
"""
from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from dronesim.config.jsbsim_aircraft import apply_jsbsim_aircraft, list_jsbsim_aircraft
from dronesim.config.jsbsim_presets import apply_jsbsim_preset, list_jsbsim_presets
from dronesim.models import MapSpec, RunConfig, ScenarioSpec
from dronesim.services.terrain import (
    MapAsset,
    MapCacheMiss,
    bounding_box,
    compute_run_clearance_m,
    encode_cesium_heightmap,
    local_to_lat_lon,
)
from dronesim.web import analysis as analysis_mod
from dronesim.web import run_session
from dronesim.web.serialization import scenario_from_payload
from dronesim.web.state import get_state

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="DroneSim", version="0.2.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_spec_from_payload(payload: dict[str, Any]) -> MapSpec:
    fields = {
        k: v
        for k, v in payload.items()
        if k in MapSpec.__dataclass_fields__  # type: ignore[attr-defined]
    }
    return MapSpec(**fields)


def _extent_m(asset: MapAsset) -> dict[str, float]:
    """East/north span of a map asset in local meters (for the XYZ engine)."""
    x = asset.x_grid_m
    y = asset.y_grid_m
    return {
        "width": float(x.max() - x.min()),
        "height": float(y.max() - y.min()),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


def _cache_miss_payload(exc: MapCacheMiss) -> dict[str, Any]:
    spec = exc.spec
    return {
        "error": "map_cache_miss",
        "message": str(exc),
        "requested": {
            "center_lat": spec.center_lat,
            "center_lon": spec.center_lon,
            "radius_km": spec.radius_km,
            "resolution": spec.resolution,
        },
        "missing_tiles": exc.missing_tiles,
        "alternatives": exc.available_caches,
    }


# ---------------------------------------------------------------------------
# Backends + scenarios
# ---------------------------------------------------------------------------


@app.get("/api/backends")
def list_backends() -> list[dict[str, str]]:
    return get_state().factory.available()


@app.get("/api/jsbsim/presets")
def jsbsim_presets() -> dict[str, Any]:
    return {"presets": list_jsbsim_presets()}


@app.post("/api/jsbsim/apply-preset")
def jsbsim_apply_preset(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    preset_id = payload.get("preset_id")
    if not preset_id:
        raise HTTPException(status_code=400, detail="preset_id is required")
    scenario = payload.get("scenario") or {}
    try:
        return apply_jsbsim_preset(scenario, str(preset_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jsbsim/aircraft")
def jsbsim_aircraft() -> dict[str, Any]:
    return {"aircraft": list_jsbsim_aircraft()}


@app.post("/api/jsbsim/apply-aircraft")
def jsbsim_apply_aircraft(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    aircraft_id = payload.get("aircraft_id")
    if not aircraft_id:
        raise HTTPException(status_code=400, detail="aircraft_id is required")
    scenario = payload.get("scenario") or {}
    try:
        return apply_jsbsim_aircraft(scenario, str(aircraft_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    rows = get_state().scenario_manager.list_scenarios()
    return [
        {
            "scenario_id": s.scenario_id,
            "name": s.name,
            "description": s.description,
            "n_waypoints": len(s.waypoints.waypoints),
            "n_markers": len(s.markers.markers),
            "updated_utc": s.updated_utc,
        }
        for s in rows
    ]


@app.get("/api/scenarios/default")
def default_scenario() -> dict[str, Any]:
    return get_state().scenario_manager.default_scenario("DroneSim Demo").to_dict()


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> dict[str, Any]:
    try:
        return get_state().scenario_manager.load(scenario_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}") from exc


@app.post("/api/scenarios/validate")
def validate_scenario(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        scenario = scenario_from_payload(payload, validate=False)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "messages": [str(exc)]}
    messages = get_state().scenario_manager.validate(scenario)
    return {"ok": not messages, "messages": messages}


@app.post("/api/scenarios")
def save_scenario(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    try:
        scenario = scenario_from_payload(payload, validate=False)
        messages = state.scenario_manager.validate(scenario)
        if messages:
            raise HTTPException(status_code=422, detail={"messages": messages})
        path = state.scenario_manager.save(scenario)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scenario": scenario.to_dict(), "path": str(path)}


@app.post("/api/scenarios/{scenario_id}/duplicate")
def duplicate_scenario(scenario_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    state = get_state()
    try:
        scenario = scenario_from_payload(payload, validate=False) if payload else state.scenario_manager.load(scenario_id)
        duplicate = state.scenario_manager.duplicate(scenario, name=payload.get("name") if payload else None)
        path = state.scenario_manager.save(duplicate)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scenario": duplicate.to_dict(), "path": str(path)}


# ---------------------------------------------------------------------------
# Map / terrain
# ---------------------------------------------------------------------------


@app.post("/api/map/build")
def build_map(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    map_payload = payload.get("map", payload)
    fetch_remote = bool(payload.get("fetch_remote", False))
    try:
        spec = _map_spec_from_payload(map_payload)
        spec.validate()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        asset = state.terrain_service.fetch_map(spec, fetch_remote=fetch_remote)
    except MapCacheMiss as exc:
        return JSONResponse(status_code=409, content=_cache_miss_payload(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    key = state.cache_map_asset(asset)
    lat_min, lon_min, lat_max, lon_max = asset.bounds
    hm = encode_cesium_heightmap(
        asset.elevation_m,
        vertical_exaggeration=spec.vertical_exaggeration,
    )
    return {
        "key": key,
        "origin": asset.origin,
        "center_lat": spec.center_lat,
        "center_lon": spec.center_lon,
        "vertical_exaggeration": spec.vertical_exaggeration,
        "bounds": {"west": lon_min, "east": lon_max, "south": lat_min, "north": lat_max},
        "extent_m": _extent_m(asset),
        "imagery_url": f"/api/map/imagery/{key}.png",
        "heightmap_url": f"/api/map/heightmap/{key}.bin",
        "heightmap": {
            "width": hm["width"],
            "height": hm["height"],
            "height_offset": hm["height_offset"],
            "height_scale": hm["height_scale"],
        },
    }


@app.get("/api/map/caches")
def list_map_caches() -> list[dict[str, Any]]:
    """List processed map caches available on disk for the XYZ map picker."""
    return get_state().terrain_service._list_available_caches()


def _decode_elevation_upload(filename: str, data: bytes) -> np.ndarray:
    """Decode an uploaded elevation file (.npy or grayscale image) to meters."""
    name = (filename or "").lower()
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(data))
        return np.asarray(arr, dtype=np.float32)
    with Image.open(io.BytesIO(data)) as img:
        gray = img.convert("F")
        return np.asarray(gray, dtype=np.float32)


@app.post("/api/map/upload")
async def upload_map(
    imagery: UploadFile = File(...),
    elevation: UploadFile | None = File(default=None),
    scale_m: float = Form(default=200.0),
) -> dict[str, Any]:
    """Build a synthetic local-meters map asset from uploaded files.

    The map is centered at the origin (0, 0) and spans ``scale_m`` meters in the
    east-west direction; the north-south span follows the elevation/imagery
    aspect ratio. Returns the same shape as ``/api/map/build``.
    """
    state = get_state()
    try:
        sat = Image.open(io.BytesIO(await imagery.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad imagery file: {exc}") from exc

    if elevation is not None:
        try:
            elevation_m = _decode_elevation_upload(elevation.filename, await elevation.read())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Bad elevation file: {exc}") from exc
        if elevation_m.ndim != 2:
            raise HTTPException(status_code=400, detail="Elevation must be a 2D grid")
    else:
        elevation_m = np.zeros((sat.height, sat.width), dtype=np.float32)

    if scale_m <= 0:
        raise HTTPException(status_code=400, detail="scale_m must be positive")

    rows, cols = elevation_m.shape
    if rows < 2 or cols < 2:
        raise HTTPException(status_code=400, detail="Elevation grid too small")

    width_m = float(scale_m)
    height_m = width_m * (rows / cols)
    half_w = width_m / 2.0
    half_h = height_m / 2.0

    xs = np.linspace(-half_w, half_w, cols)
    ys = np.linspace(half_h, -half_h, rows)  # row 0 = north edge
    x_grid, y_grid = np.meshgrid(xs, ys)
    lat_grid, lon_grid = local_to_lat_lon(x_grid, y_grid, 0.0, 0.0)

    radius_km = max(half_w, half_h) / 1000.0
    key = f"upload_{uuid.uuid4().hex[:12]}"
    spec = MapSpec(
        center_lat=0.0,
        center_lon=0.0,
        radius_km=max(radius_km, 0.001),
        resolution=max(cols, 16),
        name="uploaded_map",
        cache_key=key,
    )
    bounds = (
        float(lat_grid.min()),
        float(lon_grid.min()),
        float(lat_grid.max()),
        float(lon_grid.max()),
    )
    asset = MapAsset(
        spec=spec,
        bounds=bounds,
        zoom=0,
        satellite=sat,
        elevation_m=elevation_m,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        x_grid_m=x_grid,
        y_grid_m=y_grid,
        cache_dir=state.terrain_service.cache_root / key,
        origin="upload",
    )
    state.cache_map_asset(asset)

    hm = encode_cesium_heightmap(elevation_m, vertical_exaggeration=spec.vertical_exaggeration)
    lat_min, lon_min, lat_max, lon_max = bounds
    return {
        "key": key,
        "origin": "upload",
        "center_lat": 0.0,
        "center_lon": 0.0,
        "vertical_exaggeration": spec.vertical_exaggeration,
        "bounds": {"west": lon_min, "east": lon_max, "south": lat_min, "north": lat_max},
        "extent_m": _extent_m(asset),
        "imagery_url": f"/api/map/imagery/{key}.png",
        "heightmap_url": f"/api/map/heightmap/{key}.bin",
        "heightmap": {
            "width": hm["width"],
            "height": hm["height"],
            "height_offset": hm["height_offset"],
            "height_scale": hm["height_scale"],
        },
    }


@app.get("/api/map/heightmap/{key}.bin")
def map_heightmap(key: str) -> StreamingResponse:
    asset = get_state().get_map_asset(key)
    if asset is None:
        raise HTTPException(status_code=404, detail="Map heightmap not built/cached for this key")
    hm = encode_cesium_heightmap(
        asset.elevation_m,
        vertical_exaggeration=asset.spec.vertical_exaggeration,
    )
    buf = io.BytesIO(hm["buffer"])
    return StreamingResponse(buf, media_type="application/octet-stream")


@app.get("/api/map/imagery/{key}.png")
def map_imagery(key: str) -> StreamingResponse:
    asset = get_state().get_map_asset(key)
    if asset is None:
        raise HTTPException(status_code=404, detail="Map imagery not built/cached for this key")
    buf = io.BytesIO()
    asset.satellite.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------------------
# Scenario editing (stateless; coordinate sync stays server-side)
# ---------------------------------------------------------------------------


def _edit_scenario(payload: dict[str, Any]) -> ScenarioSpec:
    scenario_payload = payload.get("scenario")
    if not isinstance(scenario_payload, dict):
        raise HTTPException(status_code=400, detail="Request must include a 'scenario' object")
    try:
        return scenario_from_payload(scenario_payload, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad scenario: {exc}") from exc


@app.post("/api/edit/add-waypoint")
def add_waypoint(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    scenario = _edit_scenario(payload)
    try:
        index = state.scenario_editor.add_waypoint(
            scenario,
            x_m=payload.get("x_m"),
            y_m=payload.get("y_m"),
            z_m=payload.get("z_m"),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
            alt_m=payload.get("alt_m"),
            label=payload.get("label"),
            index=payload.get("at_index"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scenario": scenario.to_dict(), "index": index}


@app.post("/api/edit/add-marker")
def add_marker(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    scenario = _edit_scenario(payload)
    try:
        index = state.scenario_editor.add_marker(
            scenario,
            x_m=payload.get("x_m"),
            y_m=payload.get("y_m"),
            z_m=payload.get("z_m"),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
            alt_m=payload.get("alt_m"),
            label=payload.get("label"),
            color=payload.get("color", "yellow"),
            size=float(payload.get("size", 10.0)),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scenario": scenario.to_dict(), "index": index}


@app.post("/api/edit/move")
def move_entity(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    scenario = _edit_scenario(payload)
    kind = payload.get("kind")
    index = int(payload.get("index", -1))
    x_m, y_m, z_m = payload.get("x_m"), payload.get("y_m"), payload.get("z_m")
    if kind == "waypoint":
        ok = state.scenario_editor.move_waypoint(scenario, index, x_m=x_m, y_m=y_m, z_m=z_m)
    elif kind == "marker":
        ok = state.scenario_editor.move_marker(scenario, index, x_m=x_m, y_m=y_m, z_m=z_m)
    else:
        raise HTTPException(status_code=400, detail="kind must be 'waypoint' or 'marker'")
    return {"scenario": scenario.to_dict(), "ok": ok}


@app.post("/api/edit/delete")
def delete_entity(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    scenario = _edit_scenario(payload)
    kind = payload.get("kind")
    index = int(payload.get("index", -1))
    if kind == "waypoint":
        ok = state.scenario_editor.delete_waypoint(scenario, index)
    elif kind == "marker":
        ok = state.scenario_editor.delete_marker(scenario, index)
    else:
        raise HTTPException(status_code=400, detail="kind must be 'waypoint' or 'marker'")
    return {"scenario": scenario.to_dict(), "ok": ok}


@app.post("/api/edit/reorder")
def reorder_entity(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    scenario = _edit_scenario(payload)
    kind = payload.get("kind")
    index = int(payload.get("index", -1))
    direction = payload.get("direction", "up")
    new_index = index - 1 if direction == "up" else index + 1
    if kind == "waypoint":
        ok = state.scenario_editor.reorder_waypoint(scenario, index, new_index)
    elif kind == "marker":
        ok = state.scenario_editor.reorder_marker(scenario, index, new_index)
    else:
        raise HTTPException(status_code=400, detail="kind must be 'waypoint' or 'marker'")
    return {"scenario": scenario.to_dict(), "ok": ok, "new_index": new_index if ok else index}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs(scenario_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
    return get_state().run_store.list_run_summaries(scenario_id)


@app.get("/api/runs/result")
def run_result(path: str = Query(...)) -> dict[str, Any]:
    state = get_state()
    try:
        run = state.run_store.load(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Could not load run: {exc}") from exc

    clearance: list[float] | None = None
    center = {"lat": None, "lon": None}
    try:
        scenario = state.scenario_manager.load(run.scenario_id)
        center = {"lat": scenario.map.center_lat, "lon": scenario.map.center_lon}
        try:
            asset = state.terrain_service.fetch_map(scenario.map, fetch_remote=False)
            clearance = compute_run_clearance_m(run, asset)
        except Exception:  # noqa: BLE001 - terrain optional for analysis
            clearance = None
    except Exception:  # noqa: BLE001 - scenario may be gone
        scenario = None

    return {
        "run": run.to_dict(),
        "analysis": analysis_mod.analysis_block(run, clearance),
        "center": center,
    }


@app.post("/api/runs/start")
def start_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    state = get_state()
    scenario_payload = payload.get("scenario")
    if not isinstance(scenario_payload, dict):
        raise HTTPException(status_code=400, detail="Request must include a 'scenario' object")
    try:
        scenario = scenario_from_payload(scenario_payload, validate=False)
        messages = state.scenario_manager.validate(scenario)
        if messages:
            raise HTTPException(status_code=422, detail={"messages": messages})
        state.scenario_manager.save(scenario)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rc_payload = payload.get("run_config") or scenario_payload.get("run_config")
    if rc_payload:
        rc_fields = {
            k: v
            for k, v in rc_payload.items()
            if k in RunConfig.__dataclass_fields__  # type: ignore[attr-defined]
        }
        run_config = RunConfig(**rc_fields)
    else:
        run_config = scenario.run_config

    session = run_session.registry().create()
    mc = run_config.monte_carlo or {}
    n_trials = int(mc.get("n_trials") or 1)
    if n_trials > 1:
        run_session.start_monte_carlo_run(
            session,
            scenario,
            run_config,
            state.run_store,
            n_trials=n_trials,
            workers=int(mc.get("workers") or 1),
            base_seed=int(mc.get("base_seed") if mc.get("base_seed") is not None else run_config.seed or 0),
        )
    else:
        run_session.start_single_run(session, scenario, run_config, state.run_store)
    return {"run_token": session.token}


@app.post("/api/runs/cancel")
def cancel_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    token = payload.get("run_token")
    if not token:
        raise HTTPException(status_code=400, detail="run_token required")
    session = run_session.registry().get(str(token))
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown run token")
    session.manager.cancel()
    return {"ok": True}


@app.post("/api/runs/mc-analysis")
def mc_analysis(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="Request must include a non-empty 'paths' list")
    state = get_state()
    runs = []
    for path in paths:
        try:
            runs.append(state.run_store.load(str(path)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"Could not load run: {path}: {exc}") from exc
    return analysis_mod.mc_analysis_block(runs)


@app.post("/api/runs/mc-replay")
def mc_replay(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="Request must include a non-empty 'paths' list")
    state = get_state()
    runs = []
    for path in paths:
        try:
            runs.append(state.run_store.load(str(path)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"Could not load run: {path}: {exc}") from exc

    center: dict[str, float | None] = {"lat": None, "lon": None}
    try:
        scenario = state.scenario_manager.load(runs[0].scenario_id)
        center = {"lat": scenario.map.center_lat, "lon": scenario.map.center_lon}
    except Exception:  # noqa: BLE001 - scenario may be gone
        pass
    return analysis_mod.mc_replay_block(runs, center=center)


@app.websocket("/ws/run/{token}")
async def run_ws(websocket: WebSocket, token: str) -> None:
    await websocket.accept()
    session = run_session.registry().get(token)
    if session is None:
        await websocket.send_json({"type": "error", "message": "Unknown run token"})
        await websocket.close()
        return
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                event = await loop.run_in_executor(None, session.events.get, True, 0.5)
            except Exception:  # queue.Empty after timeout
                if session.finished.is_set() and session.events.empty():
                    break
                continue
            await websocket.send_json(event)
            if event.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        run_session.registry().remove(token)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
