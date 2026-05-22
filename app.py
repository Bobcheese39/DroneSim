"""DroneSim Panel application.

Run with:

    panel serve dronesim/gui/app.py --show --autoreload
"""
from __future__ import annotations

import datetime as _dt
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from dronesim.models import (
    DroneModelSpec,
    MapSpec,
    Marker,
    RunConfig,
    RunResult,
    ScenarioSpec,
    Waypoint,
    WaypointSet,
    MarkerSet,
)
from dronesim.services.scenario import ScenarioManager
from dronesim.services.terrain import TerrainService, build_terrain_figure
from dronesim.sim import BackendUnavailable, SimulationManager
from dronesim.storage import RunStore

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


class ConsoleLog:
    def __init__(self, max_lines: int = 300) -> None:
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.pane = pn.pane.HTML("<div class='console-log'></div>", sizing_mode="stretch_width")

    def write(self, msg: str, level: str = "ok") -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        color = {"ok": "#39ff14", "warn": "#ffb454", "err": "#ff3860"}.get(level, "#c9d1d9")
        self.lines.append(f"<span style='color:#8b949e'>[{ts}]</span> <span style='color:{color}'>{msg}</span>")
        self.lines = self.lines[-self.max_lines :]
        self.pane.object = "<div class='console-log'>" + "<br/>".join(self.lines) + "</div>"


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", title=title, height=520)
    return fig


def _parse_waypoints(text: str, altitude_m: float) -> list[Waypoint]:
    waypoints: list[Waypoint] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Waypoint line {line_no} needs at least x,y")
        x_m = float(parts[0])
        y_m = float(parts[1])
        z_m = float(parts[2]) if len(parts) >= 3 and parts[2] else altitude_m
        waypoints.append(Waypoint.local(x_m, y_m, z_m, label=f"WP{len(waypoints)}"))
    if len(waypoints) < 2:
        raise ValueError("Enter at least two waypoints")
    return waypoints


def _markers_from_table(df: pd.DataFrame) -> list[Marker]:
    markers: list[Marker] = []
    if df is None or df.empty:
        return markers
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        markers.append(
            Marker(
                label=label,
                x_m=float(row.get("x_m", 0.0)),
                y_m=float(row.get("y_m", 0.0)),
                z_m=float(row.get("z_m", row.get("alt_m", 0.0))),
                alt_m=float(row.get("alt_m", row.get("z_m", 0.0))),
                color=str(row.get("color", "red") or "red"),
                size=float(row.get("size", 10.0)),
                notes=str(row.get("notes", "") or ""),
            )
        )
    return markers


def _summary_frame(run: RunResult | None) -> pd.DataFrame:
    if run is None:
        return pd.DataFrame()
    summary = run.summary
    return pd.DataFrame(
        [
            {"metric": "status", "value": run.status},
            {"metric": "success", "value": summary.success},
            {"metric": "miss_distance_m", "value": summary.miss_distance_m},
            {"metric": "duration_s", "value": summary.duration_s},
            {"metric": "settle_steps", "value": summary.settle_steps},
            {"metric": "max_tracking_error_m", "value": summary.max_tracking_error_m},
            {"metric": "mean_tracking_error_m", "value": summary.mean_tracking_error_m},
            {"metric": "max_altitude_m", "value": summary.max_altitude_m},
            {"metric": "min_altitude_m", "value": summary.min_altitude_m},
            {"metric": "wallclock_s", "value": summary.wallclock_s},
        ]
    )


def _analysis_figure(run: RunResult | None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", title="Tracking Error", height=360)
    if run is None or not run.time_s:
        return fig
    fig.add_trace(
        go.Scatter(
            x=run.time_s,
            y=run.tracking_error_m,
            mode="lines",
            name="tracking error",
            line=dict(color="#39ff14"),
        )
    )
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Error (m)")
    return fig


def build_app() -> pn.template.FastListTemplate:
    scenario_manager = ScenarioManager()
    terrain_service = TerrainService()
    sim_manager = SimulationManager()
    run_store = RunStore()
    console = ConsoleLog()
    events: "queue.Queue[tuple[str, object]]" = queue.Queue()

    default_scenario = scenario_manager.default_scenario("DroneSim Demo")
    state: dict[str, object] = {
        "scenario": default_scenario,
        "map_asset": terrain_service.build_blank_asset(default_scenario.map),
        "last_run": None,
        "worker": None,
    }

    backend_options = {b["display_name"]: b["backend_id"] for b in sim_manager.factory.available()}

    scenario_name = pn.widgets.TextInput(name="Scenario name", value=default_scenario.name)
    scenario_description = pn.widgets.TextAreaInput(
        name="Description", value=default_scenario.description, height=80
    )
    scenario_select = pn.widgets.Select(name="Load scenario", options=[""])
    refresh_scenarios_btn = pn.widgets.Button(name="Refresh scenarios", button_type="default")
    load_scenario_btn = pn.widgets.Button(name="Load selected", button_type="primary")

    lat_input = pn.widgets.FloatInput(name="Map center lat", value=default_scenario.map.center_lat)
    lon_input = pn.widgets.FloatInput(name="Map center lon", value=default_scenario.map.center_lon)
    radius_input = pn.widgets.FloatInput(name="Radius (km)", value=default_scenario.map.radius_km, step=0.1)
    resolution_input = pn.widgets.IntInput(name="Resolution", value=default_scenario.map.resolution, step=50)
    fetch_remote_toggle = pn.widgets.Checkbox(name="Fetch remote imagery/elevation", value=False)
    build_map_btn = pn.widgets.Button(name="Build map preview", button_type="primary")

    backend_select = pn.widgets.Select(name="Backend", options=backend_options, value="inhouse_mpc_quad")
    altitude_input = pn.widgets.FloatInput(name="Target altitude (m)", value=5.0, step=0.5)
    dt_input = pn.widgets.FloatInput(name="dt (s)", value=0.1, step=0.01)
    horizon_input = pn.widgets.IntInput(name="MPC horizon", value=20, step=1)
    max_steps_input = pn.widgets.IntInput(name="Max steps", value=250, step=10)
    lookahead_input = pn.widgets.IntInput(name="Spline lookahead", value=60, step=5)
    seed_input = pn.widgets.IntInput(name="Seed", value=0, step=1)

    waypoint_text = pn.widgets.TextAreaInput(
        name="Waypoints: x,y,z per line in local meters",
        value="0,0,5\n1,2,5\n2,4.5,5\n3,3,5",
        height=180,
    )
    marker_table = pn.widgets.Tabulator(
        value=pd.DataFrame(
            [{"label": "Launch", "x_m": 0.0, "y_m": 0.0, "z_m": 5.0, "color": "red", "size": 10.0, "notes": ""}]
        ),
        show_index=False,
        height=180,
        theme="midnight",
    )

    save_scenario_btn = pn.widgets.Button(name="Save scenario", button_type="success")
    run_btn = pn.widgets.Button(name="Run scenario", button_type="success")
    progress = pn.indicators.Progress(name="Run progress", value=0, max=1, bar_color="success")
    status_md = pn.pane.Markdown("**Idle.**", styles={"color": "#39ff14"})

    map_pane = pn.pane.Plotly(build_terrain_figure(state["map_asset"]), height=580)
    replay_pane = pn.pane.Plotly(_empty_figure("Replay"), height=580)
    summary_table = pn.widgets.Tabulator(value=pd.DataFrame(), show_index=False, height=280, theme="midnight")
    analysis_pane = pn.pane.Plotly(_analysis_figure(None), height=380)
    runs_table = pn.widgets.Tabulator(value=pd.DataFrame(), show_index=False, height=360, theme="midnight")
    refresh_runs_btn = pn.widgets.Button(name="Refresh runs", button_type="default")

    def _refresh_scenario_options() -> None:
        rows = scenario_manager.list_scenarios()
        scenario_select.options = [""] + [f"{s.name} ({s.scenario_id})" for s in rows]
        state["scenario_options"] = {f"{s.name} ({s.scenario_id})": s.scenario_id for s in rows}

    def _current_map_spec() -> MapSpec:
        return MapSpec(
            name=scenario_name.value.strip() or "map",
            center_lat=float(lat_input.value),
            center_lon=float(lon_input.value),
            radius_km=float(radius_input.value),
            resolution=int(resolution_input.value),
        )

    def _build_scenario_from_widgets() -> ScenarioSpec:
        map_spec = _current_map_spec()
        waypoints = _parse_waypoints(waypoint_text.value, float(altitude_input.value))
        markers = _markers_from_table(marker_table.value)
        run_config = RunConfig(
            backend_id=str(backend_select.value),
            dt_s=float(dt_input.value),
            max_steps=int(max_steps_input.value),
            target_altitude_m=float(altitude_input.value),
            horizon=int(horizon_input.value),
            lookahead=int(lookahead_input.value),
            seed=int(seed_input.value),
        )
        scenario = ScenarioSpec(
            name=scenario_name.value.strip() or "Untitled Scenario",
            description=scenario_description.value,
            map=map_spec,
            waypoints=WaypointSet(waypoints=waypoints, default_alt_m=float(altitude_input.value)),
            markers=MarkerSet(markers=markers),
            vehicle=DroneModelSpec(backend_id=str(backend_select.value)),
            run_config=run_config,
        )
        scenario.validate()
        return scenario

    def _set_widgets_from_scenario(scenario: ScenarioSpec) -> None:
        scenario_name.value = scenario.name
        scenario_description.value = scenario.description
        lat_input.value = scenario.map.center_lat
        lon_input.value = scenario.map.center_lon
        radius_input.value = scenario.map.radius_km
        resolution_input.value = scenario.map.resolution
        backend_select.value = scenario.run_config.backend_id
        altitude_input.value = scenario.run_config.target_altitude_m
        dt_input.value = scenario.run_config.dt_s
        horizon_input.value = scenario.run_config.horizon
        max_steps_input.value = scenario.run_config.max_steps
        lookahead_input.value = scenario.run_config.lookahead
        seed_input.value = scenario.run_config.seed or 0
        waypoint_text.value = "\n".join(
            f"{wp.x_m or 0.0},{wp.y_m or 0.0},{wp.z_m if wp.z_m is not None else wp.alt_m}"
            for wp in scenario.waypoints.waypoints
        )
        marker_table.value = pd.DataFrame([m.__dict__ for m in scenario.markers.markers])

    def _refresh_runs_table() -> None:
        rows = []
        for path in run_store.list_runs():
            try:
                run = run_store.load(path)
            except Exception:
                continue
            rows.append({
                "run_id": run.run_id,
                "scenario_id": run.scenario_id,
                "backend": run.backend_id,
                "status": run.status,
                "success": run.summary.success,
                "miss_m": run.summary.miss_distance_m,
                "created_utc": run.created_utc,
                "path": str(path.parent),
            })
        runs_table.value = pd.DataFrame(rows)

    def _update_replay(run: RunResult | None = None) -> None:
        scenario = state["scenario"]
        asset = state["map_asset"]
        if not isinstance(scenario, ScenarioSpec):
            return
        trajectory = None
        if run is not None and run.position_m:
            trajectory = np.asarray(run.position_m, dtype=float)
        replay_pane.object = build_terrain_figure(
            asset,
            trajectory_xyz=trajectory,
            waypoints=scenario.waypoints.waypoints,
            markers=scenario.markers.markers,
        )
        summary_table.value = _summary_frame(run)
        analysis_pane.object = _analysis_figure(run)

    def _on_build_map(_event=None) -> None:
        try:
            scenario = _build_scenario_from_widgets()
            asset = terrain_service.fetch_map(
                scenario.map,
                fetch_remote=bool(fetch_remote_toggle.value),
                progress=lambda done, total, label: events.put(("status", f"{label}: {done}/{total}")),
            )
        except Exception as exc:
            console.write(f"Map/scenario error: {exc}", level="err")
            return
        state["scenario"] = scenario
        state["map_asset"] = asset
        map_pane.object = build_terrain_figure(asset, waypoints=scenario.waypoints.waypoints, markers=scenario.markers.markers)
        _update_replay(state.get("last_run"))
        console.write(f"Map preview ready for {scenario.name}.", level="ok")

    def _on_save_scenario(_event=None) -> None:
        try:
            scenario = _build_scenario_from_widgets()
            state["scenario"] = scenario
            path = scenario_manager.save(scenario)
        except Exception as exc:
            console.write(f"Could not save scenario: {exc}", level="err")
            return
        _refresh_scenario_options()
        console.write(f"Saved scenario to {path}.", level="ok")

    def _on_load_scenario(_event=None) -> None:
        options = state.get("scenario_options", {})
        if not isinstance(options, dict) or not scenario_select.value:
            return
        scenario_id = options.get(str(scenario_select.value))
        if not scenario_id:
            return
        try:
            scenario = scenario_manager.load(scenario_id)
            asset = terrain_service.fetch_map(scenario.map, fetch_remote=False)
        except Exception as exc:
            console.write(f"Could not load scenario: {exc}", level="err")
            return
        state["scenario"] = scenario
        state["map_asset"] = asset
        _set_widgets_from_scenario(scenario)
        map_pane.object = build_terrain_figure(asset, waypoints=scenario.waypoints.waypoints, markers=scenario.markers.markers)
        _update_replay(None)
        console.write(f"Loaded scenario {scenario.name}.", level="ok")

    def _on_run(_event=None) -> None:
        worker = state.get("worker")
        if isinstance(worker, threading.Thread) and worker.is_alive():
            console.write("A run is already in progress.", level="warn")
            return
        try:
            scenario = _build_scenario_from_widgets()
            scenario_path = scenario_manager.save(scenario)
            state["scenario"] = scenario
        except Exception as exc:
            console.write(f"Run config error: {exc}", level="err")
            return

        status_md.object = "**Running simulation...**"
        progress.value = 0
        run_btn.disabled = True
        console.write(f"Running {scenario.name} using {scenario.run_config.backend_id}.", level="ok")
        console.write(f"Scenario snapshot: {scenario_path}", level="ok")

        def _worker() -> None:
            try:
                run = sim_manager.run(scenario)
                events.put(("result", run))
            except BackendUnavailable as exc:
                events.put(("error", str(exc)))
            except BaseException as exc:
                events.put(("error", repr(exc)))

        thread = threading.Thread(target=_worker, daemon=True, name="DroneSimRun")
        state["worker"] = thread
        thread.start()

    def _drain_events() -> None:
        for _ in range(50):
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                status_md.object = f"**{payload}**"
            elif kind == "result":
                run = payload
                if not isinstance(run, RunResult):
                    continue
                path = run_store.save(run)
                state["last_run"] = run
                progress.value = 1
                run_btn.disabled = False
                status_md.object = "**Idle.**"
                _update_replay(run)
                _refresh_runs_table()
                console.write(f"Run complete: {run.status}. Saved to {path}.", level="ok")
            elif kind == "error":
                progress.value = 0
                run_btn.disabled = False
                status_md.object = "**Idle.**"
                console.write(f"Run failed: {payload}", level="err")

    def _on_session_load() -> None:
        try:
            pn.state.add_periodic_callback(_drain_events, period=200)
        except RuntimeError:
            pass

    build_map_btn.on_click(_on_build_map)
    save_scenario_btn.on_click(_on_save_scenario)
    run_btn.on_click(_on_run)
    refresh_scenarios_btn.on_click(lambda _event: _refresh_scenario_options())
    load_scenario_btn.on_click(_on_load_scenario)
    refresh_runs_btn.on_click(lambda _event: _refresh_runs_table())
    if pn.state.served:
        pn.state.onload(_on_session_load)

    _refresh_scenario_options()
    _refresh_runs_table()
    console.write("DroneSim app ready. Build a map preview, save a scenario, then run.", level="ok")

    scenario_controls = pn.Column(
        pn.pane.Markdown("### Scenario"),
        scenario_name,
        scenario_description,
        pn.Row(refresh_scenarios_btn, load_scenario_btn),
        scenario_select,
        pn.layout.Divider(),
        pn.pane.Markdown("### Map"),
        lat_input,
        lon_input,
        radius_input,
        resolution_input,
        fetch_remote_toggle,
        build_map_btn,
        pn.layout.Divider(),
        pn.pane.Markdown("### Vehicle and Run"),
        backend_select,
        altitude_input,
        dt_input,
        horizon_input,
        max_steps_input,
        lookahead_input,
        seed_input,
        pn.layout.Divider(),
        save_scenario_btn,
        run_btn,
        progress,
        sizing_mode="stretch_width",
    )

    create_tab = pn.Column(
        pn.Row(
            pn.Column(waypoint_text, marker_table, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        map_pane,
        sizing_mode="stretch_width",
    )
    replay_tab = pn.Column(replay_pane, sizing_mode="stretch_width")
    analysis_tab = pn.Column(summary_table, analysis_pane, sizing_mode="stretch_width")
    runs_tab = pn.Column(refresh_runs_btn, runs_table, sizing_mode="stretch_width")

    tabs = pn.Tabs(
        ("Create Map", create_tab),
        ("Replay 3D", replay_tab),
        ("Analysis", analysis_tab),
        ("Runs", runs_tab),
        ("Console", console.pane),
        sizing_mode="stretch_width",
    )

    template = pn.template.FastListTemplate(
        title="DroneSim",
        sidebar=[scenario_controls],
        main=[status_md, tabs],
        accent_base_color="#39ff14",
        header_background="#010409",
        theme="dark",
        sidebar_width=390,
    )
    return template


if __name__.startswith("bokeh"):
    build_app().servable()
