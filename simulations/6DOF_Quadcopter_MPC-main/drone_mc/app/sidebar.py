"""Sidebar widgets: waypoint ingestion + simulation / Monte Carlo parameters."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, PointDrawTool
from bokeh.plotting import figure as bk_figure

from ..waypoints import SAMPLE_WAYPOINTS, WaypointError, load_csv, validate

WAYPOINT_MODES = ["Sample", "CSV file", "Click on grid"]


@dataclass
class WidgetBundle:
    """All sidebar widgets the main app needs to read."""

    layout: pn.Column
    waypoint_state: "WaypointState"
    sim_params: dict
    mc_params: dict
    run_single_btn: pn.widgets.Button
    run_mc_btn: pn.widgets.Button
    cancel_btn: pn.widgets.Button


class WaypointState(param.Parameterized):
    """Single source of truth for the currently-selected waypoints.

    The mode toggle decides which input mechanism is visible; whichever one is
    active writes its result into ``waypoints``.
    """

    waypoints = param.Array(default=SAMPLE_WAYPOINTS.copy())
    mode = param.Selector(objects=WAYPOINT_MODES, default="Sample")
    error = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self._csv_input = pn.widgets.FileInput(accept=".csv", name="Waypoint CSV")
        self._csv_input.param.watch(self._on_csv, "value")

        self._table = pn.widgets.Tabulator(
            value=pd.DataFrame(self.waypoints, columns=["x", "y"]),
            show_index=False,
            height=240,
            sizing_mode="stretch_width",
            theme="midnight",
        )
        self._table.param.watch(self._on_table_edit, "value")

        self._grid_source = ColumnDataSource(
            data={"x": list(self.waypoints[:, 0]), "y": list(self.waypoints[:, 1])}
        )
        self._grid_source.on_change("data", self._on_grid_change)
        self._grid_fig = self._build_grid_fig(self._grid_source)
        self._grid_pane = pn.pane.Bokeh(self._grid_fig, sizing_mode="stretch_width")

        self._mode_toggle = pn.widgets.RadioButtonGroup(
            name="Waypoint source", options=WAYPOINT_MODES, value=self.mode,
            button_type="primary", sizing_mode="stretch_width",
        )
        self._mode_toggle.param.watch(self._on_mode, "value")

        self._clear_btn = pn.widgets.Button(name="Clear waypoints", button_type="warning")
        self._clear_btn.on_click(self._on_clear)

        self._error_pane = pn.pane.Markdown("", styles={"color": "#ff3860"})

        self._csv_box = pn.Column(self._csv_input, visible=False)
        self._grid_box = pn.Column(self._grid_pane, self._clear_btn, visible=False)
        self._sample_box = pn.Column(
            pn.pane.Markdown("Using legacy demo waypoints `[[0,0],[1,2],[2,4.5],[3,3]]`."),
            visible=True,
        )

        self._table_box = pn.Column(
            pn.pane.Markdown("**Active waypoints**"),
            self._table,
        )

        self.panel = pn.Column(
            self._mode_toggle,
            self._sample_box,
            self._csv_box,
            self._grid_box,
            self._error_pane,
            self._table_box,
        )

    @staticmethod
    def _build_grid_fig(source: ColumnDataSource):
        fig = bk_figure(
            title="Click empty space to add a waypoint, drag to move",
            x_range=(-2, 8), y_range=(-2, 8),
            sizing_mode="stretch_width",
            height=320,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#0d1117",
            border_fill_color="#0d1117",
        )
        fig.grid.grid_line_color = "#30363d"
        fig.title.text_color = "#39ff14"
        fig.xaxis.axis_label = "X (m)"
        fig.yaxis.axis_label = "Y (m)"
        fig.xaxis.axis_label_text_color = "#c9d1d9"
        fig.yaxis.axis_label_text_color = "#c9d1d9"
        scatter = fig.scatter("x", "y", source=source, size=14, color="#39ff14",
                              line_color="#0d1117")
        line = fig.line("x", "y", source=source, color="#ff7b00", line_dash="dashed")
        draw_tool = PointDrawTool(renderers=[scatter, line], add=True, drag=True)
        fig.add_tools(draw_tool)
        fig.toolbar.active_tap = draw_tool
        return fig

    def _set_waypoints(self, arr: np.ndarray, source: str) -> None:
        try:
            cleaned = validate(arr)
        except WaypointError as exc:
            self.error = f"{exc}"
            self._error_pane.object = f"**{exc}**"
            return
        self.error = ""
        self._error_pane.object = ""

        # Avoid feedback loops: only update widgets that did NOT originate the change.
        self.waypoints = cleaned
        if source != "table":
            self._table.value = pd.DataFrame(cleaned, columns=["x", "y"])
        if source != "grid":
            self._grid_source.data = {
                "x": list(cleaned[:, 0]),
                "y": list(cleaned[:, 1]),
            }

    def _on_mode(self, event) -> None:
        self.mode = event.new
        self._sample_box.visible = (event.new == "Sample")
        self._csv_box.visible = (event.new == "CSV file")
        self._grid_box.visible = (event.new == "Click on grid")
        if event.new == "Sample":
            self._set_waypoints(SAMPLE_WAYPOINTS.copy(), "mode")

    def _on_csv(self, event) -> None:
        if event.new is None:
            return
        try:
            arr = load_csv(event.new)
        except WaypointError as exc:
            self.error = f"{exc}"
            self._error_pane.object = f"**{exc}**"
            return
        self._set_waypoints(arr, "csv")

    def _on_table_edit(self, event) -> None:
        df: pd.DataFrame = event.new
        if df is None or df.empty:
            return
        try:
            arr = df[["x", "y"]].to_numpy(dtype=float)
        except Exception as exc:
            self._error_pane.object = f"**Bad table value: {exc}**"
            return
        self._set_waypoints(arr, "table")

    def _on_grid_change(self, attr, old, new) -> None:
        del attr, old
        xs = new.get("x", [])
        ys = new.get("y", [])
        if not xs or len(xs) < 2:
            return
        try:
            arr = np.column_stack([xs, ys]).astype(float)
        except Exception as exc:
            self._error_pane.object = f"**Bad grid value: {exc}**"
            return
        self._set_waypoints(arr, "grid")

    def _on_clear(self, _event) -> None:
        empty = SAMPLE_WAYPOINTS[:2].copy()
        self._set_waypoints(empty, "clear")


def _slider(name: str, *, start, end, value, step=None, fmt=None) -> pn.widgets.Widget:
    if isinstance(value, int) and isinstance(start, int) and isinstance(end, int):
        return pn.widgets.IntSlider(name=name, start=start, end=end, value=value, step=step or 1)
    kwargs = {"name": name, "start": start, "end": end, "value": value}
    if step is not None:
        kwargs["step"] = step
    if fmt is not None:
        kwargs["format"] = fmt
    return pn.widgets.FloatSlider(**kwargs)


def build_sidebar() -> WidgetBundle:
    """Compose the full sidebar layout and return its widget bundle."""
    waypoint_state = WaypointState()

    # Sim params -----------------------------------------------------------
    dt = _slider("dt (s)", start=0.01, end=0.2, value=0.1, step=0.01, fmt="0.00")
    horizon = _slider("MPC horizon", start=5, end=40, value=20)
    max_steps = _slider("max sim steps", start=50, end=1000, value=250, step=10)
    altitude = _slider("Target altitude (m)", start=1.0, end=20.0, value=5.0, step=0.5, fmt="0.0")
    lookahead = _slider("Spline lookahead (pts)", start=10, end=200, value=60, step=5)

    sim_params = {
        "dt": dt, "horizon": horizon, "max_steps": max_steps,
        "altitude": altitude, "lookahead": lookahead,
    }

    # MC params ------------------------------------------------------------
    n_trials = _slider("# trials", start=1, end=500, value=25)
    workers = _slider("workers", start=1, end=32, value=4)
    base_seed = pn.widgets.IntInput(name="base seed", value=0, start=0)
    init_pos_std = _slider("init pos σ (m)", start=0.0, end=1.0, value=0.05, step=0.01, fmt="0.00")
    init_vel_std = _slider("init vel σ (m/s)", start=0.0, end=1.0, value=0.05, step=0.01, fmt="0.00")
    init_att_std = _slider("init att σ (rad)", start=0.0, end=0.3, value=0.02, step=0.005, fmt="0.000")
    force_noise_std = _slider("control noise σ", start=0.0, end=2.0, value=0.1, step=0.05, fmt="0.00")
    mass_jitter = _slider("mass jitter %", start=0.0, end=0.3, value=0.05, step=0.01, fmt="0.00")
    inertia_jitter = _slider("inertia jitter %", start=0.0, end=0.3, value=0.05, step=0.01, fmt="0.00")

    mc_params = {
        "n_trials": n_trials, "workers": workers, "base_seed": base_seed,
        "init_pos_std": init_pos_std, "init_vel_std": init_vel_std,
        "init_att_std": init_att_std, "force_noise_std": force_noise_std,
        "mass_jitter_pct": mass_jitter, "inertia_jitter_pct": inertia_jitter,
    }

    run_single_btn = pn.widgets.Button(name="Run single", button_type="success", sizing_mode="stretch_width")
    run_mc_btn = pn.widgets.Button(name="Run Monte Carlo", button_type="primary", sizing_mode="stretch_width")
    cancel_btn = pn.widgets.Button(name="Cancel", button_type="danger", sizing_mode="stretch_width", disabled=True)

    layout = pn.Column(
        pn.pane.Markdown("### Waypoints", styles={"color": "#39ff14"}),
        waypoint_state.panel,
        pn.layout.Divider(),
        pn.pane.Markdown("### Simulation", styles={"color": "#39ff14"}),
        *sim_params.values(),
        pn.layout.Divider(),
        pn.pane.Markdown("### Monte Carlo", styles={"color": "#39ff14"}),
        *mc_params.values(),
        pn.layout.Divider(),
        run_single_btn,
        run_mc_btn,
        cancel_btn,
        sizing_mode="stretch_width",
    )

    return WidgetBundle(
        layout=layout,
        waypoint_state=waypoint_state,
        sim_params=sim_params,
        mc_params=mc_params,
        run_single_btn=run_single_btn,
        run_mc_btn=run_mc_btn,
        cancel_btn=cancel_btn,
    )
