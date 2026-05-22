"""Drone MC Panel app entry point.

Run with::

    panel serve drone_mc/app/main.py --show

The app is a console-styled Panel ``FastListTemplate`` whose left sidebar
collects waypoints + sim/MC parameters and whose main area streams live
trajectories, state plots, control plots, and statistics as ProcessPool
workers complete.
"""
from __future__ import annotations

import datetime as _dt
import queue
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Optional

# `panel serve` executes this script directly rather than importing it as a
# package member, so relative imports (`from .. import visuals`) raise
# ImportError. Adding the package's grandparent to ``sys.path`` lets us use
# absolute imports both under `panel serve` and when the module is imported
# normally.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import panel as pn

from drone_mc import visuals
from drone_mc.app.sidebar import build_sidebar
from drone_mc.config import MCConfig, SimConfig
from drone_mc.monte_carlo import MonteCarloRunner
from drone_mc.simulator import SimResult, run_simulation

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

THEME_CSS_PATH = Path(__file__).with_name("theme.css")


# ---------------------------------------------------------------------------
# Console log pane
# ---------------------------------------------------------------------------


class ConsoleLog:
    """Append-only HTML console with timestamped, color-coded entries."""

    def __init__(self, max_lines: int = 400) -> None:
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.pane = pn.pane.HTML(
            "<div class='console-log'></div>",
            sizing_mode="stretch_width",
        )

    def write(self, msg: str, level: str = "ok") -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        cls = {"ok": "ok", "warn": "warn", "err": "err"}.get(level, "ok")
        self.lines.append(
            f"<span class='ts'>[{ts}]</span> <span class='{cls}'>{msg}</span>"
        )
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]
        body = "<br/>".join(self.lines)
        self.pane.object = f"<div class='console-log'>{body}</div>"

    def clear(self) -> None:
        self.lines.clear()
        self.pane.object = "<div class='console-log'></div>"


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------


def build_app() -> pn.template.FastListTemplate:
    sidebar = build_sidebar()
    console = ConsoleLog()

    # Live state shared between callbacks (main thread mutates only).
    results: list[SimResult] = []
    runner_state: dict[str, Optional[MonteCarloRunner]] = {"runner": None}
    event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

    # --- Main panes ----------------------------------------------------
    progress_bar = pn.indicators.Progress(
        name="Monte Carlo progress",
        value=0, max=100, sizing_mode="stretch_width",
        bar_color="success",
    )
    status_md = pn.pane.Markdown("**Idle.**", styles={"color": "#39ff14"})
    success_indicator = pn.indicators.Number(
        name="Success rate",
        value=0,
        format="{value:.0f} %",
        colors=[(33, "danger"), (66, "warning"), (100, "success")],
    )
    completed_indicator = pn.indicators.Number(
        name="Completed",
        value=0,
        format="{value:.0f}",
    )

    plot_3d_pane = pn.pane.Plotly(
        visuals.trajectory_3d([]), sizing_mode="stretch_width", height=550
    )
    plot_2d_pane = pn.pane.Bokeh(visuals.trajectory_2d([]), sizing_mode="stretch_width")
    linear_pane = pn.pane.Bokeh(visuals.state_grid([], "linear"), sizing_mode="stretch_width")
    angular_pane = pn.pane.Bokeh(visuals.state_grid([], "angular"), sizing_mode="stretch_width")
    control_pane = pn.pane.Bokeh(visuals.control_grid([]), sizing_mode="stretch_width")
    stats_table = pn.widgets.Tabulator(
        value=visuals.stats_dataframe([]),
        sizing_mode="stretch_width", height=380, show_index=False, theme="midnight",
    )
    hist_pane = pn.pane.Bokeh(visuals.miss_histogram([]), sizing_mode="stretch_width")

    overview_header = pn.Row(
        progress_bar,
        success_indicator,
        completed_indicator,
        sizing_mode="stretch_width",
    )

    tabs = pn.Tabs(
        ("Trajectory", pn.Column(plot_3d_pane, plot_2d_pane, sizing_mode="stretch_width")),
        ("Linear states", linear_pane),
        ("Angular states", angular_pane),
        ("Control inputs", control_pane),
        ("Statistics", pn.Column(stats_table, hist_pane, sizing_mode="stretch_width")),
        ("Console", console.pane),
        sizing_mode="stretch_width",
    )

    main_area = pn.Column(
        status_md,
        overview_header,
        tabs,
        sizing_mode="stretch_width",
    )

    # --- Config helpers ------------------------------------------------
    def _build_sim_cfg() -> SimConfig:
        wpts = sidebar.waypoint_state.waypoints
        return SimConfig(
            waypoints=np.asarray(wpts, dtype=float),
            altitude=float(sidebar.sim_params["altitude"].value),
            dt=float(sidebar.sim_params["dt"].value),
            horizon=int(sidebar.sim_params["horizon"].value),
            max_steps=int(sidebar.sim_params["max_steps"].value),
            lookahead=int(sidebar.sim_params["lookahead"].value),
            init_pos_std=float(sidebar.mc_params["init_pos_std"].value),
            init_vel_std=float(sidebar.mc_params["init_vel_std"].value),
            init_att_std=float(sidebar.mc_params["init_att_std"].value),
            force_noise_std=float(sidebar.mc_params["force_noise_std"].value),
            mass_jitter_pct=float(sidebar.mc_params["mass_jitter_pct"].value),
            inertia_jitter_pct=float(sidebar.mc_params["inertia_jitter_pct"].value),
            init_state={  # default init state with z = altitude
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "roll_dot": 0.0, "pitch_dot": 0.0, "yaw_dot": 0.0,
                "x_dot": 0.0, "y_dot": 0.0, "z_dot": 0.0,
                "x": float(wpts[0][0]), "y": float(wpts[0][1]),
                "z": float(sidebar.sim_params["altitude"].value),
            },
        )

    def _build_mc_cfg() -> MCConfig:
        base = _build_sim_cfg()
        return MCConfig(
            base=base,
            n_trials=int(sidebar.mc_params["n_trials"].value),
            workers=int(sidebar.mc_params["workers"].value),
            base_seed=int(sidebar.mc_params["base_seed"].value),
        )

    # --- Visualization refresh -----------------------------------------
    def _refresh_views(rs: list[SimResult]) -> None:
        if not rs:
            return
        last = rs[-1]
        plot_3d_pane.object = visuals.trajectory_3d(
            rs, spline=last.spline, waypoints=last.waypoints,
            altitude=float(sidebar.sim_params["altitude"].value),
        )
        plot_2d_pane.object = visuals.trajectory_2d(
            rs, spline=last.spline, waypoints=last.waypoints,
        )
        linear_pane.object = visuals.state_grid(rs, "linear")
        angular_pane.object = visuals.state_grid(rs, "angular")
        control_pane.object = visuals.control_grid(rs)
        stats_table.value = visuals.stats_dataframe(rs)
        hist_pane.object = visuals.miss_histogram(rs)
        succ = sum(1 for r in rs if r.success)
        success_indicator.value = round(100.0 * succ / max(1, len(rs)))
        completed_indicator.value = len(rs)

    # --- Periodic queue drain ------------------------------------------
    def _drain_queue() -> None:
        # Pull at most a small batch per tick so the UI stays responsive
        # even when many trials complete in quick succession.
        new_results = False
        progress_update: Optional[tuple[int, int]] = None
        for _ in range(50):
            try:
                kind, payload = event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "result":
                idx, res = payload  # type: ignore[misc]
                results.append(res)
                outcome = "OK" if res.success else "MISS"
                console.write(
                    f"Trial {idx:>3} {outcome} | miss={res.miss_distance:.3f} m "
                    f"| settle={res.settle_steps} steps | {res.wallclock_s:.2f} s",
                    level="ok" if res.success else "warn",
                )
                new_results = True
            elif kind == "progress":
                progress_update = payload  # type: ignore[assignment]
            elif kind == "error":
                idx, err = payload  # type: ignore[misc]
                console.write(f"Trial {idx} FAILED: {err}", level="err")
            elif kind == "done":
                console.write(payload or "Run complete.", level="ok")
                sidebar.run_single_btn.disabled = False
                sidebar.run_mc_btn.disabled = False
                sidebar.cancel_btn.disabled = True
                status_md.object = "**Idle.**"
                runner_state["runner"] = None
            elif kind == "status":
                status_md.object = f"**{payload}**"

        if progress_update is not None:
            done, total = progress_update
            progress_bar.max = total
            progress_bar.value = done

        if new_results:
            _refresh_views(results)

    # --- Run handlers --------------------------------------------------
    def _on_progress(done: int, total: int) -> None:
        event_queue.put(("progress", (done, total)))

    def _on_result(idx: int, res: SimResult) -> None:
        event_queue.put(("result", (idx, res)))

    def _on_error(idx: int, exc: BaseException) -> None:
        event_queue.put(("error", (idx, repr(exc))))

    def _on_done() -> None:
        event_queue.put(("done", "Monte Carlo run complete."))

    def _start_mc(_event=None) -> None:
        if runner_state["runner"] is not None:
            console.write("A run is already in progress.", level="warn")
            return
        try:
            mc_cfg = _build_mc_cfg()
        except Exception as exc:  # noqa: BLE001
            console.write(f"Config error: {exc}", level="err")
            return

        results.clear()
        _refresh_views([])
        progress_bar.value = 0
        progress_bar.max = mc_cfg.n_trials
        success_indicator.value = 0
        completed_indicator.value = 0
        status_md.object = (
            f"**Running {mc_cfg.n_trials} trials on {mc_cfg.workers} workers...**"
        )
        console.write(
            f"Launching MC: {mc_cfg.n_trials} trials, {mc_cfg.workers} workers, "
            f"seed={mc_cfg.base_seed}",
            level="ok",
        )

        runner = MonteCarloRunner(
            mc_cfg,
            on_progress=_on_progress,
            on_result=_on_result,
            on_error=_on_error,
            on_done=_on_done,
        )
        runner_state["runner"] = runner
        sidebar.run_single_btn.disabled = True
        sidebar.run_mc_btn.disabled = True
        sidebar.cancel_btn.disabled = False
        runner.start()

    def _start_single(_event=None) -> None:
        if runner_state["runner"] is not None:
            console.write("A run is already in progress.", level="warn")
            return
        try:
            cfg = _build_sim_cfg()
        except Exception as exc:  # noqa: BLE001
            console.write(f"Config error: {exc}", level="err")
            return
        results.clear()
        _refresh_views([])
        progress_bar.max = 1
        progress_bar.value = 0
        success_indicator.value = 0
        completed_indicator.value = 0
        status_md.object = "**Running single deterministic simulation...**"
        console.write("Launching single simulation (no perturbation).", level="ok")
        sidebar.run_single_btn.disabled = True
        sidebar.run_mc_btn.disabled = True

        def _run_single_thread() -> None:
            try:
                # Suppress randomness for the deterministic single-run case.
                deterministic = replace(
                    cfg,
                    init_pos_std=0.0, init_vel_std=0.0, init_att_std=0.0,
                    force_noise_std=0.0, mass_jitter_pct=0.0, inertia_jitter_pct=0.0,
                )
                res = run_simulation(deterministic)
                event_queue.put(("result", (0, res)))
                event_queue.put(("progress", (1, 1)))
                event_queue.put(("done", "Single simulation complete."))
            except BaseException as exc:  # noqa: BLE001
                event_queue.put(("error", (0, repr(exc))))
                event_queue.put(("done", f"Single simulation failed: {exc}"))

        threading.Thread(target=_run_single_thread, daemon=True, name="SingleSim").start()

    def _cancel(_event=None) -> None:
        runner = runner_state.get("runner")
        if runner is None:
            return
        console.write("Cancel requested...", level="warn")
        runner.cancel()

    sidebar.run_single_btn.on_click(_start_single)
    sidebar.run_mc_btn.on_click(_start_mc)
    sidebar.cancel_btn.on_click(_cancel)

    # Register the queue-drain callback only once a Panel session is live.
    # Calling ``add_periodic_callback`` at module load time fails outside a
    # running event loop (e.g. when the file is imported for testing); using
    # ``pn.state.onload`` defers it to ``panel serve`` session start.
    def _on_session_load() -> None:
        try:
            pn.state.add_periodic_callback(_drain_queue, period=200)
        except RuntimeError:
            pass

    pn.state.onload(_on_session_load)

    console.write("Drone MC console ready. Choose waypoints, then Run.", level="ok")

    # --- Template ------------------------------------------------------
    raw_css = ""
    if THEME_CSS_PATH.exists():
        raw_css = THEME_CSS_PATH.read_text(encoding="utf-8")

    template = pn.template.FastListTemplate(
        title="DRONE MC // QUADCOPTER MPC",
        sidebar=[sidebar.layout],
        main=[main_area],
        accent_base_color="#39ff14",
        header_background="#010409",
        theme="dark",
        sidebar_width=380,
        raw_css=[raw_css] if raw_css else [],
    )
    return template


def _entry() -> None:
    build_app().servable()


_entry()
