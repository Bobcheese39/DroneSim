"""Figure builders for the Panel app.

Each function takes a ``list[SimResult]`` (plus extras) and returns a fresh
figure object. The Panel layer is responsible for replacing panes — keeping
the visuals layer pure makes the app code thin and the figures unit-testable.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure as bk_figure

from .simulator import SimResult


# ---------------------------------------------------------------------------
# Trajectory views
# ---------------------------------------------------------------------------


def trajectory_3d(
    results: Sequence[SimResult],
    spline: np.ndarray | None = None,
    waypoints: np.ndarray | None = None,
    altitude: float = 5.0,
) -> go.Figure:
    """Plotly 3D scatter: every completed trial trajectory + reference + waypoints."""
    fig = go.Figure()

    if spline is not None and len(spline):
        fig.add_trace(
            go.Scatter3d(
                x=spline[:, 0],
                y=spline[:, 1],
                z=np.full(len(spline), altitude),
                mode="lines",
                line=dict(color="#ff7b00", width=4),
                name="Reference spline",
            )
        )

    if waypoints is not None and len(waypoints):
        fig.add_trace(
            go.Scatter3d(
                x=waypoints[:, 0],
                y=waypoints[:, 1],
                z=np.full(len(waypoints), altitude),
                mode="markers+text",
                marker=dict(size=6, color="#39ff14", symbol="diamond"),
                text=[f"WP{i}" for i in range(len(waypoints))],
                textposition="top center",
                name="Waypoints",
            )
        )

    n = len(results)
    for i, r in enumerate(results):
        # Alpha decreases as more trials accumulate so density is readable.
        alpha = max(0.15, 1.0 / max(1, n) ** 0.5)
        success_color = "#39ff14" if r.success else "#ff3860"
        fig.add_trace(
            go.Scatter3d(
                x=r.pos[:, 0],
                y=r.pos[:, 1],
                z=r.pos[:, 2],
                mode="lines",
                line=dict(color=success_color, width=2),
                opacity=alpha,
                name=f"Trial {r.trial_index}",
                showlegend=(i < 5),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D Trajectories",
        height=550,
    )
    return fig


def trajectory_2d(
    results: Sequence[SimResult],
    spline: np.ndarray | None = None,
    waypoints: np.ndarray | None = None,
):
    """Bokeh top-down x/y view with all trial trajectories overlaid."""
    p = bk_figure(
        title="Top-down (X / Y)",
        x_axis_label="X (m)",
        y_axis_label="Y (m)",
        sizing_mode="stretch_width",
        height=420,
        match_aspect=True,
        background_fill_color="#0d1117",
        border_fill_color="#0d1117",
    )
    p.grid.grid_line_color = "#30363d"
    p.xaxis.axis_label_text_color = "#c9d1d9"
    p.yaxis.axis_label_text_color = "#c9d1d9"
    p.title.text_color = "#39ff14"

    if spline is not None and len(spline):
        p.line(spline[:, 0], spline[:, 1], color="#ff7b00", line_width=2, legend_label="Reference")

    n = max(1, len(results))
    for r in results:
        color = "#39ff14" if r.success else "#ff3860"
        p.line(
            r.pos[:, 0],
            r.pos[:, 1],
            color=color,
            line_alpha=max(0.1, 1.0 / n ** 0.5),
            line_width=1.5,
        )

    if waypoints is not None and len(waypoints):
        p.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            size=10,
            color="#58a6ff",
            marker="diamond",
            legend_label="Waypoints",
        )

    if p.legend:
        p.legend.location = "top_left"
        p.legend.background_fill_alpha = 0.4
        p.legend.label_text_color = "#c9d1d9"
    return p


# ---------------------------------------------------------------------------
# State / control time series with mean ± 1σ ribbons
# ---------------------------------------------------------------------------


_LINEAR_FIELDS = [
    ("X position (m)", "pos", 0),
    ("Y position (m)", "pos", 1),
    ("Z position (m)", "pos", 2),
    ("X velocity (m/s)", "vel", 0),
    ("Y velocity (m/s)", "vel", 1),
    ("Z velocity (m/s)", "vel", 2),
]
_ANGULAR_FIELDS = [
    ("Roll (rad)", "att", 0),
    ("Pitch (rad)", "att", 1),
    ("Yaw (rad)", "att", 2),
    ("Roll rate (rad/s)", "att_rate", 0),
    ("Pitch rate (rad/s)", "att_rate", 1),
    ("Yaw rate (rad/s)", "att_rate", 2),
]
_CONTROL_FIELDS = [
    ("Thrust (N)", 0),
    ("Tau X (Nm)", 1),
    ("Tau Y (Nm)", 2),
    ("Tau Z (Nm)", 3),
]


def _stack_padded(arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad ragged 1D arrays with NaN so we can take nan-aware stats."""
    if not arrs:
        return np.zeros((0, 0)), np.zeros(0)
    T = max(len(a) for a in arrs)
    out = np.full((len(arrs), T), np.nan)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    t = np.arange(T)
    return out, t


def _ribbon_subplot(
    title: str,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    color: str = "#39ff14",
):
    p = bk_figure(
        title=title,
        sizing_mode="stretch_width",
        height=220,
        background_fill_color="#0d1117",
        border_fill_color="#0d1117",
    )
    p.grid.grid_line_color = "#30363d"
    p.title.text_color = color
    p.xaxis.axis_label = "Time (s)"
    p.xaxis.axis_label_text_color = "#c9d1d9"
    p.yaxis.axis_label_text_color = "#c9d1d9"

    if not ys:
        return p

    stacked, _ = _stack_padded(ys)
    # Use the longest time vector as the x grid.
    longest_x = max(xs, key=len) if xs else np.array([])

    n = len(ys)
    alpha = max(0.08, 1.0 / n ** 0.5)
    for x, y in zip(xs, ys):
        p.line(x, y, color=color, line_alpha=alpha, line_width=1)

    if n > 1:
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        if len(longest_x) == len(mean):
            t = longest_x
            band_x = np.concatenate([t, t[::-1]])
            band_y = np.concatenate([mean + std, (mean - std)[::-1]])
            p.patch(band_x, band_y, color=color, fill_alpha=0.12, line_alpha=0)
            p.line(t, mean, color="#f0c674", line_width=2, legend_label="mean")
            if p.legend:
                p.legend.location = "top_right"
                p.legend.background_fill_alpha = 0.3
                p.legend.label_text_color = "#c9d1d9"
    return p


def state_grid(results: Sequence[SimResult], group: str = "linear"):
    """Bokeh grid of per-axis time series with optional mean ± 1σ ribbon."""
    fields = _LINEAR_FIELDS if group == "linear" else _ANGULAR_FIELDS
    color = "#39ff14" if group == "linear" else "#58a6ff"
    plots = []
    for title, attr, axis in fields:
        xs = [r.time for r in results]
        ys = [getattr(r, attr)[:, axis] for r in results]
        plots.append(_ribbon_subplot(title, xs, ys, color=color))
    return gridplot(plots, ncols=3, sizing_mode="stretch_width", merge_tools=True)


def control_grid(results: Sequence[SimResult]):
    plots = []
    for title, axis in _CONTROL_FIELDS:
        xs = [r.time for r in results]
        ys = [r.u[:, axis] for r in results]
        plots.append(_ribbon_subplot(title, xs, ys, color="#ffb454"))
    return gridplot(plots, ncols=2, sizing_mode="stretch_width", merge_tools=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def stats_dataframe(results: Iterable[SimResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "trial": r.trial_index,
            "seed": r.seed,
            "success": r.success,
            "miss_distance_m": round(r.miss_distance, 4),
            "settle_steps": r.settle_steps,
            "duration_s": round(float(r.time[-1]) if len(r.time) else 0.0, 3),
            "max_tilt_rad": round(float(np.nanmax(np.linalg.norm(r.att[:, :2], axis=1))) if len(r.att) else 0.0, 4),
            "ctrl_effort": round(float(np.nansum(np.linalg.norm(r.u, axis=1))), 2),
            "wallclock_s": round(r.wallclock_s, 3),
        })
    return pd.DataFrame(rows)


def miss_histogram(results: Sequence[SimResult]):
    p = bk_figure(
        title="Final miss distance distribution",
        x_axis_label="Miss distance (m)",
        y_axis_label="Trials",
        sizing_mode="stretch_width",
        height=300,
        background_fill_color="#0d1117",
        border_fill_color="#0d1117",
    )
    p.grid.grid_line_color = "#30363d"
    p.title.text_color = "#39ff14"
    p.xaxis.axis_label_text_color = "#c9d1d9"
    p.yaxis.axis_label_text_color = "#c9d1d9"
    if not results:
        return p
    misses = np.array([r.miss_distance for r in results])
    bins = max(5, min(40, int(np.sqrt(len(misses)) * 2)))
    hist, edges = np.histogram(misses, bins=bins)
    src = ColumnDataSource(
        data={"top": hist, "left": edges[:-1], "right": edges[1:]}
    )
    p.quad(top="top", bottom=0, left="left", right="right", source=src,
           fill_color="#39ff14", line_color="#0d1117", fill_alpha=0.7)
    return p
