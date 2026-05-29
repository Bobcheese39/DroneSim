"""Serialize a :class:`RunResult` into chart-ready JSON for the frontend.

The frontend renders time series with uPlot, so we ship plain numeric arrays
(no figure objects). This mirrors the metrics the old Plotly ``figures.py``
produced, minus the Monte-Carlo views (deferred per the core-first scope).
"""
from __future__ import annotations

from typing import Any

from dronesim.models import RunResult


def _col(rows: list[list[float]], idx: int) -> list[float | None]:
    out: list[float | None] = []
    for row in rows:
        out.append(float(row[idx]) if row is not None and len(row) > idx else None)
    return out


def _error_decomposition(run: RunResult) -> dict[str, list[float | None]]:
    pos = run.position_m or []
    ref = run.reference_position_m or []
    n = min(len(pos), len(ref))
    ex: list[float | None] = []
    ey: list[float | None] = []
    ez: list[float | None] = []
    for i in range(n):
        p, r = pos[i], ref[i]
        ex.append(float(p[0] - r[0]) if len(p) > 0 and len(r) > 0 else None)
        ey.append(float(p[1] - r[1]) if len(p) > 1 and len(r) > 1 else None)
        ez.append(float(p[2] - r[2]) if len(p) > 2 and len(r) > 2 else None)
    return {"ex": ex, "ey": ey, "ez": ez}


def summary_rows(run: RunResult) -> list[dict[str, Any]]:
    s = run.summary
    return [
        {"metric": "status", "value": run.status},
        {"metric": "success", "value": s.success},
        {"metric": "miss_distance_m", "value": s.miss_distance_m},
        {"metric": "duration_s", "value": s.duration_s},
        {"metric": "settle_steps", "value": s.settle_steps},
        {"metric": "max_tracking_error_m", "value": s.max_tracking_error_m},
        {"metric": "mean_tracking_error_m", "value": s.mean_tracking_error_m},
        {"metric": "max_altitude_m", "value": s.max_altitude_m},
        {"metric": "min_altitude_m", "value": s.min_altitude_m},
        {"metric": "wallclock_s", "value": s.wallclock_s},
    ]


def parameter_rows(run: RunResult) -> list[dict[str, Any]]:
    cfg = run.metadata.get("cfg_summary") or {}
    if not cfg:
        return []
    return [{"parameter": str(k), "value": v} for k, v in sorted(cfg.items())]


def analysis_block(run: RunResult, clearance_m: list[float] | None = None) -> dict[str, Any]:
    """Return all series + tables needed by the Analysis and Replay views."""
    return {
        "time_s": list(run.time_s),
        "tracking_error_m": list(run.tracking_error_m),
        "error_decomposition": _error_decomposition(run),
        "velocity": {
            "vx": _col(run.velocity_mps, 0),
            "vy": _col(run.velocity_mps, 1),
            "vz": _col(run.velocity_mps, 2),
        },
        "acceleration": {
            "ax": _col(run.acceleration_mps2, 0),
            "ay": _col(run.acceleration_mps2, 1),
            "az": _col(run.acceleration_mps2, 2),
        },
        "attitude": {
            "roll": _col(run.attitude_rad, 0),
            "pitch": _col(run.attitude_rad, 1),
            "yaw": _col(run.attitude_rad, 2),
        },
        "angular_rate": {
            "p": _col(run.angular_rate_rad_s, 0),
            "q": _col(run.angular_rate_rad_s, 1),
            "r": _col(run.angular_rate_rad_s, 2),
        },
        "controls": {
            "ft": _col(run.controls, 0),
            "tx": _col(run.controls, 1),
            "ty": _col(run.controls, 2),
            "tz": _col(run.controls, 3),
        },
        "clearance_m": list(clearance_m) if clearance_m is not None else None,
        "summary": summary_rows(run),
        "parameters": parameter_rows(run),
    }
