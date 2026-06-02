"""Serialize a :class:`RunResult` into chart-ready JSON for the frontend.

The frontend renders time series with uPlot, so we ship plain numeric arrays
(no figure objects). Single-run views mirror the old Plotly ``figures.py``;
Monte Carlo batch views add trial stats, miss histogram, and envelope series.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

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


def _trial_index(run: RunResult, order: int) -> int:
    cfg = run.metadata.get("cfg_summary") or {}
    mc = cfg.get("monte_carlo") if isinstance(cfg.get("monte_carlo"), dict) else {}
    if isinstance(mc, dict) and mc.get("trial_index") is not None:
        return int(mc["trial_index"])
    return order


def _trial_seed(run: RunResult) -> int | None:
    cfg = run.metadata.get("cfg_summary") or {}
    seed = cfg.get("seed")
    return int(seed) if seed is not None else None


def trial_rows(runs: list[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        s = run.summary
        rows.append(
            {
                "trial_index": _trial_index(run, i),
                "run_id": run.run_id,
                "seed": _trial_seed(run),
                "success": s.success,
                "miss_distance_m": s.miss_distance_m,
                "duration_s": s.duration_s,
                "settle_steps": s.settle_steps,
                "wallclock_s": s.wallclock_s,
                "path": run.metadata.get("path"),
            }
        )
    return rows


def _batch_summary(runs: list[RunResult]) -> list[dict[str, Any]]:
    if not runs:
        return []
    misses = [r.summary.miss_distance_m for r in runs if r.summary.miss_distance_m is not None]
    successes = sum(1 for r in runs if r.summary.success)
    wall = sum(r.summary.wallclock_s or 0.0 for r in runs)
    rows: list[dict[str, Any]] = [
        {"metric": "n_trials", "value": len(runs)},
        {"metric": "success_rate", "value": successes / len(runs) if runs else 0.0},
        {"metric": "total_wallclock_s", "value": wall},
    ]
    if misses:
        arr = np.array(misses, dtype=float)
        rows.extend(
            [
                {"metric": "miss_mean_m", "value": float(np.mean(arr))},
                {"metric": "miss_std_m", "value": float(np.std(arr))},
                {"metric": "miss_min_m", "value": float(np.min(arr))},
                {"metric": "miss_max_m", "value": float(np.max(arr))},
            ]
        )
    return rows


def _miss_histogram(runs: list[RunResult]) -> dict[str, list[float]]:
    misses = [r.summary.miss_distance_m for r in runs if r.summary.miss_distance_m is not None]
    if not misses:
        return {"bins": [], "counts": []}
    arr = np.array(misses, dtype=float)
    n_bins = max(5, min(40, int(math.sqrt(len(arr)) * 2)))
    counts, edges = np.histogram(arr, bins=n_bins)
    return {
        "bins": [float(edges[i]) for i in range(len(edges))],
        "counts": [int(c) for c in counts],
    }


def _resample_series(time_s: list[float], values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(t) == 0 or len(v) == 0:
        return t, v
    n = min(len(t), len(v))
    return t[:n], v[:n]


def _envelope_for_scalar(runs: list[RunResult], attr: str) -> dict[str, Any]:
    """Build mean ± std envelope for a scalar time series (e.g. tracking_error_m)."""
    series_list: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        vals = getattr(run, attr, None)
        if not vals:
            continue
        t, v = _resample_series(run.time_s, list(vals))
        if len(t):
            series_list.append((t, v))

    if not series_list:
        return {"time_s": [], "mean": [], "std": [], "trials": []}

    t_max = max(t[-1] for t, _ in series_list)
    t_min = min(t[0] for t, _ in series_list)
    longest = max(series_list, key=lambda tv: len(tv[0]))
    grid = longest[0]
    if grid[-1] < t_max:
        grid = np.linspace(t_min, t_max, max(len(longest[0]), 32))

    stacked: list[np.ndarray] = []
    trial_out: list[list[float | None]] = []
    for t, v in series_list:
        interp = np.interp(grid, t, v)
        stacked.append(interp)
        trial_out.append([float(x) for x in interp])

    mat = np.vstack(stacked)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return {
        "time_s": [float(x) for x in grid],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "trials": trial_out,
    }


def _envelope_for_vector(runs: list[RunResult], rows_attr: str, axis: int, label: str) -> dict[str, Any]:
    """Envelope for one component of a vector time series (velocity, attitude, etc.)."""
    pseudo_runs: list[tuple[list[float], list[float]]] = []
    for run in runs:
        rows = getattr(run, rows_attr, None) or []
        if not rows:
            continue
        vals = [float(r[axis]) if len(r) > axis else float("nan") for r in rows]
        pseudo_runs.append((run.time_s, vals))

    if not pseudo_runs:
        return {"time_s": [], "mean": [], "std": [], "trials": [], "label": label}

    series_list = [_resample_series(t, v) for t, v in pseudo_runs]
    longest = max(series_list, key=lambda tv: len(tv[0]))
    grid = longest[0]
    stacked: list[np.ndarray] = []
    trial_out: list[list[float]] = []
    for t, v in series_list:
        interp = np.interp(grid, t, v)
        stacked.append(interp)
        trial_out.append([float(x) for x in interp])

    mat = np.vstack(stacked)
    return {
        "time_s": [float(x) for x in grid],
        "mean": [float(x) for x in np.nanmean(mat, axis=0)],
        "std": [float(x) for x in np.nanstd(mat, axis=0)],
        "trials": trial_out,
        "label": label,
    }


def mc_analysis_block(runs: list[RunResult]) -> dict[str, Any]:
    """Aggregate Monte Carlo trial runs into chart-ready JSON."""
    trials = trial_rows(runs)
    for i, run in enumerate(runs):
        if trials[i].get("path") is None:
            trials[i]["path"] = run.metadata.get("saved_path")

    return {
        "mode": "monte_carlo",
        "summary": _batch_summary(runs),
        "trials": trials,
        "histogram": _miss_histogram(runs),
        "envelopes": {
            "tracking_error": _envelope_for_scalar(runs, "tracking_error_m"),
            "velocity": {
                "vx": _envelope_for_vector(runs, "velocity_mps", 0, "vx"),
                "vy": _envelope_for_vector(runs, "velocity_mps", 1, "vy"),
                "vz": _envelope_for_vector(runs, "velocity_mps", 2, "vz"),
            },
            "attitude": {
                "roll": _envelope_for_vector(runs, "attitude_rad", 0, "roll"),
                "pitch": _envelope_for_vector(runs, "attitude_rad", 1, "pitch"),
                "yaw": _envelope_for_vector(runs, "attitude_rad", 2, "yaw"),
            },
        },
    }


def mc_replay_block(
    runs: list[RunResult],
    *,
    center: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """Return minimal trajectory data for synchronized MC batch replay."""
    empty_center: dict[str, float | None] = {"lat": None, "lon": None}
    if not runs:
        return {
            "mode": "monte_carlo_replay",
            "center": center or empty_center,
            "reference_position_m": [],
            "trials": [],
        }

    ordered = sorted(
        enumerate(runs),
        key=lambda iv: (_trial_index(iv[1], iv[0]), iv[0]),
    )
    trials: list[dict[str, Any]] = []
    for i, run in ordered:
        trials.append(
            {
                "trial_index": _trial_index(run, i),
                "run_id": run.run_id,
                "time_s": list(run.time_s),
                "position_m": run.position_m or [],
                "success": run.summary.success,
            }
        )

    first = ordered[0][1]
    return {
        "mode": "monte_carlo_replay",
        "center": center or empty_center,
        "reference_position_m": first.reference_position_m or [],
        "trials": trials,
    }
