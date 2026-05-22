"""Waypoint loading, validation, and sample data."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

SAMPLE_WAYPOINTS = np.array(
    [[0.0, 0.0], [1.0, 2.0], [2.0, 4.5], [3.0, 3.0]],
    dtype=float,
)


class WaypointError(ValueError):
    """Raised when waypoint input is malformed."""


def validate(wpts: np.ndarray) -> np.ndarray:
    """Return a clean (N, 2) float array or raise ``WaypointError``."""
    arr = np.asarray(wpts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise WaypointError(
            f"Expected an (N, 2) or (N, 3) array of waypoints, got shape {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise WaypointError("Need at least 2 waypoints to build a trajectory")
    if not np.all(np.isfinite(arr)):
        raise WaypointError("Waypoints contain non-finite values")
    return arr[:, :2].copy()


def load_csv(source: Union[str, Path, bytes, io.IOBase]) -> np.ndarray:
    """Load waypoints from a CSV file path, file-like, or raw bytes payload.

    Accepted column conventions (case-insensitive):
      - ``x, y`` (2D)
      - ``x, y, z`` (3D — z is currently ignored, altitude lives in SimConfig)

    Lines starting with ``#`` are treated as comments. Falls back to header-less
    parsing when no recognised header row is present.
    """
    if isinstance(source, (bytes, bytearray)):
        buf: io.IOBase = io.BytesIO(bytes(source))
    elif isinstance(source, (str, Path)):
        buf = open(Path(source), "rb")
    else:
        buf = source

    try:
        try:
            df = pd.read_csv(buf, comment="#")
        except Exception as exc:
            raise WaypointError(f"Could not parse CSV: {exc}") from exc

        cols_lower = {c.lower().strip(): c for c in df.columns}
        if "x" in cols_lower and "y" in cols_lower:
            x = df[cols_lower["x"]].to_numpy(dtype=float)
            y = df[cols_lower["y"]].to_numpy(dtype=float)
            arr = np.column_stack([x, y])
        elif df.shape[1] >= 2:
            # Headerless or non-standard header: take first two numeric columns.
            arr = df.iloc[:, :2].to_numpy(dtype=float)
        else:
            raise WaypointError("CSV must contain at least 2 columns (x, y)")
    finally:
        if isinstance(source, (str, Path)):
            buf.close()

    return validate(arr)


def write_sample_csv(path: Union[str, Path]) -> Path:
    """Persist the legacy demo waypoints to disk for the GUI's sample mode."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SAMPLE_WAYPOINTS, columns=["x", "y"]).to_csv(p, index=False)
    return p
