"""Cubic Catmull-Rom-style spline generator for waypoint trajectories."""
from __future__ import annotations

from math import atan2

import numpy as np


class SplineGenerator:
    """Build a dense (x, y, slope) spline through a sequence of 2D waypoints."""

    def __init__(self) -> None:
        self.waypoints: np.ndarray | None = None
        self.spline_pts: np.ndarray | None = None
        self.orig_waypoints: np.ndarray | None = None

    @property
    def spline_data(self) -> np.ndarray:
        assert self.spline_pts is not None, "Call create_splines() first"
        return self.spline_pts.copy()

    @property
    def original_waypoints(self) -> np.ndarray:
        assert self.orig_waypoints is not None, "Call create_splines() first"
        return self.orig_waypoints

    def create_splines(self, waypts: np.ndarray) -> np.ndarray:
        waypts = np.asarray(waypts, dtype=float)
        self.waypoints = waypts.reshape((-1, 2))
        self.orig_waypoints = self.waypoints.copy()

        padded = np.insert(self.waypoints, 0, self.waypoints[0], axis=0)
        padded = np.insert(padded, -1, padded[-1], axis=0)

        splines: list[list[np.ndarray]] = []
        for j in range(len(padded) - 3):
            this_spline = self.cubic_spline(
                padded[j], padded[j + 1], padded[j + 2], padded[j + 3]
            )
            splines.append(this_spline)

        self.spline_pts = np.array(
            [pt for seg in splines for pt in seg], dtype=float
        ).reshape((-1, 3))
        return self.spline_pts

    @staticmethod
    def cubic_spline(
        y0: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        y3: np.ndarray,
        delt_mu: float = 0.001,
    ) -> list[np.ndarray]:
        # Catmull-Rom-like cubic: smooth interpolation between y1 and y2 using
        # y0/y3 as tangent neighbours. Returns a dense list of (x, y, heading).
        mu = 0.0
        points: list[np.ndarray] = []
        prev_x = 0.0
        prev_y = 0.0
        while mu <= 1.0:
            mu2 = mu * mu
            a0 = y3 - y2 - y0 + y1
            a1 = y0 - y1 - a0
            a2 = y2 - y0
            a3 = y1
            mu += delt_mu
            point = a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3
            slope = atan2(point[1] - prev_y, point[0] - prev_x)
            point = np.append(point, slope)
            points.append(point)
            prev_x = point[0]
            prev_y = point[1]
        return points
