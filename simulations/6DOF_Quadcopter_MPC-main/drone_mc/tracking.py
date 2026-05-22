"""Trajectory tracking interfaces (future fox/rabbit chase).

This module defines the seam ``run_simulation`` will use to plug in
tracking-style references instead of a fixed spline. Only a trivial
``OffsetFollower`` is implemented here so the wiring is testable; richer
fox-and-rabbit policies can land in a follow-up without touching the
simulator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cos, sin

import numpy as np


@dataclass
class LeaderState:
    """Position and heading of a moving target the drone should track."""

    t: float
    x: float
    y: float
    z: float
    yaw: float = 0.0


class Tracker(ABC):
    """Generates the next (x, y, z) reference given a leader's state."""

    @abstractmethod
    def reset(self, t0: float, leader: LeaderState) -> None: ...

    @abstractmethod
    def step(self, t: float, leader: LeaderState) -> np.ndarray:
        """Return shape (3,) reference position."""


class OffsetFollower(Tracker):
    """Trivial chase: hover at a fixed offset behind the leader."""

    def __init__(self, distance: float = 1.0, bearing_rad: float = np.pi) -> None:
        self.distance = float(distance)
        self.bearing = float(bearing_rad)

    def reset(self, t0: float, leader: LeaderState) -> None:
        del t0, leader

    def step(self, t: float, leader: LeaderState) -> np.ndarray:
        del t
        bx = self.distance * cos(leader.yaw + self.bearing)
        by = self.distance * sin(leader.yaw + self.bearing)
        return np.array([leader.x + bx, leader.y + by, leader.z], dtype=float)
