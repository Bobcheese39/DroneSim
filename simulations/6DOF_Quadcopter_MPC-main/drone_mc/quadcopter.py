"""6-DOF quadcopter dynamics with parameterizable mass / inertia / dt.

Refactored from the legacy ``Quadcopter`` class so that all simulation knobs
live on the instance instead of in module globals. This is a prerequisite for
running many copies in parallel worker processes.
"""
from __future__ import annotations

from math import cos as c
from math import sin as s
from typing import Any

import control
import numpy as np


class Quadcopter:
    """Linear (for MPC) + nonlinear (for simulation) quadcopter model.

    The linearization point is hover; ``A`` and ``B`` are the same Jacobians
    used by the legacy script. ``update_states`` integrates the nonlinear EOMs
    via forward Euler with this instance's ``dt``.
    """

    STATE_DIM = 12
    CONTROL_DIM = 4

    def __init__(self, dt: float = 0.1, **init_kwargs: Any) -> None:
        self.dt = dt

        self.Ix = float(init_kwargs.get("Ix", 1.0))
        self.Iy = float(init_kwargs.get("Iy", 1.0))
        self.Iz = float(init_kwargs.get("Iz", 1.5))
        self.g = float(init_kwargs.get("g", 9.8))
        self.m = float(init_kwargs.get("mass", init_kwargs.get("m", 5.0)))

        self.x_dot = float(init_kwargs.get("x_dot", 0.0))
        self.y_dot = float(init_kwargs.get("y_dot", 0.0))
        self.z_dot = float(init_kwargs.get("z_dot", 0.0))
        self.x = float(init_kwargs.get("x", 0.0))
        self.y = float(init_kwargs.get("y", 0.0))
        self.z = float(init_kwargs.get("z", 0.0))

        self.roll_dot = float(init_kwargs.get("roll_dot", 0.0))
        self.pitch_dot = float(init_kwargs.get("pitch_dot", 0.0))
        self.yaw_dot = float(init_kwargs.get("yaw_dot", 0.0))
        self.roll = float(init_kwargs.get("roll", 0.0))
        self.pitch = float(init_kwargs.get("pitch", 0.0))
        self.yaw = float(init_kwargs.get("yaw", 0.0))

        self.A_zoh = np.eye(self.STATE_DIM)
        self.B_zoh = np.zeros((self.STATE_DIM, self.CONTROL_DIM))

    @property
    def state_vector(self) -> np.ndarray:
        return np.array(
            [
                self.roll, self.pitch, self.yaw,
                self.roll_dot, self.pitch_dot, self.yaw_dot,
                self.x_dot, self.y_dot, self.z_dot,
                self.x, self.y, self.z,
            ],
            dtype=float,
        )

    @property
    def A(self) -> np.ndarray:
        A = np.zeros((self.STATE_DIM, self.STATE_DIM))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[6, 1] = -self.g
        A[7, 0] = self.g
        A[9, 6] = 1.0
        A[10, 7] = 1.0
        A[11, 8] = 1.0
        return A

    @property
    def B(self) -> np.ndarray:
        B = np.zeros((self.STATE_DIM, self.CONTROL_DIM))
        B[3, 1] = 1.0 / self.Ix
        B[4, 2] = 1.0 / self.Iy
        B[5, 3] = 1.0 / self.Iz
        B[8, 0] = 1.0 / self.m
        return B

    @property
    def C(self) -> np.ndarray:
        return np.eye(self.STATE_DIM)

    @property
    def D(self) -> np.ndarray:
        return np.zeros((self.STATE_DIM, self.CONTROL_DIM))

    @property
    def Q(self) -> np.ndarray:
        Q = np.eye(self.STATE_DIM)
        Q[8, 8] = 5.0    # z vel
        Q[9, 9] = 10.0   # x pos
        Q[10, 10] = 10.0  # y pos
        Q[11, 11] = 100.0  # z pos
        return Q

    @property
    def R(self) -> np.ndarray:
        return np.eye(self.CONTROL_DIM) * 0.001

    def zoh(self) -> None:
        sys = control.StateSpace(self.A, self.B, self.C, self.D)
        sys_discrete = control.c2d(sys, self.dt, method="zoh")
        self.A_zoh = np.array(sys_discrete.A)
        self.B_zoh = np.array(sys_discrete.B)

    def update_states(self, ft: float, tx: float, ty: float, tz: float) -> None:
        """Forward-Euler integrate nonlinear EOMs.

        Equations follow Sabatino, KTH 2015
        (https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf).
        """
        roll_ddot = ((self.Iy - self.Iz) / self.Ix) * (self.pitch_dot * self.yaw_dot) + tx / self.Ix
        pitch_ddot = ((self.Iz - self.Ix) / self.Iy) * (self.roll_dot * self.yaw_dot) + ty / self.Iy
        yaw_ddot = ((self.Ix - self.Iy) / self.Iz) * (self.roll_dot * self.pitch_dot) + tz / self.Iz
        x_ddot = -(ft / self.m) * (s(self.roll) * s(self.yaw) + c(self.roll) * c(self.yaw) * s(self.pitch))
        y_ddot = -(ft / self.m) * (c(self.roll) * s(self.yaw) * s(self.pitch) - c(self.yaw) * s(self.roll))
        z_ddot = -1 * (self.g - (ft / self.m) * (c(self.roll) * c(self.pitch)))

        dt = self.dt
        self.roll_dot += roll_ddot * dt
        self.roll += self.roll_dot * dt
        self.pitch_dot += pitch_ddot * dt
        self.pitch += self.pitch_dot * dt
        self.yaw_dot += yaw_ddot * dt
        self.yaw += self.yaw_dot * dt

        self.x_dot += x_ddot * dt
        self.x += self.x_dot * dt
        self.y_dot += y_ddot * dt
        self.y += self.y_dot * dt
        self.z_dot += z_ddot * dt
        self.z += self.z_dot * dt

    def near(self, target_xy: np.ndarray, threshold: float) -> bool:
        return bool(np.linalg.norm(np.array([self.x, self.y]) - target_xy) <= threshold)

    def apply(self, ft: float = 0.0, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> None:
        """Apply control as deltas on top of the hover thrust."""
        hover = self.m * self.g
        self.update_states(hover + ft, tx, ty, tz)
