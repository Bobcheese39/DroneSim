"""Parameterized MPC controller for the linearized 6-DOF quadcopter.

The cvxpy ``Variable``s and ``Parameter``s live on the instance so that each
worker process can construct its own copy. Only the ``Parameter`` values are
mutated per-step so OSQP can warm-start.
"""
from __future__ import annotations

import numpy as np

import cvxpy as cp

from .quadcopter import Quadcopter

INF = np.inf

# Default state bounds matching the legacy script.
DEFAULT_XMIN = np.array(
    [-0.2, -0.2, -2 * np.pi, -0.25, -0.25, -0.25, -INF, -INF, -INF, -INF, -INF, -INF]
)
DEFAULT_XMAX = np.array(
    [0.2, 0.2, 2 * np.pi, 0.25, 0.25, 0.25, INF, INF, INF, INF, INF, INF]
)


class MpcController:
    """Linear-quadratic MPC over a finite horizon ``N``.

    A single ``cp.Problem`` is built once at construction time. Per-step we
    mutate ``x_init.value`` and ``xr.value`` and re-solve with warm starting.
    """

    def __init__(
        self,
        quad: Quadcopter,
        horizon: int = 20,
        xmin: np.ndarray | None = None,
        xmax: np.ndarray | None = None,
    ) -> None:
        self.N = int(horizon)
        nx = quad.STATE_DIM
        nu = quad.CONTROL_DIM

        self.xmin = DEFAULT_XMIN if xmin is None else np.asarray(xmin, dtype=float)
        self.xmax = DEFAULT_XMAX if xmax is None else np.asarray(xmax, dtype=float)

        # Decision variables / parameters. cvxpy Parameters are how we feed
        # changing data into a problem without rebuilding it.
        self._x = cp.Variable((nx, self.N + 1))
        self._u = cp.Variable((nu, self.N))
        self._x_init = cp.Parameter(nx)
        self._xr = cp.Parameter(nx)

        Q = quad.Q
        R = quad.R
        A = quad.A_zoh
        B = quad.B_zoh

        cost = 0
        constr = [self._x[:, 0] == self._x_init]
        for t in range(self.N):
            cost += cp.quad_form(self._xr - self._x[:, t], Q) + cp.quad_form(self._u[:, t], R)
            constr += [self.xmin <= self._x[:, t], self._x[:, t] <= self.xmax]
            constr += [self._x[:, t + 1] == A @ self._x[:, t] + B @ self._u[:, t]]
        cost += cp.quad_form(self._x[:, self.N] - self._xr, Q)

        self._problem = cp.Problem(cp.Minimize(cost), constr)
        self._A_zoh = A
        self._B_zoh = B

    def solve(self, x0: np.ndarray, xr: np.ndarray) -> np.ndarray:
        """Solve for the next control vector and return shape (CONTROL_DIM,).

        Falls back to a zero control vector if the solver fails so the
        Monte-Carlo loop does not crash on a single bad trial.
        """
        self._x_init.value = np.asarray(x0, dtype=float)
        self._xr.value = np.asarray(xr, dtype=float)
        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True)
        except cp.error.SolverError:
            return np.zeros(self._u.shape[0])
        u_val = self._u[:, 0].value
        if u_val is None:
            return np.zeros(self._u.shape[0])
        return np.asarray(u_val, dtype=float)

    def step_state(self, x0: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """Roll the discrete linear model forward one step (used by legacy x0)."""
        return self._A_zoh.dot(x0) + self._B_zoh.dot(u0)
