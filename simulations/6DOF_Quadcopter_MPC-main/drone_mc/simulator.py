"""Pure-function simulation entry point used by the GUI and Monte Carlo runner."""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SimConfig
from .mpc import MpcController
from .quadcopter import Quadcopter
from .spline import SplineGenerator
from .tracking import LeaderState, Tracker


@dataclass
class SimResult:
    """All state/control time series plus summary statistics for a single run."""

    time: np.ndarray
    pos: np.ndarray          # (T, 3)  -> x, y, z
    vel: np.ndarray          # (T, 3)
    att: np.ndarray          # (T, 3)  -> roll, pitch, yaw
    att_rate: np.ndarray     # (T, 3)
    u: np.ndarray            # (T, 4)  -> ft, tx, ty, tz (ft includes hover)
    ref_xy: np.ndarray       # (T, 2)
    spline: np.ndarray       # (S, 3)  -> x, y, slope
    waypoints: np.ndarray    # (W, 2)
    success: bool
    miss_distance: float
    settle_steps: int
    seed: int
    trial_index: int
    wallclock_s: float
    cfg_summary: dict = field(default_factory=dict)


def _build_initial_state(cfg: SimConfig) -> np.ndarray:
    s = cfg.init_state
    return np.array([
        s["roll"], s["pitch"], s["yaw"],
        s["roll_dot"], s["pitch_dot"], s["yaw_dot"],
        s["x_dot"], s["y_dot"], s["z_dot"],
        s["x"], s["y"], s["z"],
    ], dtype=float)


def run_simulation(
    cfg: SimConfig,
    tracker: Optional[Tracker] = None,
) -> SimResult:
    """Run one MPC simulation and return the time series + summary stats.

    The function is deliberately self-contained (no module globals, no
    plotting, no I/O) so it can be invoked by ``ProcessPoolExecutor`` in a
    worker process.
    """
    t_wall_start = _time.perf_counter()
    seed = cfg.seed if cfg.seed is not None else cfg.trial_index
    rng = np.random.default_rng(seed)

    # Build trajectory.
    spline_gen = SplineGenerator()
    spline_data = spline_gen.create_splines(np.asarray(cfg.waypoints, dtype=float))
    spline_x = spline_data[:, 0]
    spline_y = spline_data[:, 1]

    # Build dynamics + controller.
    init_kwargs = dict(cfg.init_state)
    init_kwargs.update({"mass": cfg.mass, "Ix": cfg.Ix, "Iy": cfg.Iy, "Iz": cfg.Iz})
    quad = Quadcopter(dt=cfg.dt, **init_kwargs)
    quad.zoh()
    controller = MpcController(quad, horizon=cfg.horizon)

    x0 = _build_initial_state(cfg)
    final_waypt = np.asarray(cfg.waypoints[-1], dtype=float)

    T = int(cfg.max_steps)
    pos = np.zeros((T, 3))
    vel = np.zeros((T, 3))
    att = np.zeros((T, 3))
    att_rate = np.zeros((T, 3))
    u_hist = np.zeros((T, 4))
    ref_xy = np.zeros((T, 2))
    time_arr = np.arange(1, T + 1) * cfg.dt

    idx = cfg.lookahead
    success = False
    settle_steps = T
    written = 0
    desired_z = float(cfg.altitude)

    if tracker is not None:
        tracker.reset(0.0, LeaderState(t=0.0, x=quad.x, y=quad.y, z=desired_z))

    for i in range(T):
        # Pick the next reference: tracker takes precedence if supplied.
        if tracker is not None:
            leader = LeaderState(
                t=i * cfg.dt,
                x=float(spline_x[min(idx, len(spline_x) - 1)]),
                y=float(spline_y[min(idx, len(spline_y) - 1)]),
                z=desired_z,
            )
            ref = tracker.step(i * cfg.dt, leader)
            ref_x, ref_y = float(ref[0]), float(ref[1])
        else:
            ref_x = float(spline_x[min(idx, len(spline_x) - 1)])
            ref_y = float(spline_y[min(idx, len(spline_y) - 1)])

        if quad.near(np.array([ref_x, ref_y]), cfg.waypt_thresh):
            idx += cfg.lookahead
        idx = min(idx, len(spline_x) - 1)

        xr = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            ref_x, ref_y, desired_z,
        ])

        u_step = controller.solve(x0, xr)
        # MPC operates on the discrete linearization for warm-start continuity.
        x0 = controller.step_state(x0, u_step)

        # Inject process noise on the actual control sent to the nonlinear plant.
        if cfg.force_noise_std > 0.0:
            noise = rng.normal(0.0, cfg.force_noise_std, size=4)
        else:
            noise = np.zeros(4)
        ft, tx, ty, tz = (u_step + noise).tolist()
        quad.apply(ft, tx, ty, tz)

        pos[i] = (quad.x, quad.y, quad.z)
        vel[i] = (quad.x_dot, quad.y_dot, quad.z_dot)
        att[i] = (quad.roll, quad.pitch, quad.yaw)
        att_rate[i] = (quad.roll_dot, quad.pitch_dot, quad.yaw_dot)
        u_hist[i] = (quad.m * quad.g + ft, tx, ty, tz)
        ref_xy[i] = (ref_x, ref_y)
        written = i + 1

        if quad.near(final_waypt, 0.15):
            success = True
            settle_steps = i + 1
            break

    pos = pos[:written]
    vel = vel[:written]
    att = att[:written]
    att_rate = att_rate[:written]
    u_hist = u_hist[:written]
    ref_xy = ref_xy[:written]
    time_arr = time_arr[:written]

    miss_distance = float(np.linalg.norm(pos[-1, :2] - final_waypt))

    return SimResult(
        time=time_arr,
        pos=pos,
        vel=vel,
        att=att,
        att_rate=att_rate,
        u=u_hist,
        ref_xy=ref_xy,
        spline=spline_data,
        waypoints=spline_gen.original_waypoints,
        success=success,
        miss_distance=miss_distance,
        settle_steps=settle_steps,
        seed=seed,
        trial_index=cfg.trial_index,
        wallclock_s=_time.perf_counter() - t_wall_start,
        cfg_summary={
            "mass": cfg.mass, "Ix": cfg.Ix, "Iy": cfg.Iy, "Iz": cfg.Iz,
            "init_pos_std": cfg.init_pos_std,
            "init_vel_std": cfg.init_vel_std,
            "init_att_std": cfg.init_att_std,
            "force_noise_std": cfg.force_noise_std,
            "mass_jitter_pct": cfg.mass_jitter_pct,
            "inertia_jitter_pct": cfg.inertia_jitter_pct,
        },
    )
