"""Local replica of ``drone_mc.simulator.run_simulation`` with fidelity hooks.

The vendor ``run_simulation`` entrypoint hard-codes ``Quadcopter(dt=cfg.dt,
**init_kwargs)``. To add Phase 6 fidelity (drag, wind, RK4, terrain
collision) without forking the vendored package, we re-implement the same
loop here and accept two hooks:

* ``dynamics_factory(sim_cfg) -> Quadcopter`` -- defaults to the vendor
  class, so the loop is bit-identical to vendor behavior when no extended
  dynamics are requested.
* ``terrain_query(x_m, y_m) -> ground_elev_m`` -- when provided, the loop
  checks ground clearance after each integrator step and terminates early
  with ``cfg_summary["terminated_by"] = "terrain_collision"``.

The returned :class:`drone_mc.simulator.SimResult` is the exact same type
used by the vendor path so downstream normalization in
:class:`dronesim.sim.backends.InHouseMpcQuadBackend` stays identical.
"""
from __future__ import annotations

import time as _time
from typing import Any, Callable

import numpy as np

from dronesim.sim.fidelity import (
    CollisionEvent,
    ExtendedQuadcopter,
    TerrainCollision,
    _ensure_vendor_path,
)

StepProgressCb = Callable[[int, int], None]
DynamicsFactory = Callable[[Any], Any]
TerrainQuery = Callable[[float, float], float]


def _step_report_interval(total_steps: int) -> int:
    if total_steps <= 500:
        return 1
    return max(1, total_steps // 200)


def _notify_step_progress(
    callback: StepProgressCb | None, current_step: int, total_steps: int
) -> None:
    if callback is None:
        return
    try:
        callback(current_step, total_steps)
    except Exception:
        # Mirrors vendor behavior: never let a GUI callback abort a sim.
        pass


def _build_initial_state(cfg) -> np.ndarray:
    s = cfg.init_state
    return np.array(
        [
            s["roll"], s["pitch"], s["yaw"],
            s["roll_dot"], s["pitch_dot"], s["yaw_dot"],
            s["x_dot"], s["y_dot"], s["z_dot"],
            s["x"], s["y"], s["z"],
        ],
        dtype=float,
    )


def run_simulation_local(
    cfg,
    *,
    dynamics_factory: DynamicsFactory | None = None,
    terrain_query: TerrainQuery | None = None,
    terrain_offset_m: float = 0.5,
    on_step_progress: StepProgressCb | None = None,
):
    """Run one MPC simulation using the supplied dynamics factory.

    Parameters mirror ``drone_mc.simulator.run_simulation``; extras let the
    backend layer inject ExtendedQuadcopter and a terrain elevation lookup.
    """
    # Vendor imports happen here so the function works even when called
    # before the backend has touched sys.path. ``_ensure_vendor_path`` is
    # idempotent and very cheap on subsequent calls.
    _ensure_vendor_path()
    from drone_mc.config import SimConfig  # noqa: F401  (type hint reference)
    from drone_mc.mpc import MpcController
    from drone_mc.quadcopter import Quadcopter
    from drone_mc.simulator import SimResult
    from drone_mc.spline import SplineGenerator

    t_wall_start = _time.perf_counter()
    seed = cfg.seed if cfg.seed is not None else cfg.trial_index
    rng = np.random.default_rng(seed)

    spline_gen = SplineGenerator()
    spline_data = spline_gen.create_splines(np.asarray(cfg.waypoints, dtype=float))
    spline_x = spline_data[:, 0]
    spline_y = spline_data[:, 1]

    if dynamics_factory is None:
        init_kwargs = dict(cfg.init_state)
        init_kwargs.update({"mass": cfg.mass, "Ix": cfg.Ix, "Iy": cfg.Iy, "Iz": cfg.Iz})
        quad = Quadcopter(dt=cfg.dt, **init_kwargs)
    else:
        quad = dynamics_factory(cfg)
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
    collision_event: CollisionEvent | None = None
    terminated_by: str | None = None

    report_every = _step_report_interval(T)

    for i in range(T):
        ref_x = float(spline_x[min(idx, len(spline_x) - 1)])
        ref_y = float(spline_y[min(idx, len(spline_y) - 1)])

        if quad.near(np.array([ref_x, ref_y]), cfg.waypt_thresh):
            idx += cfg.lookahead
        idx = min(idx, len(spline_x) - 1)

        xr = np.array(
            [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                ref_x, ref_y, desired_z,
            ]
        )

        u_step = controller.solve(x0, xr)
        x0 = controller.step_state(x0, u_step)

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

        # Terrain collision is queried after the integrator step so the
        # exact crash position is recorded in pos[i].
        if terrain_query is not None:
            # If the quad is an ExtendedQuadcopter, let it own the bookkeeping.
            if isinstance(quad, ExtendedQuadcopter) and quad.collision.is_active():
                collision_event = quad.check_terrain(i, time_arr[i])
            else:
                detector = TerrainCollision(query=terrain_query, offset_m=terrain_offset_m)
                collision_event = detector.check(
                    i, time_arr[i], float(quad.x), float(quad.y), float(quad.z)
                )

        current_step = i + 1
        if (
            on_step_progress is not None
            and (current_step % report_every == 0 or current_step == T)
        ):
            _notify_step_progress(on_step_progress, current_step, T)

        if collision_event is not None:
            terminated_by = "terrain_collision"
            settle_steps = current_step
            _notify_step_progress(on_step_progress, current_step, T)
            break

        if quad.near(final_waypt, 0.15):
            success = True
            settle_steps = current_step
            _notify_step_progress(on_step_progress, current_step, T)
            break

    pos = pos[:written]
    vel = vel[:written]
    att = att[:written]
    att_rate = att_rate[:written]
    u_hist = u_hist[:written]
    ref_xy = ref_xy[:written]
    time_arr = time_arr[:written]

    miss_distance = (
        float(np.linalg.norm(pos[-1, :2] - final_waypt)) if len(pos) else float("nan")
    )

    cfg_summary: dict[str, Any] = {
        "mass": cfg.mass,
        "Ix": cfg.Ix,
        "Iy": cfg.Iy,
        "Iz": cfg.Iz,
        "init_pos_std": cfg.init_pos_std,
        "init_vel_std": cfg.init_vel_std,
        "init_att_std": cfg.init_att_std,
        "force_noise_std": cfg.force_noise_std,
        "mass_jitter_pct": cfg.mass_jitter_pct,
        "inertia_jitter_pct": cfg.inertia_jitter_pct,
    }
    if terminated_by is not None:
        cfg_summary["terminated_by"] = terminated_by
    if collision_event is not None:
        cfg_summary["collision"] = {
            "step_index": collision_event.step_index,
            "time_s": collision_event.time_s,
            "position_m": list(collision_event.position_m),
            "ground_elev_m": collision_event.ground_elev_m,
            "clearance_m": collision_event.clearance_m,
        }

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
        cfg_summary=cfg_summary,
    )


__all__ = ["run_simulation_local", "DynamicsFactory", "TerrainQuery"]
