"""3DOF point-mass flight dynamics for quadcopter and fixed-wing vehicles.

Both models share a common design philosophy: remove the high-frequency
rotational loops that make 6DOF models hard to tune and replace them with
kinematic guidance laws that produce smooth, oscillation-free trajectories
suitable for preliminary analysis and parameter selection.

Quadcopter (cartesian):
    A PD position controller produces a commanded acceleration that is
    clamped to a configurable limit.  The resulting equations are three
    double-integrators in x/y/z, analytically stable and drift-free.

Fixed-wing (speed-gamma-track):
    A coordinated point-mass model parameterised by airspeed V, flight-path
    angle gamma, and course/track angle chi.  Rate limits on each quantity
    prevent abrupt manoeuvres and keep the trajectory smooth.

Integration
-----------
Both models are stepped via RK4 at the caller-supplied dt.

Reused helpers
--------------
:class:`~dronesim.sim.fidelity.WindField` and
:class:`~dronesim.sim.fidelity.TerrainCollision` are imported from the
existing fidelity module so that wind + terrain-collision behaviour is
identical to the extended 6DOF path.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from dronesim.sim.fidelity import TerrainCollision, WindField

StepProgressCb = Callable[[int, int], None]

_G = 9.80665  # m/s^2, standard gravity


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QuadPointMassParams:
    """Parameters for the 3DOF cartesian quadcopter plant."""

    mass: float = 5.0
    max_accel_mps2: float = 5.0
    max_speed_mps: float = 10.0
    kp_pos: float = 1.2
    kd_pos: float = 1.4
    waypoint_capture_radius_m: float = 0.5
    target_altitude_m: float = 5.0

    @classmethod
    def from_dict(cls, params: dict[str, Any], *, target_altitude_m: float = 5.0) -> "QuadPointMassParams":
        return cls(
            mass=float(params.get("mass", 5.0)),
            max_accel_mps2=float(params.get("max_accel_mps2", 5.0)),
            max_speed_mps=float(params.get("max_speed_mps", 10.0)),
            kp_pos=float(params.get("kp_pos", 1.2)),
            kd_pos=float(params.get("kd_pos", 1.4)),
            waypoint_capture_radius_m=float(params.get("waypoint_capture_radius_m", 0.5)),
            target_altitude_m=float(target_altitude_m),
        )


@dataclass
class FixedWingPointMassParams:
    """Parameters for the 3DOF speed-gamma-track fixed-wing plant."""

    cruise_speed_mps: float = 40.0
    min_speed_mps: float = 25.0
    max_speed_mps: float = 70.0
    max_bank_deg: float = 30.0
    max_climb_deg: float = 8.0
    max_descent_deg: float = 8.0
    turn_rate_limit_deg_s: float = 10.0
    climb_rate_limit_mps: float = 4.0
    heading_gain: float = 1.5
    altitude_gain: float = 0.03
    waypoint_capture_radius_m: float = 75.0
    target_altitude_m: float = 100.0

    @classmethod
    def from_dict(cls, params: dict[str, Any], *, target_altitude_m: float = 100.0) -> "FixedWingPointMassParams":
        ctl = params.get("controller") or {}
        return cls(
            cruise_speed_mps=float(params.get("cruise_speed_mps", ctl.get("cruise_speed_mps", 40.0))),
            min_speed_mps=float(params.get("min_speed_mps", 25.0)),
            max_speed_mps=float(params.get("max_speed_mps", 70.0)),
            max_bank_deg=float(params.get("max_bank_deg", ctl.get("max_bank_deg", 30.0))),
            max_climb_deg=float(params.get("max_climb_deg", 8.0)),
            max_descent_deg=float(params.get("max_descent_deg", 8.0)),
            turn_rate_limit_deg_s=float(params.get("turn_rate_limit_deg_s", 10.0)),
            climb_rate_limit_mps=float(params.get("climb_rate_limit_mps", 4.0)),
            heading_gain=float(params.get("heading_gain", ctl.get("heading_gain", 1.5))),
            altitude_gain=float(params.get("altitude_gain", ctl.get("altitude_gain", 0.03))),
            waypoint_capture_radius_m=float(params.get("waypoint_capture_radius_m", ctl.get("waypoint_capture_radius_m", 75.0))),
            target_altitude_m=float(target_altitude_m),
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PointMassResult:
    """Normalised output of a 3DOF simulation run."""

    time_s: np.ndarray
    position_m: np.ndarray        # (N, 3) x/y/z
    velocity_mps: np.ndarray      # (N, 3)
    acceleration_mps2: np.ndarray # (N, 3)
    attitude_rad: np.ndarray      # (N, 3) roll/pitch/yaw – reconstructed
    angular_rate_rad_s: np.ndarray  # (N, 3) – zero for point-mass
    controls: np.ndarray          # (N, ?) backend-specific
    reference_position_m: np.ndarray  # (N, 3)
    tracking_error_m: np.ndarray  # (N,)
    success: bool = False
    miss_distance_m: float = float("nan")
    settle_steps: int = 0
    wallclock_s: float = 0.0
    terminated_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _angle_diff(target: float, current: float) -> float:
    """Signed shortest-path difference target - current in [-pi, pi]."""
    return _wrap_angle(target - current)


def _rk4_step(deriv_fn: Callable[[np.ndarray], np.ndarray], state: np.ndarray, dt: float) -> np.ndarray:
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5 * dt * k1)
    k3 = deriv_fn(state + 0.5 * dt * k2)
    k4 = deriv_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _step_report_interval(total_steps: int) -> int:
    if total_steps <= 500:
        return 1
    return max(1, total_steps // 200)


def _notify(cb: StepProgressCb | None, step: int, total: int) -> None:
    if cb is None:
        return
    try:
        cb(step, total)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Quadcopter 3DOF simulation
# ---------------------------------------------------------------------------


def run_quad_pointmass(
    waypoints_xy: np.ndarray,
    params: QuadPointMassParams,
    *,
    dt: float = 0.1,
    max_steps: int = 500,
    wind: WindField | None = None,
    collision: TerrainCollision | None = None,
    init_pos: np.ndarray | None = None,
    init_vel: np.ndarray | None = None,
    on_step_progress: StepProgressCb | None = None,
) -> PointMassResult:
    """Run a 3DOF cartesian point-mass quadcopter simulation.

    State vector: [x, y, z, vx, vy, vz]  (ENU, z-up)

    The guidance law is a simple clamped PD controller that issues a
    commanded acceleration.  Roll/pitch are reconstructed from the
    horizontal acceleration for display; yaw follows the velocity heading.

    Parameters
    ----------
    waypoints_xy:
        Array of shape ``(M, 2)`` or ``(M, 3)`` in local ENU metres.
        When 3-wide the third column is used as the per-waypoint target
        altitude; when 2-wide every waypoint is assigned
        ``params.target_altitude_m`` (backward-compatible default).
    """
    t_wall = _time.perf_counter()

    wind = wind or WindField()
    collision = collision or TerrainCollision()

    # Build ordered (M, 3) waypoint list — honour per-waypoint z when provided
    _wps = np.asarray(waypoints_xy, dtype=float)
    if _wps.ndim == 2 and _wps.shape[1] == 3:
        waypoints = _wps
    else:
        waypoints = np.column_stack([_wps, np.full(len(_wps), params.target_altitude_m)])

    # Initial state
    pos = init_pos.copy() if init_pos is not None else np.array(
        [waypoints[0, 0], waypoints[0, 1], waypoints[0, 2]], dtype=float
    )
    vel = init_vel.copy() if init_vel is not None else np.zeros(3, dtype=float)

    kp = params.kp_pos
    kd = params.kd_pos
    a_max = params.max_accel_mps2
    v_max = params.max_speed_mps
    cap_r = params.waypoint_capture_radius_m
    mass = params.mass

    T = max_steps
    out_pos = np.zeros((T, 3))
    out_vel = np.zeros((T, 3))
    out_acc = np.zeros((T, 3))
    out_att = np.zeros((T, 3))
    out_att_r = np.zeros((T, 3))
    out_ctrl = np.zeros((T, 4))   # [ft_N, tx_Nm, ty_Nm, tz_Nm]
    out_ref = np.zeros((T, 3))
    out_err = np.zeros(T)
    time_arr = np.arange(1, T + 1, dtype=float) * dt

    wp_idx = 1  # start navigating toward second waypoint
    success = False
    settle_steps = T
    written = 0
    terminated_by: str | None = None
    report_every = _step_report_interval(T)

    for i in range(T):
        if wp_idx >= len(waypoints):
            wp_idx = len(waypoints) - 1

        target = waypoints[wp_idx]
        ref_pos = target.copy()

        # Guidance: PD acceleration command
        pos_err = target - pos
        a_cmd = kp * pos_err - kd * vel
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm > a_max:
            a_cmd = a_cmd * (a_max / a_norm)

        # Wind disturbance
        w = wind.step(dt)

        def _quad_deriv(state: np.ndarray) -> np.ndarray:
            # state = [x, y, z, vx, vy, vz]
            v_state = state[3:6]
            # clamp speed
            v_speed = float(np.linalg.norm(v_state))
            if v_speed > v_max:
                v_state = v_state * (v_max / v_speed)
            return np.concatenate([v_state + w, a_cmd])

        state = np.concatenate([pos, vel])
        state_next = _rk4_step(_quad_deriv, state, dt)
        pos = state_next[:3]
        vel = state_next[3:6]

        # Speed cap after integration
        speed = float(np.linalg.norm(vel))
        if speed > v_max:
            vel = vel * (v_max / speed)

        # Reconstruct attitude from acceleration command for display
        ax_h, ay_h = float(a_cmd[0]), float(a_cmd[1])
        az_cmd = float(a_cmd[2])
        roll = math.atan2(-ay_h, _G + az_cmd)
        pitch = math.atan2(ax_h, _G + az_cmd)
        if speed > 0.05:
            yaw = math.atan2(float(vel[1]), float(vel[0]))
        else:
            yaw = 0.0

        # Thrust command for records
        ft_N = mass * float(np.linalg.norm(np.array([ax_h, ay_h, _G + az_cmd])))

        tracking_err = float(np.linalg.norm(pos - ref_pos))

        out_pos[i] = pos
        out_vel[i] = vel
        out_acc[i] = a_cmd
        out_att[i] = (roll, pitch, yaw)
        out_ctrl[i] = (ft_N, 0.0, 0.0, 0.0)
        out_ref[i] = ref_pos
        out_err[i] = tracking_err
        written = i + 1

        # Terrain collision
        if collision.is_active():
            ev = collision.check(i, time_arr[i], float(pos[0]), float(pos[1]), float(pos[2]))
            if ev is not None:
                terminated_by = "terrain_collision"
                settle_steps = written
                _notify(on_step_progress, written, T)
                break

        # Waypoint capture
        horiz_err = float(np.linalg.norm(pos[:2] - target[:2]))
        if horiz_err <= cap_r:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                success = True
                settle_steps = written
                _notify(on_step_progress, written, T)
                break

        if on_step_progress is not None and (written % report_every == 0 or written == T):
            _notify(on_step_progress, written, T)

    miss_distance = float(np.linalg.norm(out_pos[written - 1, :2] - waypoints[-1, :2])) if written > 0 else float("nan")

    return PointMassResult(
        time_s=time_arr[:written],
        position_m=out_pos[:written],
        velocity_mps=out_vel[:written],
        acceleration_mps2=out_acc[:written],
        attitude_rad=out_att[:written],
        angular_rate_rad_s=out_att_r[:written],
        controls=out_ctrl[:written],
        reference_position_m=out_ref[:written],
        tracking_error_m=out_err[:written],
        success=success,
        miss_distance_m=miss_distance,
        settle_steps=settle_steps,
        wallclock_s=_time.perf_counter() - t_wall,
        terminated_by=terminated_by,
        metadata={"model": "quad_pointmass_3dof"},
    )


# ---------------------------------------------------------------------------
# Fixed-wing 3DOF simulation
# ---------------------------------------------------------------------------


def run_fixedwing_pointmass(
    waypoints_xy: np.ndarray,
    params: FixedWingPointMassParams,
    *,
    dt: float = 0.05,
    max_steps: int = 2000,
    wind: WindField | None = None,
    collision: TerrainCollision | None = None,
    init_pos: np.ndarray | None = None,
    init_speed: float | None = None,
    init_chi: float | None = None,
    on_step_progress: StepProgressCb | None = None,
) -> PointMassResult:
    """Run a 3DOF speed-gamma-track point-mass fixed-wing simulation.

    State vector: [x, y, z, V, gamma, chi]  (ENU, z-up)

    - x, y, z: position in local ENU metres
    - V: airspeed (m/s)
    - gamma: flight-path angle (rad), positive = climbing
    - chi: course/track angle (rad), 0 = East, pi/2 = North

    Parameters
    ----------
    waypoints_xy:
        Array of shape ``(M, 2)`` or ``(M, 3)`` in local ENU metres.
        When 3-wide the third column is used as the per-waypoint target
        altitude; when 2-wide every waypoint is assigned
        ``params.target_altitude_m`` (backward-compatible default).

    Guidance
    --------
    * chi_cmd:   bearing to active waypoint, with heading_gain proportional
                 turn command rate-limited by turn_rate_limit_deg_s.
    * gamma_cmd: proportional to altitude error for each waypoint, clamped
                 to +/- max_climb/descent.
    * V:         held at cruise_speed_mps (constant speed model).

    Bank angle mu is derived from coordinated-turn kinematics and written to
    the attitude output for display.
    """
    t_wall = _time.perf_counter()

    wind = wind or WindField()
    collision = collision or TerrainCollision()

    # Build ordered (M, 3) waypoint list — honour per-waypoint z when provided
    _wps = np.asarray(waypoints_xy, dtype=float)
    if _wps.ndim == 2 and _wps.shape[1] == 3:
        waypoints = _wps
    else:
        waypoints = np.column_stack([_wps, np.full(len(_wps), params.target_altitude_m)])

    V0 = init_speed if init_speed is not None else params.cruise_speed_mps
    chi0 = init_chi if init_chi is not None else (
        math.atan2(
            float(waypoints[1, 1] - waypoints[0, 1]),
            float(waypoints[1, 0] - waypoints[0, 0]),
        ) if len(waypoints) > 1 else 0.0
    )

    if init_pos is not None:
        pos = init_pos.copy()
    else:
        pos = np.array([waypoints[0, 0], waypoints[0, 1], waypoints[0, 2]], dtype=float)

    V = float(np.clip(V0, params.min_speed_mps, params.max_speed_mps))
    gamma = 0.0
    chi = float(chi0)

    v_min = params.min_speed_mps
    v_max = params.max_speed_mps
    v_cruise = params.cruise_speed_mps
    mu_max = math.radians(params.max_bank_deg)
    gamma_max_cl = math.radians(params.max_climb_deg)
    gamma_max_ds = math.radians(params.max_descent_deg)
    turn_lim = math.radians(params.turn_rate_limit_deg_s)
    climb_lim = params.climb_rate_limit_mps
    h_gain = params.heading_gain
    a_gain = params.altitude_gain
    cap_r = params.waypoint_capture_radius_m

    T = max_steps
    out_pos = np.zeros((T, 3))
    out_vel = np.zeros((T, 3))
    out_acc = np.zeros((T, 3))
    out_att = np.zeros((T, 3))    # roll=bank, pitch=gamma, yaw=chi
    out_att_r = np.zeros((T, 3))
    out_ctrl = np.zeros((T, 4))   # [throttle_norm, aileron_norm, elevator_norm, rudder_norm]
    out_ref = np.zeros((T, 3))
    out_err = np.zeros(T)
    time_arr = np.arange(1, T + 1, dtype=float) * dt

    wp_idx = 1
    success = False
    settle_steps = T
    written = 0
    terminated_by: str | None = None
    report_every = _step_report_interval(T)

    prev_acc = np.zeros(3)

    for i in range(T):
        if wp_idx >= len(waypoints):
            wp_idx = len(waypoints) - 1

        target = waypoints[wp_idx]
        ref_pos = target.copy()

        # Guidance: compute desired chi from bearing to waypoint
        dx = float(target[0] - pos[0])
        dy = float(target[1] - pos[1])
        chi_des = math.atan2(dy, dx)
        chi_err = _angle_diff(chi_des, chi)
        chi_dot_cmd = _clamp(h_gain * chi_err, -turn_lim, turn_lim)

        # Guidance: desired gamma from altitude error
        alt_err = float(target[2] - pos[2])
        gamma_des = _clamp(a_gain * alt_err, -gamma_max_ds, gamma_max_cl)
        gamma_dot_cmd = _clamp((gamma_des - gamma) / max(dt, 1e-6), -climb_lim / max(V, 1e-3), climb_lim / max(V, 1e-3))
        gamma_new = float(np.clip(gamma + gamma_dot_cmd * dt, -gamma_max_ds, gamma_max_cl))

        # Chi update with rate limit
        chi_new = chi + chi_dot_cmd * dt

        # Bank angle from coordinated turn: tan(mu) = V * chi_dot / g
        mu = math.atan2(V * chi_dot_cmd, _G)
        mu = float(np.clip(mu, -mu_max, mu_max))

        # Speed held at cruise (constant speed model)
        V_new = v_cruise

        # Wind
        w = wind.step(dt)

        def _fw_deriv(state: np.ndarray) -> np.ndarray:
            _x, _y, _z, _V, _gamma, _chi = state
            cos_g = math.cos(_gamma)
            sin_g = math.sin(_gamma)
            cos_c = math.cos(_chi)
            sin_c = math.sin(_chi)
            xdot = _V * cos_g * cos_c + w[0]
            ydot = _V * cos_g * sin_c + w[1]
            zdot = _V * sin_g + w[2]
            # V, gamma, chi commanded directly (first-order update in rate)
            Vdot = (V_new - _V) / max(dt, 1e-6)
            gammadot = gamma_dot_cmd
            chidot = chi_dot_cmd
            return np.array([xdot, ydot, zdot, Vdot, gammadot, chidot])

        state = np.array([pos[0], pos[1], pos[2], V, gamma, chi])
        state_next = _rk4_step(_fw_deriv, state, dt)

        pos = state_next[:3]
        V = float(np.clip(state_next[3], v_min, v_max))
        gamma = float(np.clip(state_next[4], -gamma_max_ds, gamma_max_cl))
        chi = state_next[5]

        # Velocity in ENU
        cos_g = math.cos(gamma)
        vx = V * cos_g * math.cos(chi)
        vy = V * cos_g * math.sin(chi)
        vz = V * math.sin(gamma)
        vel = np.array([vx, vy, vz])

        # Acceleration approximated by finite difference
        if i == 0:
            acc = np.zeros(3)
        else:
            acc = (vel - np.array([
                V * math.cos(gamma) * math.cos(chi),  # same frame, reuse
                V * math.cos(gamma) * math.sin(chi),
                V * math.sin(gamma),
            ])) / dt
        prev_acc = acc

        # Attitude: roll=bank, pitch=flight-path, yaw=chi (ENU heading)
        att = np.array([mu, gamma, chi])

        # Throttle ~ normalised cruise reference (proxy control)
        throttle_norm = float(np.clip(V / v_max, 0.0, 1.0))
        aileron_norm = float(np.clip(mu / mu_max, -1.0, 1.0))
        elevator_norm = float(np.clip(gamma / max(gamma_max_cl, 1e-6), -1.0, 1.0))

        tracking_err = float(np.linalg.norm(pos - ref_pos))

        out_pos[i] = pos
        out_vel[i] = vel
        out_acc[i] = acc
        out_att[i] = att
        out_ctrl[i] = (throttle_norm, aileron_norm, elevator_norm, 0.0)
        out_ref[i] = ref_pos
        out_err[i] = tracking_err
        written = i + 1

        # Terrain collision
        if collision.is_active():
            ev = collision.check(i, time_arr[i], float(pos[0]), float(pos[1]), float(pos[2]))
            if ev is not None:
                terminated_by = "terrain_collision"
                settle_steps = written
                _notify(on_step_progress, written, T)
                break

        # Waypoint capture (horizontal distance only)
        horiz_err = float(np.linalg.norm(pos[:2] - target[:2]))
        if horiz_err <= cap_r:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                success = True
                settle_steps = written
                _notify(on_step_progress, written, T)
                break

        if on_step_progress is not None and (written % report_every == 0 or written == T):
            _notify(on_step_progress, written, T)

    miss_distance = float(np.linalg.norm(out_pos[written - 1, :2] - waypoints[-1, :2])) if written > 0 else float("nan")

    return PointMassResult(
        time_s=time_arr[:written],
        position_m=out_pos[:written],
        velocity_mps=out_vel[:written],
        acceleration_mps2=out_acc[:written],
        attitude_rad=out_att[:written],
        angular_rate_rad_s=out_att_r[:written],
        controls=out_ctrl[:written],
        reference_position_m=out_ref[:written],
        tracking_error_m=out_err[:written],
        success=success,
        miss_distance_m=miss_distance,
        settle_steps=settle_steps,
        wallclock_s=_time.perf_counter() - t_wall,
        terminated_by=terminated_by,
        metadata={"model": "fixedwing_pointmass_3dof"},
    )


__all__ = [
    "FixedWingPointMassParams",
    "PointMassResult",
    "QuadPointMassParams",
    "run_fixedwing_pointmass",
    "run_quad_pointmass",
]
