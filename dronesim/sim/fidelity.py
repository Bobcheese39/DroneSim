"""Phase 6 fidelity wrappers around the vendored quadcopter dynamics.

This module layers a small set of optional disturbances on top of the
vendored :class:`drone_mc.quadcopter.Quadcopter` model without touching the
vendor source tree. The vendor MPC controller continues to issue
``(ft, tx, ty, tz)`` body-frame commands; we intercept the integration step
and add:

* aerodynamic drag (linear + quadratic in the body-relative wind)
* mean wind plus optional Ornstein-Uhlenbeck gust series
* RK4 integration of the same nonlinear EOMs (opt-in)
* terrain-collision detection sampled from a :class:`TerrainService` asset

Rotor allocation, motor lag, thrust/torque coefficients, actuator
saturation, battery sag and sensor noise are intentionally deferred --
they appear here only as documented TODO scaffolds so a later pass can
slot them in next to drag/wind without re-reading the plan.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import cos as _cos
from math import sin as _sin
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Lazy import: the vendored package is only importable once the backend has
# prepended its source root to ``sys.path``. Importing it at module load time
# would force every consumer (including the test runner) to set that up first.
_Quadcopter = None  # type: ignore[assignment]


def _ensure_vendor_path() -> None:
    """Mirror :meth:`InHouseMpcQuadBackend._ensure_vendor_path` for direct callers.

    Lets unit tests construct :class:`ExtendedQuadcopter` without first
    instantiating the backend.
    """
    root = Path(__file__).resolve().parents[2]
    vendor_root = root / "simulations" / "6DOF_Quadcopter_MPC-main"
    vendor_str = str(vendor_root)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def _quadcopter_cls():
    """Resolve the vendor Quadcopter class lazily."""
    global _Quadcopter
    if _Quadcopter is None:
        _ensure_vendor_path()
        from drone_mc.quadcopter import Quadcopter as _Q  # noqa: WPS433 - intentional lazy import

        _Quadcopter = _Q
    return _Quadcopter


TerrainQuery = Callable[[float, float], float]


@dataclass
class AeroParams:
    """Aerodynamic drag coefficients (zero == legacy / no drag)."""

    cd_linear: float = 0.0
    cd_quadratic: float = 0.0
    reference_area_m2: float = 0.1
    air_density_kg_m3: float = 1.225

    @classmethod
    def from_vehicle_params(
        cls, params: dict, air_density_kg_m3: float = 1.225
    ) -> "AeroParams":
        aero = params.get("aero") or {}
        return cls(
            cd_linear=float(aero.get("cd_linear", 0.0)),
            cd_quadratic=float(aero.get("cd_quadratic", 0.0)),
            reference_area_m2=float(aero.get("reference_area_m2", 0.1)),
            air_density_kg_m3=float(air_density_kg_m3),
        )

    def is_active(self) -> bool:
        return self.cd_linear > 0.0 or self.cd_quadratic > 0.0


@dataclass
class WindField:
    """Mean wind plus an optional discrete Ornstein-Uhlenbeck gust series.

    The OU process is fully described by ``std`` and ``tau``; setting
    ``std == 0`` recovers a deterministic mean-only wind.
    """

    mean_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    gust_std_mps: float = 0.0
    gust_decorrelation_s: float = 2.0
    rng: np.random.Generator | None = None

    _state: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.mean_mps = np.asarray(self.mean_mps, dtype=float).reshape(3)
        self._state = self.mean_mps.copy()

    @classmethod
    def from_environment(
        cls,
        wind_mps: Sequence[float] | None,
        gust_std_mps: float,
        gust_decorrelation_s: float,
        rng: np.random.Generator | None = None,
    ) -> "WindField":
        mean = np.zeros(3, dtype=float)
        if wind_mps is not None:
            arr = np.asarray(list(wind_mps), dtype=float).reshape(-1)
            mean[: min(3, arr.size)] = arr[: min(3, arr.size)]
        return cls(
            mean_mps=mean,
            gust_std_mps=float(gust_std_mps),
            gust_decorrelation_s=max(1e-6, float(gust_decorrelation_s)),
            rng=rng,
        )

    def is_active(self) -> bool:
        return bool(np.any(self.mean_mps != 0.0)) or self.gust_std_mps > 0.0

    def step(self, dt: float) -> np.ndarray:
        """Advance the wind state by ``dt`` and return the new wind vector.

        When ``gust_std_mps == 0`` the state stays equal to the mean wind so
        callers get a deterministic, allocation-free path.
        """
        if self.gust_std_mps <= 0.0 or self.rng is None:
            self._state = self.mean_mps.copy()
            return self._state
        tau = self.gust_decorrelation_s
        decay = float(np.exp(-dt / tau))
        diffusion = float(np.sqrt(max(0.0, 1.0 - decay * decay))) * self.gust_std_mps
        noise = self.rng.standard_normal(3) * diffusion
        self._state = self.mean_mps + (self._state - self.mean_mps) * decay + noise
        return self._state

    def current(self) -> np.ndarray:
        return self._state.copy()


@dataclass
class CollisionEvent:
    step_index: int
    time_s: float
    position_m: tuple[float, float, float]
    ground_elev_m: float
    clearance_m: float


@dataclass
class TerrainCollision:
    """Ground-impact detector backed by a TerrainService elevation lookup."""

    query: TerrainQuery | None = None
    offset_m: float = 0.5

    def is_active(self) -> bool:
        return self.query is not None

    def check(
        self,
        step_index: int,
        time_s: float,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> CollisionEvent | None:
        if self.query is None:
            return None
        try:
            ground = float(self.query(x_m, y_m))
        except Exception:
            return None
        clearance = z_m - ground
        if clearance <= self.offset_m:
            return CollisionEvent(
                step_index=step_index,
                time_s=time_s,
                position_m=(float(x_m), float(y_m), float(z_m)),
                ground_elev_m=ground,
                clearance_m=clearance,
            )
        return None


def _state_derivative(
    state: np.ndarray,
    ft: float,
    tx: float,
    ty: float,
    tz: float,
    *,
    mass: float,
    Ix: float,
    Iy: float,
    Iz: float,
    g: float,
    wind: np.ndarray,
    aero: AeroParams,
) -> np.ndarray:
    """Return d(state)/dt for the 12-state nonlinear quad model.

    State order matches the vendor Quadcopter / SimConfig convention::

        [roll, pitch, yaw,
         roll_dot, pitch_dot, yaw_dot,
         x_dot, y_dot, z_dot,
         x, y, z]
    """
    roll, pitch, yaw = state[0], state[1], state[2]
    rdot, pdot, ydot = state[3], state[4], state[5]
    vx, vy, vz = state[6], state[7], state[8]

    # Vendor body-torque rotational dynamics.
    roll_ddot = ((Iy - Iz) / Ix) * (pdot * ydot) + tx / Ix
    pitch_ddot = ((Iz - Ix) / Iy) * (rdot * ydot) + ty / Iy
    yaw_ddot = ((Ix - Iy) / Iz) * (rdot * pdot) + tz / Iz

    # Vendor translational dynamics with body-fixed thrust projected via the
    # Sabatino convention. Sign of the gravity term matches the legacy code:
    # z is altitude (positive up), and z_ddot = (ft/m) * cos(roll)*cos(pitch) - g.
    cr, cp_, sy = _cos(roll), _cos(pitch), _sin(yaw)
    sr, sp, cy = _sin(roll), _sin(pitch), _cos(yaw)
    ax = -(ft / mass) * (sr * sy + cr * cy * sp)
    ay = -(ft / mass) * (cr * sy * sp - cy * sr)
    az = -(g - (ft / mass) * (cr * cp_))

    # Phase 6: aerodynamic drag against body-relative wind.
    if aero.is_active():
        vrel = np.array([vx - wind[0], vy - wind[1], vz - wind[2]], dtype=float)
        speed = float(np.linalg.norm(vrel))
        # Linear + quadratic drag, opposing motion through the air mass.
        drag_force = -(aero.cd_linear * vrel + aero.cd_quadratic * speed * vrel)
        ax += drag_force[0] / mass
        ay += drag_force[1] / mass
        az += drag_force[2] / mass

    return np.array(
        [
            rdot,
            pdot,
            ydot,
            roll_ddot,
            pitch_ddot,
            yaw_ddot,
            ax,
            ay,
            az,
            vx,
            vy,
            vz,
        ],
        dtype=float,
    )


class ExtendedQuadcopter:
    """Drop-in replacement for the vendor Quadcopter with optional fidelity.

    The class deliberately mirrors the vendor public surface
    (``x``, ``y``, ``z``, ``x_dot`` ... ``state_vector``, ``apply``, ``near``,
    ``zoh``, ``A_zoh``, ``B_zoh``) so the MPC controller can construct it
    interchangeably. We compose with the vendor class rather than inherit
    because the vendor ``update_states`` is hard-coded to forward Euler with
    no wind / drag awareness.
    """

    STATE_DIM = 12
    CONTROL_DIM = 4

    def __init__(
        self,
        dt: float = 0.1,
        *,
        aero: AeroParams | None = None,
        wind: WindField | None = None,
        integration_method: str = "euler",
        collision: TerrainCollision | None = None,
        **init_kwargs,
    ) -> None:
        QuadCls = _quadcopter_cls()
        self._inner = QuadCls(dt=dt, **init_kwargs)
        self.aero = aero or AeroParams()
        self.wind = wind or WindField()
        method = (integration_method or "euler").lower()
        if method not in {"euler", "rk4"}:
            raise ValueError(
                f"Unknown integration_method '{integration_method}'. Use 'euler' or 'rk4'."
            )
        self.integration_method = method
        self.collision = collision or TerrainCollision()
        self.last_collision: CollisionEvent | None = None
        self._step_index = 0

    # ------------------------------------------------------------------
    # Forwarded attribute access so MPC / Tracker code keeps working.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def state_vector(self) -> np.ndarray:
        return self._inner.state_vector

    @property
    def A_zoh(self) -> np.ndarray:
        return self._inner.A_zoh

    @property
    def B_zoh(self) -> np.ndarray:
        return self._inner.B_zoh

    @property
    def A(self) -> np.ndarray:
        return self._inner.A

    @property
    def B(self) -> np.ndarray:
        return self._inner.B

    @property
    def Q(self) -> np.ndarray:
        return self._inner.Q

    @property
    def R(self) -> np.ndarray:
        return self._inner.R

    def zoh(self) -> None:
        self._inner.zoh()

    # ------------------------------------------------------------------
    # Position / velocity / attitude proxies.
    # ------------------------------------------------------------------
    def _get_state_array(self) -> np.ndarray:
        q = self._inner
        return np.array(
            [
                q.roll, q.pitch, q.yaw,
                q.roll_dot, q.pitch_dot, q.yaw_dot,
                q.x_dot, q.y_dot, q.z_dot,
                q.x, q.y, q.z,
            ],
            dtype=float,
        )

    def _set_state_array(self, state: np.ndarray) -> None:
        q = self._inner
        (
            q.roll, q.pitch, q.yaw,
            q.roll_dot, q.pitch_dot, q.yaw_dot,
            q.x_dot, q.y_dot, q.z_dot,
            q.x, q.y, q.z,
        ) = state.tolist()

    # ------------------------------------------------------------------
    # Integration step. Wraps vendor Quadcopter.update_states semantics.
    # ------------------------------------------------------------------
    def update_states(self, ft: float, tx: float, ty: float, tz: float) -> None:
        if self._is_legacy_step():
            # Bit-identical fallback: use the vendor's own forward-Euler step
            # so configurations without any fidelity knobs remain reproducible.
            self._inner.update_states(ft, tx, ty, tz)
            self._step_index += 1
            return

        dt = self._inner.dt
        wind = self.wind.step(dt)
        state = self._get_state_array()

        def deriv(s: np.ndarray) -> np.ndarray:
            return _state_derivative(
                s,
                ft,
                tx,
                ty,
                tz,
                mass=self._inner.m,
                Ix=self._inner.Ix,
                Iy=self._inner.Iy,
                Iz=self._inner.Iz,
                g=self._inner.g,
                wind=wind,
                aero=self.aero,
            )

        if self.integration_method == "rk4":
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt * k1)
            k3 = deriv(state + 0.5 * dt * k2)
            k4 = deriv(state + dt * k3)
            new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            new_state = state + dt * deriv(state)

        self._set_state_array(new_state)
        self._step_index += 1

    def apply(self, ft: float = 0.0, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> None:
        hover = self._inner.m * self._inner.g
        self.update_states(hover + ft, tx, ty, tz)

    def near(self, target_xy, threshold: float) -> bool:
        return self._inner.near(target_xy, threshold)

    # ------------------------------------------------------------------
    # Terrain collision is queried by the local runtime after each apply().
    # ------------------------------------------------------------------
    def check_terrain(self, step_index: int, time_s: float) -> CollisionEvent | None:
        event = self.collision.check(
            step_index,
            time_s,
            self._inner.x,
            self._inner.y,
            self._inner.z,
        )
        if event is not None:
            self.last_collision = event
        return event

    # ------------------------------------------------------------------
    def _is_legacy_step(self) -> bool:
        """Return True when no fidelity knob deviates from the legacy path."""
        if self.integration_method != "euler":
            return False
        if self.aero.is_active():
            return False
        if self.wind.is_active():
            return False
        return True


def build_extended_quad_from_sim_cfg(
    sim_cfg,
    *,
    aero: AeroParams,
    wind: WindField,
    integration_method: str,
    collision: TerrainCollision,
):
    """Construct an ExtendedQuadcopter mirroring drone_mc.simulator wiring.

    Mirrors the ``Quadcopter(dt=cfg.dt, **init_kwargs)`` call used inside
    :func:`drone_mc.simulator.run_simulation` so the local runtime can swap
    in this class without reimplementing the initialization conventions.
    """
    init_kwargs = dict(sim_cfg.init_state)
    init_kwargs.update(
        {
            "mass": sim_cfg.mass,
            "Ix": sim_cfg.Ix,
            "Iy": sim_cfg.Iy,
            "Iz": sim_cfg.Iz,
        }
    )
    return ExtendedQuadcopter(
        dt=sim_cfg.dt,
        aero=aero,
        wind=wind,
        integration_method=integration_method,
        collision=collision,
        **init_kwargs,
    )


__all__ = [
    "AeroParams",
    "CollisionEvent",
    "ExtendedQuadcopter",
    "TerrainCollision",
    "TerrainQuery",
    "WindField",
    "build_extended_quad_from_sim_cfg",
]
