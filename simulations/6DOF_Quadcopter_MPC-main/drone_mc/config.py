"""Dataclasses describing a single simulation and a Monte Carlo batch."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np


def _default_init_state() -> dict[str, float]:
    return {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "roll_dot": 0.0, "pitch_dot": 0.0, "yaw_dot": 0.0,
        "x_dot": 0.0, "y_dot": 0.0, "z_dot": 0.0,
        "x": 0.0, "y": 0.0, "z": 5.0,
    }


@dataclass
class SimConfig:
    """Configuration for one simulation run.

    All randomization stds default to ``0.0`` so a config without perturbations
    produces a deterministic legacy-equivalent run.
    """

    waypoints: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.5], [3.0, 3.0]]))
    altitude: float = 5.0
    dt: float = 0.1
    horizon: int = 20
    max_steps: int = 250
    waypt_thresh: float = 0.25
    lookahead: int = 60

    mass: float = 5.0
    Ix: float = 1.0
    Iy: float = 1.0
    Iz: float = 1.5

    init_state: dict = field(default_factory=_default_init_state)

    init_pos_std: float = 0.0
    init_vel_std: float = 0.0
    init_att_std: float = 0.0
    force_noise_std: float = 0.0
    mass_jitter_pct: float = 0.0
    inertia_jitter_pct: float = 0.0

    seed: Optional[int] = None
    trial_index: int = 0

    def sample(self, rng: np.random.Generator) -> "SimConfig":
        """Return a perturbed copy for a single Monte Carlo trial."""
        init = dict(self.init_state)
        init["x"] += float(rng.normal(0.0, self.init_pos_std))
        init["y"] += float(rng.normal(0.0, self.init_pos_std))
        init["z"] += float(rng.normal(0.0, self.init_pos_std))
        init["x_dot"] += float(rng.normal(0.0, self.init_vel_std))
        init["y_dot"] += float(rng.normal(0.0, self.init_vel_std))
        init["z_dot"] += float(rng.normal(0.0, self.init_vel_std))
        init["roll"] += float(rng.normal(0.0, self.init_att_std))
        init["pitch"] += float(rng.normal(0.0, self.init_att_std))
        init["yaw"] += float(rng.normal(0.0, self.init_att_std))

        mass = self.mass * (1.0 + float(rng.normal(0.0, self.mass_jitter_pct)))
        Ix = self.Ix * (1.0 + float(rng.normal(0.0, self.inertia_jitter_pct)))
        Iy = self.Iy * (1.0 + float(rng.normal(0.0, self.inertia_jitter_pct)))
        Iz = self.Iz * (1.0 + float(rng.normal(0.0, self.inertia_jitter_pct)))

        # Clamp to physically plausible positives so the dynamics stay sane.
        mass = max(0.1, mass)
        Ix = max(0.05, Ix)
        Iy = max(0.05, Iy)
        Iz = max(0.05, Iz)

        return replace(self, init_state=init, mass=mass, Ix=Ix, Iy=Iy, Iz=Iz)


@dataclass
class MCConfig:
    base: SimConfig
    n_trials: int = 100
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    base_seed: int = 0
