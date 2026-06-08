"""Configuration registries (presets, defaults)."""

from dronesim.config.jsbsim_aircraft import (
    apply_jsbsim_aircraft,
    list_jsbsim_aircraft,
    load_jsbsim_aircraft,
)
from dronesim.config.jsbsim_presets import (
    apply_jsbsim_preset,
    list_jsbsim_presets,
    load_jsbsim_presets,
)

__all__ = [
    "apply_jsbsim_aircraft",
    "apply_jsbsim_preset",
    "list_jsbsim_aircraft",
    "list_jsbsim_presets",
    "load_jsbsim_aircraft",
    "load_jsbsim_presets",
]
