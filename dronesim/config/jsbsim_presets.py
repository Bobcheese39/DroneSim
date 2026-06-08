"""Named JSBSim flight-profile presets (aircraft-agnostic)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from dronesim.models import read_json

_PRESETS_PATH = Path(__file__).resolve().parent / "jsbsim_presets.json"
_CACHE: dict[str, Any] | None = None


def load_jsbsim_presets(*, reload: bool = False) -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE
    payload = read_json(_PRESETS_PATH)
    _CACHE = payload.get("presets", {})
    return _CACHE


def list_jsbsim_presets() -> list[dict[str, str]]:
    presets = load_jsbsim_presets()
    rows: list[dict[str, str]] = []
    for preset_id, body in presets.items():
        rows.append({
            "id": preset_id,
            "label": str(body.get("label", preset_id)),
            "description": str(body.get("description", "")),
        })
    return rows


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_jsbsim_preset(
    scenario_dict: dict[str, Any],
    preset_id: str,
) -> dict[str, Any]:
    """Merge a named preset into a scenario-shaped dict (vehicle + run_config)."""
    presets = load_jsbsim_presets()
    if preset_id not in presets:
        raise KeyError(f"Unknown JSBSim preset: {preset_id}")
    preset = presets[preset_id]
    result = copy.deepcopy(scenario_dict)
    vehicle = result.setdefault("vehicle", {})
    if "vehicle" in preset:
        vehicle = _deep_merge(vehicle, preset["vehicle"])
        result["vehicle"] = vehicle
    run_config = result.setdefault("run_config", {})
    if "run_config" in preset:
        result["run_config"] = _deep_merge(run_config, preset["run_config"])
    result.setdefault("metadata", {})["jsbsim_preset"] = preset_id
    return result
