"""Named JSBSim aircraft catalog with per-model autopilot tuning."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from dronesim.models import read_json

_AIRCRAFT_PATH = Path(__file__).resolve().parent / "jsbsim_aircraft.json"
_CACHE: dict[str, Any] | None = None


def load_jsbsim_aircraft(*, reload: bool = False) -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE
    payload = read_json(_AIRCRAFT_PATH)
    _CACHE = payload.get("aircraft", {})
    return _CACHE


def list_jsbsim_aircraft() -> list[dict[str, str]]:
    catalog = load_jsbsim_aircraft()
    rows: list[dict[str, str]] = []
    for aircraft_id, body in catalog.items():
        rows.append({
            "id": aircraft_id,
            "label": str(body.get("label", aircraft_id)),
            "description": str(body.get("description", "")),
            "jsbsim_model": str(body.get("jsbsim_model", aircraft_id)),
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


def apply_jsbsim_aircraft(
    scenario_dict: dict[str, Any],
    aircraft_id: str,
) -> dict[str, Any]:
    """Merge a catalog aircraft into a scenario-shaped dict (vehicle only)."""
    catalog = load_jsbsim_aircraft()
    if aircraft_id not in catalog:
        raise KeyError(f"Unknown JSBSim aircraft: {aircraft_id}")
    entry = catalog[aircraft_id]
    result = copy.deepcopy(scenario_dict)
    vehicle = result.setdefault("vehicle", {})
    if "vehicle" in entry:
        vehicle = _deep_merge(vehicle, entry["vehicle"])
        result["vehicle"] = vehicle
    result.setdefault("metadata", {})["jsbsim_aircraft"] = aircraft_id
    return result
