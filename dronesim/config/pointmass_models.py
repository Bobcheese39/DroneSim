"""Catalog loader for 3DOF point-mass model presets."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from dronesim.models import read_json

_MODELS_PATH = Path(__file__).resolve().parent / "pointmass_models.json"
_CACHE: dict[str, Any] | None = None


def _load_models(*, reload: bool = False) -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE
    payload = read_json(_MODELS_PATH)
    _CACHE = payload.get("models", {})
    return _CACHE


def list_pointmass_models() -> list[dict[str, str]]:
    """Return a summary list of available point-mass model presets."""
    catalog = _load_models()
    rows: list[dict[str, str]] = []
    for model_id, body in catalog.items():
        rows.append({
            "id": model_id,
            "label": str(body.get("label", model_id)),
            "description": str(body.get("description", "")),
            "model_type": str(body.get("model_type", "")),
        })
    return rows


def apply_pointmass_model(
    scenario_dict: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Merge a point-mass model preset into a scenario-shaped dict.

    Only the ``vehicle`` sub-tree is modified; everything else is preserved.
    """
    catalog = _load_models()
    if model_id not in catalog:
        raise KeyError(f"Unknown point-mass model: {model_id}")
    entry = catalog[model_id]
    result = copy.deepcopy(scenario_dict)
    if "vehicle" in entry:
        result["vehicle"] = copy.deepcopy(entry["vehicle"])
    result.setdefault("metadata", {})["pointmass_model"] = model_id
    return result


__all__ = [
    "apply_pointmass_model",
    "list_pointmass_models",
]
