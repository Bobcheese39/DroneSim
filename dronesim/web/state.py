"""Shared service singletons and a process-local map-asset cache.

The web layer is otherwise stateless: scenarios travel in request bodies and
are persisted via :class:`ScenarioManager`. The only mutable server state is a
small cache of built :class:`MapAsset` objects (keyed by ``MapSpec.key()``) so
the Cesium imagery and heightmap endpoints can stream assets for the map the
client just built.
"""
from __future__ import annotations

import threading

from dronesim.services.scenario import ScenarioManager
from dronesim.services.scenario_editor import ScenarioEditor
from dronesim.services.terrain import MapAsset, TerrainService
from dronesim.sim import DroneFactory
from dronesim.storage import RunStore


class AppState:
    """Lazily-constructed, thread-safe container for shared services."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scenario_manager = ScenarioManager()
        self.terrain_service = TerrainService()
        self.scenario_editor = ScenarioEditor(terrain_service=self.terrain_service)
        self.factory = DroneFactory()
        self.run_store = RunStore()
        self._map_assets: dict[str, MapAsset] = {}

    def cache_map_asset(self, asset: MapAsset) -> str:
        key = asset.spec.key()
        with self._lock:
            self._map_assets[key] = asset
        return key

    def get_map_asset(self, key: str) -> MapAsset | None:
        with self._lock:
            return self._map_assets.get(key)


_STATE: AppState | None = None
_STATE_LOCK = threading.Lock()


def get_state() -> AppState:
    """Return the process-wide :class:`AppState`, creating it on first use."""
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = AppState()
    return _STATE
