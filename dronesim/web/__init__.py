"""FastAPI web application for DroneSim.

Serves the vanilla HTML/JS frontend and exposes the existing domain
services (scenarios, terrain/maps, editing, runs) over REST + WebSocket.

Run with:

    uvicorn dronesim.web.server:app --reload
"""
from __future__ import annotations

__all__ = ["server"]
