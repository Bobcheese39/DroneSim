"""Bridge the thread-based :class:`RunManager` to a WebSocket stream.

``RunManager`` fires callbacks on its own worker thread. Each launched run
gets a :class:`RunSession` holding a thread-safe queue; the WebSocket handler
drains that queue (via an executor) and forwards events to the browser until a
``done`` event arrives.
"""
from __future__ import annotations

import queue
import threading
import uuid
from typing import Any

from dronesim.models import RunConfig, RunResult, ScenarioSpec
from dronesim.sim import BackendUnavailable, DroneFactory, RunManager
from dronesim.storage import RunStore


class RunSession:
    def __init__(self, token: str) -> None:
        self.token = token
        self.events: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.manager = RunManager(factory=DroneFactory())
        self.finished = threading.Event()

    def emit(self, event: dict[str, Any]) -> None:
        self.events.put(event)


class RunSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, RunSession] = {}
        self._lock = threading.Lock()

    def create(self) -> RunSession:
        token = uuid.uuid4().hex
        session = RunSession(token)
        with self._lock:
            self._sessions[token] = session
        return session

    def get(self, token: str) -> RunSession | None:
        with self._lock:
            return self._sessions.get(token)

    def remove(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)


_REGISTRY = RunSessionRegistry()


def registry() -> RunSessionRegistry:
    return _REGISTRY


def start_single_run(
    session: RunSession,
    scenario: ScenarioSpec,
    run_config: RunConfig,
    run_store: RunStore,
) -> None:
    """Launch one run, pushing progress/result/done events onto the session."""

    def _on_step_progress(step: int, total: int) -> None:
        session.emit({"type": "step_progress", "step": int(step), "total": int(total)})

    def _on_result(idx: int, run: RunResult) -> None:
        path: str | None = None
        try:
            path = str(run_store.save(run))
        except Exception as exc:  # noqa: BLE001 - surface to client
            session.emit({"type": "log", "level": "err", "message": f"Failed to save run: {exc}"})
        session.emit(
            {
                "type": "result",
                "run_id": run.run_id,
                "scenario_id": run.scenario_id,
                "status": run.status,
                "success": bool(run.summary.success),
                "miss_distance_m": run.summary.miss_distance_m,
                "n_steps": len(run.time_s),
                "path": path,
            }
        )

    def _on_error(idx: int, exc: BaseException) -> None:
        msg = str(exc) if isinstance(exc, BackendUnavailable) else repr(exc)
        session.emit({"type": "error", "message": msg})

    def _on_done() -> None:
        session.emit({"type": "done"})
        session.finished.set()

    session.manager.start_single(
        scenario,
        run_config,
        on_result=_on_result,
        on_step_progress=_on_step_progress,
        on_error=_on_error,
        on_done=_on_done,
    )
