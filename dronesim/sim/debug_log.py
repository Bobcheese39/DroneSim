"""Debug logging helpers for simulation backends and the GUI console bridge."""
from __future__ import annotations

import logging
from typing import Any

SIM_LOGGER_NAMES = (
    "dronesim.sim.backends",
    "dronesim.sim.run_manager",
)

_LOG_FORMAT = "%(name)s | %(message)s"
_log_sink: Any | None = None


def get_sim_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger under the sim debug namespace."""
    return logging.getLogger(name)


def configure_sim_debug_logging(
    target_queue: Any | None = None,
    *,
    level: int = logging.DEBUG,
) -> None:
    """Attach queue forwarding handlers to sim loggers.

    When ``target_queue`` is provided, each log record is enqueued as
    ``("log", (console_level, message))`` for the GUI event drain loop.
    """
    global _log_sink
    _log_sink = target_queue

    handler: ForwardingLogHandler | None = None
    if target_queue is not None:
        handler = ForwardingLogHandler(target_queue)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    for logger_name in SIM_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers = [
            existing
            for existing in logger.handlers
            if not isinstance(existing, ForwardingLogHandler)
        ]
        if handler is not None:
            logger.addHandler(handler)


def attach_worker_debug_logging(worker_queue: Any) -> None:
    """Configure sim loggers inside a ProcessPoolExecutor worker."""
    handler = ForwardingLogHandler(worker_queue, wrap_event=False)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    for logger_name in SIM_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers = [handler]


def drain_worker_log_queue(worker_queue: Any) -> int:
    """Forward worker-process log records onto the GUI sink, if configured."""
    if _log_sink is None:
        return 0
    forwarded = 0
    while True:
        try:
            payload = worker_queue.get_nowait()
        except Exception:
            break
        try:
            _log_sink.put(("log", payload))
            forwarded += 1
        except Exception:
            break
    return forwarded


class ForwardingLogHandler(logging.Handler):
    """Emit formatted records to a queue for cross-thread/process forwarding."""

    def __init__(self, target_queue: Any, *, wrap_event: bool = True) -> None:
        super().__init__()
        self.target_queue = target_queue
        self.wrap_event = wrap_event

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            payload = (_console_level(record), message)
            if self.wrap_event:
                self.target_queue.put(("log", payload))
            else:
                self.target_queue.put(payload)
        except Exception:
            self.handleError(record)


def _console_level(record: logging.LogRecord) -> str:
    if record.levelno >= logging.ERROR:
        return "err"
    if record.levelno >= logging.WARNING:
        return "warn"
    if record.levelno <= logging.DEBUG:
        return "debug"
    return "ok"


__all__ = [
    "SIM_LOGGER_NAMES",
    "ForwardingLogHandler",
    "attach_worker_debug_logging",
    "configure_sim_debug_logging",
    "drain_worker_log_queue",
    "get_sim_logger",
]
