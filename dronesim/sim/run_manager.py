"""Backend-agnostic background run manager.

The manager preserves the Monte-Carlo style streaming/cancellation ergonomics
from ``simulations/6DOF_Quadcopter_MPC-main/drone_mc/monte_carlo.py`` but
operates on the normalized :class:`dronesim.models.ScenarioSpec` /
:class:`dronesim.models.RunResult` contracts so it works with any registered
backend.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time as _time
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from dronesim.models import RunConfig, RunResult, ScenarioSpec, new_id
from dronesim.sim.backends import DroneFactory
from dronesim.sim.debug_log import attach_worker_debug_logging, drain_worker_log_queue, get_sim_logger

logger = get_sim_logger(__name__)

# Parent of the vendored ``drone_mc`` package so spawned worker processes
# (Windows / macOS-spawn) can ``import drone_mc`` when the in-house backend
# is selected. Mirrors the trick in drone_mc.monte_carlo.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_VENDOR_ROOT = str(Path(_PROJECT_ROOT) / "simulations" / "6DOF_Quadcopter_MPC-main")


ProgressCb = Callable[[int, int], None]
StepProgressCb = Callable[[int, int], None]
ResultCb = Callable[[int, RunResult], None]
ErrorCb = Callable[[int, BaseException], None]
DoneCb = Callable[[], None]


def _ensure_pythonpath_for_workers() -> None:
    """Prepend the project + vendor roots to ``PYTHONPATH`` for spawned workers.

    Runtime ``sys.path`` mutations in the parent process do not propagate to
    workers spawned by ``ProcessPoolExecutor`` on Windows / macOS-spawn, but
    the workers do inherit the parent's environment variables and Python
    populates ``sys.path`` from ``PYTHONPATH`` before unpickling the
    bootstrap payload.

    Both roots are needed:

    * ``_PROJECT_ROOT`` lets workers ``import dronesim`` (required for
      unpickling ``_worker_run`` itself -- a worker that cannot import the
      ``dronesim`` package will fail with ``ModuleNotFoundError`` before
      :func:`_worker_init` even runs).
    * ``_VENDOR_ROOT`` lets workers ``import drone_mc`` when the in-house
      MPC backend is selected.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    changed = False
    # ``insert`` in reverse order so the final ``parts`` starts with
    # ``_PROJECT_ROOT`` then ``_VENDOR_ROOT`` -- closest to the parent's
    # actual sys.path layout.
    for root in (_VENDOR_ROOT, _PROJECT_ROOT):
        if root not in parts:
            parts.insert(0, root)
            changed = True
    if not changed:
        logger.debug("Worker PYTHONPATH already contains project + vendor roots")
        return
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    logger.debug(
        "Prepended project+vendor roots to worker PYTHONPATH: %s | %s",
        _PROJECT_ROOT,
        _VENDOR_ROOT,
    )


def _worker_init(
    vendor_root: str,
    worker_log_queue: Any | None = None,
    project_root: str | None = None,
) -> None:
    """Belt-and-braces: also patch ``sys.path`` once the worker is alive."""
    # ``project_root`` is optional for backwards compatibility with older
    # pickled initargs; new spawns always pass it.
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.debug("Worker initialized sys.path with project root %s", project_root)
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
        logger.debug("Worker initialized sys.path with vendor root %s", vendor_root)
    if worker_log_queue is not None:
        attach_worker_debug_logging(worker_log_queue)
        logger.debug("Worker debug logging attached")


def _worker_run(backend_id: str, scenario_dict: dict, run_config_dict: dict) -> dict:
    """Top-level picklable entry point for ProcessPoolExecutor workers.

    Workers rebuild their own DroneFactory because backend instances are not
    guaranteed to survive pickling; the default registry covers all
    production-shipped backends.
    """
    run_id = run_config_dict.get("run_id", "?")
    trial_index = run_config_dict.get("monte_carlo", {}).get("trial_index", "?")
    logger.info(
        "Worker starting backend=%s run_id=%s trial=%s scenario=%s",
        backend_id,
        run_id,
        trial_index,
        scenario_dict.get("scenario_id", "?"),
    )
    backend = DroneFactory().get(backend_id)
    scenario = ScenarioSpec.from_dict(scenario_dict)
    run_config = RunConfig(**run_config_dict)
    result = backend.run(scenario, run_config)
    logger.info(
        "Worker finished backend=%s run_id=%s status=%s success=%s",
        backend_id,
        run_id,
        result.status,
        result.summary.success,
    )
    return result.to_dict()


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


class RunManager:
    """Stream RunResults back to the caller via callbacks.

    Two entry points mirror the GUI's needs:

    * :meth:`start_single` runs one scenario in a daemon thread.
    * :meth:`start_monte_carlo` fans N trials out across a
      :class:`~concurrent.futures.ProcessPoolExecutor` (or thread pool for
      tests / CPU-light backends), assigning each trial a distinct ``seed``
      and ``monte_carlo['trial_index']``.

    Callbacks fire on the runner's internal thread; the GUI is expected to
    forward them onto a ``queue.Queue`` and drain that from the Panel
    callback loop, just like the existing :mod:`drone_mc.app.main` pattern.
    """

    def __init__(self, factory: DroneFactory | None = None) -> None:
        self.factory = factory or DroneFactory()
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[Executor] = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._results: list[RunResult] = []
        self._lock = threading.Lock()
        logger.debug("RunManager initialized")

    @property
    def results(self) -> list[RunResult]:
        with self._lock:
            return list(self._results)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        logger.warning("RunManager cancel requested (running=%s)", self.is_running)
        self._cancel.set()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
                logger.debug("Executor shutdown requested with cancel_futures=True")
            except Exception as exc:
                logger.warning("Executor shutdown failed during cancel: %s", exc)

    def join(self, timeout: Optional[float] = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def start_single(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        on_result: Optional[ResultCb] = None,
        on_progress: Optional[ProgressCb] = None,
        on_step_progress: Optional[StepProgressCb] = None,
        on_error: Optional[ErrorCb] = None,
        on_done: Optional[DoneCb] = None,
    ) -> None:
        """Run a single scenario in a background thread."""
        if self.is_running:
            logger.error("start_single rejected: RunManager already running")
            raise RuntimeError("RunManager is already in progress")
        cfg = run_config or scenario.run_config
        logger.info(
            "Starting single run scenario=%s run_id=%s backend=%s seed=%s",
            scenario.scenario_id,
            cfg.run_id,
            cfg.backend_id,
            cfg.seed,
        )

        def _target() -> None:
            try:
                backend = self.factory.get(cfg.backend_id)
                logger.debug(
                    "Single-run thread resolved backend=%s max_steps=%d dt=%.4f",
                    cfg.backend_id,
                    cfg.max_steps,
                    cfg.dt_s,
                )

                def _forward_step(step: int, total: int) -> None:
                    if on_step_progress is not None:
                        if step == 1 or step == total or step % max(1, total // 10) == 0:
                            logger.debug("Simulation step progress %d/%d", step, total)
                        self._safe_callback(on_step_progress, step, total)

                result = backend.run(scenario, cfg, on_step_progress=_forward_step)
                with self._lock:
                    self._results.append(result)
                logger.info(
                    "Single run finished run_id=%s status=%s success=%s",
                    result.run_id,
                    result.status,
                    result.summary.success,
                )
                if on_result is not None:
                    self._safe_callback(on_result, 0, result)
                if on_progress is not None:
                    self._safe_callback(on_progress, 1, 1)
            except BaseException as exc:  # noqa: BLE001 - surface everything to UI
                logger.exception("Single run failed run_id=%s: %s", cfg.run_id, exc)
                if on_error is not None:
                    self._safe_callback(on_error, 0, exc)
            finally:
                self._done.set()
                logger.debug("Single run thread exiting run_id=%s", cfg.run_id)
                if on_done is not None:
                    self._safe_callback(on_done)

        self._reset_state()
        self._thread = threading.Thread(target=_target, daemon=True, name="RunManagerSingle")
        self._thread.start()

    def start_monte_carlo(
        self,
        scenario: ScenarioSpec,
        run_config: RunConfig | None = None,
        *,
        n_trials: int,
        workers: int | None = None,
        base_seed: int = 0,
        executor: str = "process",
        on_result: Optional[ResultCb] = None,
        on_progress: Optional[ProgressCb] = None,
        on_error: Optional[ErrorCb] = None,
        on_done: Optional[DoneCb] = None,
    ) -> None:
        """Fan ``n_trials`` perturbed copies of ``run_config`` across workers."""
        if self.is_running:
            logger.error("start_monte_carlo rejected: RunManager already running")
            raise RuntimeError("RunManager is already in progress")
        if n_trials < 1:
            logger.error("start_monte_carlo rejected: n_trials=%s", n_trials)
            raise ValueError("n_trials must be >= 1")
        cfg = run_config or scenario.run_config
        trial_cfgs = self._build_trial_configs(cfg, n_trials=n_trials, base_seed=base_seed)
        worker_count = max(1, int(workers) if workers else _default_workers())
        logger.info(
            "Starting Monte Carlo scenario=%s backend=%s trials=%d workers=%d executor=%s base_seed=%d",
            scenario.scenario_id,
            cfg.backend_id,
            n_trials,
            worker_count,
            executor,
            base_seed,
        )
        logger.debug(
            "MC trial run_ids=%s",
            [trial_cfg.run_id for trial_cfg in trial_cfgs[:5]] + (["..."] if len(trial_cfgs) > 5 else []),
        )

        self._reset_state()
        self._thread = threading.Thread(
            target=self._run_pool,
            args=(scenario, trial_cfgs, worker_count, executor),
            kwargs={
                "on_result": on_result,
                "on_progress": on_progress,
                "on_error": on_error,
                "on_done": on_done,
            },
            daemon=True,
            name="RunManagerMC",
        )
        self._thread.start()

    @staticmethod
    def _build_trial_configs(
        base_cfg: RunConfig,
        *,
        n_trials: int,
        base_seed: int,
    ) -> list[RunConfig]:
        configs: list[RunConfig] = []
        for i in range(n_trials):
            mc = dict(base_cfg.monte_carlo)
            mc["trial_index"] = i
            configs.append(
                replace(
                    base_cfg,
                    run_id=new_id("run"),
                    seed=int(base_seed + i),
                    monte_carlo=mc,
                )
            )
        return configs

    def _reset_state(self) -> None:
        logger.debug("Resetting RunManager state")
        self._cancel.clear()
        self._done.clear()
        with self._lock:
            self._results.clear()

    def _run_pool(
        self,
        scenario: ScenarioSpec,
        trial_cfgs: list[RunConfig],
        workers: int,
        executor_kind: str,
        *,
        on_result: Optional[ResultCb],
        on_progress: Optional[ProgressCb],
        on_error: Optional[ErrorCb],
        on_done: Optional[DoneCb],
    ) -> None:
        total = len(trial_cfgs)
        use_processes = executor_kind != "thread"
        scenario_dict = scenario.to_dict() if use_processes else None
        backend_id = trial_cfgs[0].backend_id if trial_cfgs else "inhouse_mpc_quad"
        future_to_idx: dict[Future, int] = {}
        executor: Executor
        worker_log_queue: Any | None = None
        try:
            if use_processes:
                _ensure_pythonpath_for_workers()
                worker_log_queue = multiprocessing.Queue()
                logger.debug(
                    "Creating ProcessPoolExecutor workers=%d backend=%s",
                    workers,
                    backend_id,
                )
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_worker_init,
                    initargs=(_VENDOR_ROOT, worker_log_queue, _PROJECT_ROOT),
                )
            else:
                logger.debug(
                    "Creating ThreadPoolExecutor workers=%d backend=%s",
                    workers,
                    backend_id,
                )
                executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="RunMgrThr")
            self._executor = executor

            submitted = 0
            for i, cfg in enumerate(trial_cfgs):
                if self._cancel.is_set():
                    logger.warning("Stopped submitting MC trials after %d/%d due to cancel", submitted, total)
                    break
                if use_processes:
                    fut = executor.submit(
                        _worker_run,
                        backend_id,
                        scenario_dict,
                        _run_config_to_dict(cfg),
                    )
                else:
                    # Thread executor stays in-process: invoke the manager's own
                    # factory so test-registered or runtime-registered backends
                    # remain visible.
                    fut = executor.submit(self._invoke_backend, backend_id, scenario, cfg)
                future_to_idx[fut] = i
                submitted += 1
                logger.debug(
                    "Submitted MC trial idx=%d run_id=%s seed=%s",
                    i,
                    cfg.run_id,
                    cfg.seed,
                )
            logger.info("Submitted %d/%d Monte Carlo trial(s)", submitted, total)

            done = 0
            for fut in as_completed(future_to_idx):
                if worker_log_queue is not None:
                    drain_worker_log_queue(worker_log_queue)
                if self._cancel.is_set():
                    logger.warning("MC pool exiting early due to cancel (%d/%d complete)", done, total)
                    break
                idx = future_to_idx[fut]
                try:
                    payload = fut.result()
                except BaseException as exc:  # noqa: BLE001 - surface to UI
                    logger.exception("MC trial idx=%d failed: %s", idx, exc)
                    if on_error is not None:
                        self._safe_callback(on_error, idx, exc)
                else:
                    if isinstance(payload, RunResult):
                        result = payload
                    elif isinstance(payload, dict):
                        result = RunResult.from_dict(payload)
                    else:
                        logger.warning("MC trial idx=%d returned unexpected payload type %s", idx, type(payload))
                        continue
                    with self._lock:
                        self._results.append(result)
                    logger.info(
                        "MC trial idx=%d finished run_id=%s status=%s success=%s (%d/%d)",
                        idx,
                        result.run_id,
                        result.status,
                        result.summary.success,
                        done + 1,
                        total,
                    )
                    if on_result is not None:
                        self._safe_callback(on_result, idx, result)
                done += 1
                if on_progress is not None:
                    self._safe_callback(on_progress, done, total)
            if worker_log_queue is not None:
                drain_worker_log_queue(worker_log_queue)
        except BaseException as exc:  # noqa: BLE001 - executor setup or submission failure
            logger.exception("Monte Carlo pool failed during setup or execution: %s", exc)
            if on_error is not None:
                self._safe_callback(on_error, -1, exc)
        finally:
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    logger.debug("MC executor shutdown complete")
                except Exception as exc:
                    logger.warning("MC executor shutdown failed: %s", exc)
                self._executor = None
            self._done.set()
            logger.info(
                "Monte Carlo pool finished collected=%d cancelled=%s",
                len(self.results),
                self._cancel.is_set(),
            )
            if on_done is not None:
                self._safe_callback(on_done)

    def _invoke_backend(self, backend_id: str, scenario: ScenarioSpec, cfg: RunConfig) -> RunResult:
        logger.debug(
            "Thread-pool invoking backend=%s run_id=%s trial=%s",
            backend_id,
            cfg.run_id,
            cfg.monte_carlo.get("trial_index", 0),
        )
        return self.factory.get(backend_id).run(scenario, cfg)

    @staticmethod
    def _safe_callback(cb: Callable[..., Any], *args: Any) -> None:
        try:
            cb(*args)
        except Exception as exc:
            logger.warning("RunManager callback %s failed: %s", getattr(cb, "__name__", cb), exc)


def _run_config_to_dict(cfg: RunConfig) -> dict:
    """Convert RunConfig to a plain dict that survives a pickle round-trip."""
    return {
        "run_id": cfg.run_id,
        "backend_id": cfg.backend_id,
        "dt_s": cfg.dt_s,
        "max_steps": cfg.max_steps,
        "target_altitude_m": cfg.target_altitude_m,
        "horizon": cfg.horizon,
        "lookahead": cfg.lookahead,
        "waypoint_threshold_m": cfg.waypoint_threshold_m,
        "seed": cfg.seed,
        "integration_method": getattr(cfg, "integration_method", "euler"),
        "fidelity_mode": getattr(cfg, "fidelity_mode", "auto"),
        "monte_carlo": dict(cfg.monte_carlo),
    }


def wait_for(manager: RunManager, timeout: float = 30.0) -> bool:
    """Test helper that waits for a manager to finish or times out."""
    deadline = _time.monotonic() + timeout
    while manager.is_running and _time.monotonic() < deadline:
        _time.sleep(0.01)
    return not manager.is_running


__all__ = [
    "ProgressCb",
    "StepProgressCb",
    "ResultCb",
    "ErrorCb",
    "DoneCb",
    "RunManager",
    "wait_for",
]


def _iter_backend_ids(backends: Iterable[Any]) -> list[str]:
    """Internal helper kept for callers that want to advertise registered ids."""
    return [getattr(b, "backend_id", "") for b in backends]
