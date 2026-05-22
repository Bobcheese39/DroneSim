"""Process-pool Monte Carlo runner.

OSQP / cvxpy are CPU-bound, so true multi-process parallelism is required to
scale beyond a single core. The public API still mirrors a thread pool's
``submit``-style ergonomics through callback streams.
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import MCConfig, SimConfig
from .simulator import SimResult, run_simulation

# Parent of the ``drone_mc`` package — needed so spawned worker processes
# (Windows / macOS-spawn) can ``import drone_mc``.
_PKG_ROOT = str(Path(__file__).resolve().parents[1])


def _ensure_pythonpath_for_workers() -> None:
    """Prepend ``_PKG_ROOT`` to ``PYTHONPATH`` so spawned workers inherit it.

    Runtime ``sys.path`` mutations in the parent process do NOT propagate to
    workers spawned by ``ProcessPoolExecutor`` on Windows/macOS-spawn. The
    workers do, however, inherit the parent's environment variables, and the
    Python interpreter populates ``sys.path`` from ``PYTHONPATH`` *before* it
    starts unpickling the bootstrap payload — which is the only way to fix
    the ``ModuleNotFoundError: No module named 'drone_mc'`` raised inside
    ``multiprocessing.spawn._main``.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if _PKG_ROOT in parts:
        return
    parts.insert(0, _PKG_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def _worker_init(pkg_root: str) -> None:
    """Belt-and-braces: also patch ``sys.path`` once the worker is alive."""
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _worker(cfg: SimConfig) -> SimResult:
    """Top-level (picklable) entry point for ProcessPoolExecutor."""
    return run_simulation(cfg)


ProgressCb = Callable[[int, int], None]
ResultCb = Callable[[int, SimResult], None]
ErrorCb = Callable[[int, BaseException], None]


class MonteCarloRunner:
    """Streams MC results back to the caller via callbacks.

    Usage:
        runner = MonteCarloRunner(mc_cfg, on_progress=..., on_result=...)
        runner.start()
        # ... runner runs in a background thread; UI thread drains queues ...
        runner.join()  # or runner.cancel() on user request
    """

    def __init__(
        self,
        mc_cfg: MCConfig,
        on_progress: Optional[ProgressCb] = None,
        on_result: Optional[ResultCb] = None,
        on_error: Optional[ErrorCb] = None,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        self.mc_cfg = mc_cfg
        self.on_progress = on_progress
        self.on_result = on_result
        self.on_error = on_error
        self.on_done = on_done

        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ProcessPoolExecutor] = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._results: list[SimResult] = []
        self._lock = threading.Lock()

    @property
    def results(self) -> list[SimResult]:
        with self._lock:
            return list(self._results)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("Runner is already in progress")
        self._cancel.clear()
        self._done.clear()
        with self._lock:
            self._results.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="MCRunner")
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def join(self, timeout: Optional[float] = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _build_trial_cfgs(self) -> list[SimConfig]:
        cfgs: list[SimConfig] = []
        for i in range(self.mc_cfg.n_trials):
            seed = self.mc_cfg.base_seed + i
            rng = np.random.default_rng(seed)
            sampled = self.mc_cfg.base.sample(rng)
            cfgs.append(replace(sampled, seed=seed, trial_index=i))
        return cfgs

    def _run(self) -> None:
        total = self.mc_cfg.n_trials
        cfgs = self._build_trial_cfgs()
        future_to_idx: dict[Future, int] = {}
        try:
            _ensure_pythonpath_for_workers()
            self._executor = ProcessPoolExecutor(
                max_workers=self.mc_cfg.workers,
                initializer=_worker_init,
                initargs=(_PKG_ROOT,),
            )
            for i, c in enumerate(cfgs):
                fut = self._executor.submit(_worker, c)
                future_to_idx[fut] = i

            done = 0
            for fut in as_completed(future_to_idx):
                if self._cancel.is_set():
                    break
                idx = future_to_idx[fut]
                try:
                    result = fut.result()
                except BaseException as exc:  # noqa: BLE001 - surface to UI
                    if self.on_error is not None:
                        try:
                            self.on_error(idx, exc)
                        except Exception:
                            pass
                else:
                    with self._lock:
                        self._results.append(result)
                    if self.on_result is not None:
                        try:
                            self.on_result(idx, result)
                        except Exception:
                            pass
                done += 1
                if self.on_progress is not None:
                    try:
                        self.on_progress(done, total)
                    except Exception:
                        pass
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            self._done.set()
            if self.on_done is not None:
                try:
                    self.on_done()
                except Exception:
                    pass
