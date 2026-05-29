"""Persistence helpers for simulation run artifacts."""
from __future__ import annotations

import csv
from pathlib import Path

from dronesim.models import RunResult, read_json, write_json


class RunStore:
    """Save and load normalized run results."""

    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run: RunResult) -> Path:
        return self.runs_root / run.scenario_id / run.run_id

    def save(self, run: RunResult) -> Path:
        target_dir = self.run_dir(run)
        target_dir.mkdir(parents=True, exist_ok=True)
        write_json(target_dir / "run_result.json", run.to_dict())
        self._write_trajectory_csv(target_dir / "trajectory.csv", run)
        return target_dir

    def _write_trajectory_csv(self, path: Path, run: RunResult) -> None:
        rows = run.trajectory_rows()
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def load(self, path: str | Path) -> RunResult:
        p = Path(path)
        if p.is_dir():
            p = p / "run_result.json"
        return RunResult.from_dict(read_json(p))

    def list_runs(self, scenario_id: str | None = None) -> list[Path]:
        root = self.runs_root / scenario_id if scenario_id else self.runs_root
        if not root.exists():
            return []
        return sorted(root.glob("**/run_result.json"))

    def list_run_summaries(self, scenario_id: str | None = None) -> list[dict[str, object]]:
        """Return lightweight run metadata without loading full time series."""
        summaries: list[dict[str, object]] = []
        for path in self.list_runs(scenario_id):
            try:
                payload = read_json(path)
            except Exception:
                continue
            summary = payload.get("summary", {})
            summaries.append({
                "run_id": payload.get("run_id", path.parent.name),
                "scenario_id": payload.get("scenario_id", ""),
                "backend": payload.get("backend_id", ""),
                "status": payload.get("status", "unknown"),
                "success": summary.get("success", False),
                "miss_m": summary.get("miss_distance_m"),
                "created_utc": payload.get("created_utc", ""),
                "path": str(path.parent),
            })
        return summaries

    def load_scenario_runs(self, scenario_id: str) -> list[RunResult]:
        """Load all runs for a scenario, newest first."""
        runs: list[RunResult] = []
        for path in reversed(self.list_runs(scenario_id)):
            try:
                runs.append(self.load(path))
            except Exception:
                continue
        return runs
