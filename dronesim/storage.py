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
