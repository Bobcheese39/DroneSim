# 6DOF_Quadcopter_MPC

Trajectory tracking of a 6-DOF quadcopter via model predictive control, plus a
Panel-based GUI for waypoint ingestion and Monte Carlo robustness studies.

![](https://github.com/TylerReimer13/6DOF_Quadcopter_MPC/blob/main/Viz/quadcopter_mpc_3d.gif)
![](https://github.com/TylerReimer13/6DOF_Quadcopter_MPC/blob/main/Viz/quadcopter_mpc_2d.gif)
![](https://github.com/TylerReimer13/6DOF_Quadcopter_MPC/blob/main/Viz/Quadcopter_MPC_Trajectory.png)

## Layout

- `quadcopter_MPC.py` — original monolithic script, kept for reference.
- `drone_mc/` — refactored library:
  - `quadcopter.py`, `spline.py`, `mpc.py` — dynamics, splines, MPC controller.
  - `simulator.py` — pure `run_simulation(cfg) -> SimResult` entry point.
  - `monte_carlo.py` — process-pool runner streaming results via callbacks.
  - `config.py` — `SimConfig` / `MCConfig` dataclasses with randomization knobs.
  - `waypoints.py` — CSV loader / validator + sample data helpers.
  - `visuals.py` — Plotly / Bokeh figure builders for the GUI.
  - `tracking.py` — seam for future fox-and-rabbit chase trackers.
  - `app/` — Panel GUI (`main.py`, `sidebar.py`, `theme.css`).
- `data/sample_trajectory.csv` — example waypoint file for the GUI.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Run the GUI

```bash
panel serve drone_mc/app/main.py --show --autoreload
```

The left sidebar lets you:

- Pick a waypoint source (sample, CSV upload, or click-to-place on a 2D grid).
- Tune simulation parameters (`dt`, MPC horizon, max steps, target altitude).
- Tune Monte Carlo parameters (trial count, workers, init / control noise σ,
  mass / inertia jitter, base seed).
- Hit **Run single** for a deterministic dry run, **Run Monte Carlo** for the
  full batch, or **Cancel** mid-run.

The main panel updates live as workers finish:

- **Trajectory** — Plotly 3D + Bokeh top-down with all completed trial paths.
- **Linear / Angular states** — per-axis time series with mean ± 1σ ribbons.
- **Control inputs** — thrust + body torques over time.
- **Statistics** — per-trial table + final-miss-distance histogram.
- **Console** — timestamped event log.

## Run programmatically

```python
from drone_mc import SimConfig, MCConfig, MonteCarloRunner, run_simulation

cfg = SimConfig()                # legacy demo waypoints by default
result = run_simulation(cfg)     # one deterministic run
print(result.success, result.miss_distance)

mc = MCConfig(base=cfg, n_trials=100, workers=4)
runner = MonteCarloRunner(mc, on_progress=lambda d, t: print(f"{d}/{t}"))
runner.start()
runner.join()
print(len(runner.results), "trials complete")
```

## Roadmap

- Fox / rabbit chase trackers (the `Tracker` interface in
  `drone_mc/tracking.py` is the seam — `run_simulation(cfg, tracker=...)`
  already routes references through it).
- Persisting MC runs to disk for offline post-processing.
