// Analysis view: uPlot time-series charts + summary/parameter tables.
// Supports single-run and Monte Carlo batch modes.

import { api } from "../api.js";
import { store, setStatus } from "../state.js";
import { makeLineChart, makeHistogramChart, makeEnvelopeChart, makeTrajectory3dChart, tableHtml } from "../charts.js";

export function renderAnalysis(ctx) {
  if (store.mcBatch?.analysis) {
    renderMcAnalysis(ctx);
    return;
  }
  renderSingleAnalysis();
}

function renderSingleAnalysis() {
  const stage = document.getElementById("analysis-stage");
  if (stage._analysisDestroy) stage._analysisDestroy();
  const inspector = document.getElementById("inspector");
  const run = store.run;

  if (!run || !run.analysis) {
    inspector.innerHTML = `<div class="insp-section"><h3 class="insp-title">Analysis</h3>
      <div class="card"><div class="hint">No run loaded. Run a scenario or load one from Runs.</div></div></div>`;
    stage.innerHTML = '<div class="empty">No analysis data. Run a scenario first.</div>';
    return;
  }

  const a = run.analysis;
  const x = a.time_s;

  inspector.innerHTML = `
    <div class="insp-section"><h3 class="insp-title">Summary</h3>
      <div class="card">${tableHtml(a.summary, [
        { key: "metric", label: "Metric" },
        { key: "value", label: "Value" },
      ])}</div></div>
    <div class="insp-section"><h3 class="insp-title">Parameters</h3>
      <div class="card">${tableHtml(a.parameters, [
        { key: "parameter", label: "Parameter" },
        { key: "value", label: "Value" },
      ])}</div></div>`;

  const charts = buildSingleCharts(a, x);
  renderSingleChartGrid(stage, "Analysis", charts, x);
}

function renderMcAnalysis(ctx) {
  const stage = document.getElementById("analysis-stage");
  const inspector = document.getElementById("inspector");
  const batch = store.mcBatch.analysis;
  const trials = batch.trials || store.mcBatch.trials || [];

  inspector.innerHTML = `
    <div class="insp-section"><h3 class="insp-title">MC Summary</h3>
      <div class="card">${tableHtml(batch.summary, [
        { key: "metric", label: "Metric" },
        { key: "value", label: "Value" },
      ])}</div></div>
    <div class="insp-section"><h3 class="insp-title">Trials</h3>
      <div class="card hint" style="margin-bottom:6px;">Click a row to replay one trial, or replay all at once.</div>
      <button class="btn full mc-replay-all" id="mc-replay-all" style="margin-bottom:8px;">Replay all trials</button>
      <div class="card" id="mc-trial-table">${tableHtml(trials, [
        { key: "trial_index", label: "Trial" },
        { key: "success", label: "OK" },
        { key: "miss_distance_m", label: "Miss (m)" },
        { key: "duration_s", label: "Duration (s)" },
        { key: "wallclock_s", label: "Wall (s)" },
      ])}</div></div>`;

  inspector.querySelector("#mc-replay-all").addEventListener("click", () => loadAllTrials(ctx, trials));

  inspector.querySelectorAll("#mc-trial-table .data-table tbody tr").forEach((tr, i) => {
    tr.classList.add("clickable");
    tr.addEventListener("click", () => loadTrial(ctx, trials[i]));
  });

  const env = batch.envelopes || {};
  const charts = [];

  if (batch.histogram?.bins?.length) {
    charts.push({
      type: "histogram",
      title: "Miss distance distribution",
      data: batch.histogram,
    });
  }
  if (env.tracking_error?.time_s?.length) {
    charts.push({
      type: "envelope",
      title: "Tracking error envelope",
      yLabel: "m",
      env: env.tracking_error,
    });
  }
  for (const [label, key] of [
    ["Velocity vx", "vx"],
    ["Velocity vy", "vy"],
    ["Velocity vz", "vz"],
  ]) {
    const e = env.velocity?.[key];
    if (e?.time_s?.length) {
      charts.push({ type: "envelope", title: label, yLabel: "m/s", env: e });
    }
  }
  for (const [label, key] of [
    ["Roll", "roll"],
    ["Pitch", "pitch"],
    ["Yaw", "yaw"],
  ]) {
    const e = env.attitude?.[key];
    if (e?.time_s?.length) {
      charts.push({ type: "envelope", title: `${label} envelope`, yLabel: "rad", env: e });
    }
  }

  stage.innerHTML =
    '<h2 class="section-head">Monte Carlo Analysis</h2><div class="chart-grid" id="chart-grid"></div>';
  const grid = document.getElementById("chart-grid");

  requestAnimationFrame(() => {
    charts.forEach((c) => {
      const card = document.createElement("div");
      card.className = "chart-card";
      const host = document.createElement("div");
      card.appendChild(host);
      grid.appendChild(card);
      if (c.type === "histogram") {
        makeHistogramChart(host, { title: c.title, bins: c.data.bins, counts: c.data.counts });
      } else {
        makeEnvelopeChart(host, { title: c.title, yLabel: c.yLabel, env: c.env });
      }
    });
  });
}

async function loadAllTrials(ctx, trials) {
  const paths = trials.map((t) => t.path).filter(Boolean);
  if (!paths.length) return;
  setStatus("Loading batch replay\u2026", "busy");
  try {
    store.mcBatch.replay = await api.mcReplay(paths);
    store.run = null;
    setStatus(`Loaded ${paths.length} trials for replay.`, "ok");
    ctx.setView("replay");
  } catch (e) {
    setStatus(`Batch replay failed: ${e.message}`, "err");
  }
}

async function loadTrial(ctx, trial) {
  const path = trial?.path;
  if (!path) return;
  try {
    store.run = await api.runResult(path);
    store.mcBatch = null;
    ctx.setView("replay");
  } catch (_e) {
    /* ignore */
  }
}

function buildSingleCharts(a, x) {
  const charts = [];

  const traj = a.trajectory;
  if (traj?.position_m?.length >= 2) {
    charts.push({
      type: "trajectory3d",
      title: "3D trajectory (XYZ)",
      trajectory: traj,
      wide: true,
    });
  }

  if (a.fuel_kg?.length) {
    charts.push({
      type: "line",
      title: "Fuel remaining",
      yLabel: "kg",
      series: [{ label: "fuel", values: a.fuel_kg }],
    });
  }

  if (a.battery_soc_pct?.length) {
    charts.push({
      type: "line",
      title: "Battery state of charge",
      yLabel: "%",
      series: [{ label: "soc", values: a.battery_soc_pct }],
    });
  }

  charts.push(
    { type: "line", title: "Tracking error", yLabel: "m", series: [{ label: "error", values: a.tracking_error_m }] },
    {
      type: "line",
      title: "Position error (actual \u2212 reference)",
      yLabel: "m",
      series: [
        { label: "ex", values: a.error_decomposition.ex },
        { label: "ey", values: a.error_decomposition.ey },
        { label: "ez", values: a.error_decomposition.ez },
      ],
    },
    {
      type: "line",
      title: "Velocity",
      yLabel: "m/s",
      series: [
        { label: "vx", values: a.velocity.vx },
        { label: "vy", values: a.velocity.vy },
        { label: "vz", values: a.velocity.vz },
      ],
    },
    {
      type: "line",
      title: "Acceleration",
      yLabel: "m/s\u00b2",
      series: [
        { label: "ax", values: a.acceleration.ax },
        { label: "ay", values: a.acceleration.ay },
        { label: "az", values: a.acceleration.az },
      ],
    },
    {
      type: "line",
      title: "Attitude",
      yLabel: "rad",
      series: [
        { label: "roll", values: a.attitude.roll },
        { label: "pitch", values: a.attitude.pitch },
        { label: "yaw", values: a.attitude.yaw },
      ],
    },
    {
      type: "line",
      title: "Angular rate",
      yLabel: "rad/s",
      series: [
        { label: "p", values: a.angular_rate.p },
        { label: "q", values: a.angular_rate.q },
        { label: "r", values: a.angular_rate.r },
      ],
    },
    {
      type: "line",
      title: "Control effort",
      yLabel: a.controls?.y_label === "norm" ? "norm" : "N / Nm",
      series: Array.isArray(a.controls?.series) && a.controls.series.length
        ? a.controls.series.map((row) => ({ label: row.label, values: row.values }))
        : [
            { label: "ft", values: a.controls.ft },
            { label: "tx", values: a.controls.tx },
            { label: "ty", values: a.controls.ty },
            { label: "tz", values: a.controls.tz },
          ],
    }
  );

  if (a.clearance_m) {
    charts.push({
      type: "line",
      title: "Terrain clearance",
      yLabel: "m",
      series: [
        { label: "clearance", values: a.clearance_m },
        { label: "1 m warn", values: a.clearance_m.map(() => 1.0), color: "#ff453a" },
      ],
    });
  }
  return charts;
}

function renderSingleChartGrid(stage, heading, charts, x) {
  stage.innerHTML = `<h2 class="section-head">${heading}</h2><div class="chart-grid" id="chart-grid"></div>`;
  const grid = document.getElementById("chart-grid");
  const destroyers = [];

  charts.forEach((c) => {
    const card = document.createElement("div");
    card.className = c.wide ? "chart-card chart-card-wide" : "chart-card";
    const host = document.createElement("div");
    card.appendChild(host);
    grid.appendChild(card);

    if (c.type === "trajectory3d") {
      makeTrajectory3dChart(host, { title: c.title, trajectory: c.trajectory }).then((plot) => {
        if (plot?.destroy) destroyers.push(plot.destroy);
      });
      return;
    }

    requestAnimationFrame(() => {
      makeLineChart(host, { title: c.title, x, series: c.series, yLabel: c.yLabel });
    });
  });

  stage._analysisDestroy = () => {
    for (const fn of destroyers) fn();
    stage._analysisDestroy = null;
  };
}

function renderChartGrid(stage, heading, charts, x) {
  stage.innerHTML = `<h2 class="section-head">${heading}</h2><div class="chart-grid" id="chart-grid"></div>`;
  const grid = document.getElementById("chart-grid");
  const containers = charts.map(() => {
    const card = document.createElement("div");
    card.className = "chart-card";
    const host = document.createElement("div");
    card.appendChild(host);
    grid.appendChild(card);
    return host;
  });
  requestAnimationFrame(() => {
    charts.forEach((c, i) =>
      makeLineChart(containers[i], { title: c.title, x, series: c.series, yLabel: c.yLabel })
    );
  });
}
