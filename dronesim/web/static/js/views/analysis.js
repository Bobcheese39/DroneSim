// Analysis view: uPlot time-series charts + summary/parameter tables.

import { store } from "../state.js";
import { makeLineChart, tableHtml } from "../charts.js";

export function renderAnalysis() {
  const stage = document.getElementById("analysis-stage");
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

  const charts = [
    { title: "Tracking error", yLabel: "m", series: [{ label: "error", values: a.tracking_error_m }] },
    {
      title: "Position error (actual \u2212 reference)",
      yLabel: "m",
      series: [
        { label: "ex", values: a.error_decomposition.ex },
        { label: "ey", values: a.error_decomposition.ey },
        { label: "ez", values: a.error_decomposition.ez },
      ],
    },
    {
      title: "Velocity",
      yLabel: "m/s",
      series: [
        { label: "vx", values: a.velocity.vx },
        { label: "vy", values: a.velocity.vy },
        { label: "vz", values: a.velocity.vz },
      ],
    },
    {
      title: "Acceleration",
      yLabel: "m/s\u00b2",
      series: [
        { label: "ax", values: a.acceleration.ax },
        { label: "ay", values: a.acceleration.ay },
        { label: "az", values: a.acceleration.az },
      ],
    },
    {
      title: "Attitude",
      yLabel: "rad",
      series: [
        { label: "roll", values: a.attitude.roll },
        { label: "pitch", values: a.attitude.pitch },
        { label: "yaw", values: a.attitude.yaw },
      ],
    },
    {
      title: "Angular rate",
      yLabel: "rad/s",
      series: [
        { label: "p", values: a.angular_rate.p },
        { label: "q", values: a.angular_rate.q },
        { label: "r", values: a.angular_rate.r },
      ],
    },
    {
      title: "Control effort",
      yLabel: "N / Nm",
      series: [
        { label: "ft", values: a.controls.ft },
        { label: "tx", values: a.controls.tx },
        { label: "ty", values: a.controls.ty },
        { label: "tz", values: a.controls.tz },
      ],
    },
  ];

  if (a.clearance_m) {
    charts.push({
      title: "Terrain clearance",
      yLabel: "m",
      series: [
        { label: "clearance", values: a.clearance_m },
        { label: "1 m warn", values: a.clearance_m.map(() => 1.0), color: "#ff453a" },
      ],
    });
  }

  stage.innerHTML = '<h2 class="section-head">Analysis</h2><div class="chart-grid" id="chart-grid"></div>';
  const grid = document.getElementById("chart-grid");
  const containers = charts.map(() => {
    const card = document.createElement("div");
    card.className = "chart-card";
    const host = document.createElement("div");
    card.appendChild(host);
    grid.appendChild(card);
    return host;
  });
  // create plots after layout so clientWidth is correct
  requestAnimationFrame(() => {
    charts.forEach((c, i) =>
      makeLineChart(containers[i], { title: c.title, x, series: c.series, yLabel: c.yLabel })
    );
  });
}
