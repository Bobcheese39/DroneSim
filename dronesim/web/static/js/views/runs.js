// Runs view: browse saved runs and load one into Replay + Analysis.

import { api } from "../api.js";
import { store, setStatus, setScenario } from "../state.js";
import { tableHtml } from "../charts.js";

export async function renderRuns(ctx) {
  const stage = document.getElementById("runs-stage");
  const inspector = document.getElementById("inspector");

  inspector.innerHTML = `<div class="insp-section"><h3 class="insp-title">Runs</h3>
    <div class="card"><div class="hint">Select a run to load its trajectory into Replay and its metrics into Analysis.</div>
    <button class="btn full" id="refresh-runs" style="margin-top:8px;">Refresh</button></div></div>`;
  inspector.querySelector("#refresh-runs").addEventListener("click", () => renderRuns(ctx));

  stage.innerHTML = '<h2 class="section-head">Runs</h2><div id="runs-table" class="chart-card"><div class="empty">Loading\u2026</div></div>';

  let rows = [];
  try {
    rows = await api.listRuns();
  } catch (e) {
    stage.querySelector("#runs-table").innerHTML = `<div class="empty">Failed to load runs: ${e.message}</div>`;
    return;
  }
  if (!rows.length) {
    stage.querySelector("#runs-table").innerHTML = '<div class="empty">No saved runs yet.</div>';
    return;
  }

  const html = tableHtml(rows, [
    { key: "run_id", label: "Run" },
    { key: "backend", label: "Backend" },
    { key: "status", label: "Status" },
    { key: "success", label: "Success" },
    { key: "miss_m", label: "Miss (m)" },
    { key: "created_utc", label: "Created" },
  ]);
  stage.querySelector("#runs-table").innerHTML = html;

  // make rows clickable
  const trs = stage.querySelectorAll(".data-table tbody tr");
  trs.forEach((tr, i) => {
    tr.classList.add("clickable");
    tr.addEventListener("click", () => loadRun(ctx, rows[i]));
  });
}

async function loadRun(ctx, row) {
  setStatus("Loading run\u2026", "busy");
  try {
    const result = await api.runResult(row.path);
    store.run = result;

    // Try to load + build the run's scenario map so replay has imagery.
    try {
      const scenario = await api.getScenario(result.run.scenario_id);
      setScenario(scenario);
      const info = await api.buildMap(scenario.map, false);
      store.map = info;
      await ctx.scene.setMap(info);
    } catch (_e) {
      /* map optional; replay still works with terrain ellipsoid */
    }

    setStatus(`Loaded run ${row.run_id}.`, "ok");
    ctx.setView("replay");
  } catch (e) {
    setStatus(`Could not load run: ${e.message}`, "err");
  }
}
