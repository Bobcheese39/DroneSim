// Scenarios view: browse saved scenarios, open/duplicate/validate, create new.

import { api } from "../api.js";
import { store, setStatus, setScenario } from "../state.js";
import { tableHtml } from "../charts.js";

let selectedRow = null;

export async function renderScenarios(ctx) {
  const stage = document.getElementById("scenarios-stage");
  const inspector = document.getElementById("inspector");

  inspector.innerHTML = `
    <div class="insp-section">
      <h3 class="insp-title">Scenarios</h3>
      <div class="card">
        <div class="hint">Select a scenario to open, duplicate, or validate.</div>
        <div class="btn-row" style="margin-top:8px;">
          <button class="btn primary" id="sc-new">New</button>
          <button class="btn" id="sc-open" disabled>Open</button>
          <button class="btn" id="sc-duplicate" disabled>Duplicate</button>
        </div>
        <button class="btn full" id="sc-validate" style="margin-top:6px;" disabled>Validate</button>
        <button class="btn full" id="sc-refresh" style="margin-top:6px;">Refresh</button>
        <div id="sc-validation" style="margin-top:8px;"></div>
      </div>
    </div>`;

  stage.innerHTML =
    '<h2 class="section-head">Scenarios</h2><div id="scenarios-table" class="chart-card"><div class="empty">Loading\u2026</div></div>';

  inspector.querySelector("#sc-refresh").addEventListener("click", () => renderScenarios(ctx));
  inspector.querySelector("#sc-new").addEventListener("click", () => newScenario(ctx));
  inspector.querySelector("#sc-open").addEventListener("click", () => openSelected(ctx));
  inspector.querySelector("#sc-duplicate").addEventListener("click", () => duplicateSelected(ctx));
  inspector.querySelector("#sc-validate").addEventListener("click", () => validateSelected());

  let rows = [];
  try {
    rows = await api.listScenarios();
  } catch (e) {
    stage.querySelector("#scenarios-table").innerHTML = `<div class="empty">Failed to load scenarios: ${e.message}</div>`;
    return;
  }

  if (!rows.length) {
    stage.querySelector("#scenarios-table").innerHTML = '<div class="empty">No saved scenarios yet.</div>';
    return;
  }

  const html = tableHtml(rows, [
    { key: "name", label: "Name" },
    { key: "scenario_id", label: "ID" },
    { key: "n_waypoints", label: "WPs" },
    { key: "n_markers", label: "Markers" },
    { key: "updated_utc", label: "Updated" },
  ]);
  stage.querySelector("#scenarios-table").innerHTML = html;

  const trs = stage.querySelectorAll(".data-table tbody tr");
  trs.forEach((tr, i) => {
    tr.classList.add("clickable");
    tr.addEventListener("click", () => selectRow(tr, rows[i]));
  });
}

function selectRow(tr, row) {
  selectedRow = row;
  stageTrs().forEach((t) => t.classList.remove("selected"));
  tr.classList.add("selected");
  const openBtn = document.querySelector("#sc-open");
  const dupBtn = document.querySelector("#sc-duplicate");
  const valBtn = document.querySelector("#sc-validate");
  if (openBtn) openBtn.disabled = false;
  if (dupBtn) dupBtn.disabled = false;
  if (valBtn) valBtn.disabled = false;
}

function stageTrs() {
  return document.querySelectorAll("#scenarios-table .data-table tbody tr");
}

async function newScenario(ctx) {
  setStatus("Creating new scenario\u2026", "busy");
  try {
    const scenario = await api.defaultScenario();
    setScenario(scenario);
    store.mcBatch = null;
    store.run = null;
    await ctx.buildMap(false);
    ctx.refreshEntities();
    setStatus("New scenario loaded (unsaved).", "ok");
    ctx.setView("create");
  } catch (e) {
    setStatus(`Failed to create scenario: ${e.message}`, "err");
  }
}

async function openSelected(ctx) {
  if (!selectedRow) return;
  setStatus("Opening scenario\u2026", "busy");
  try {
    const scenario = await api.getScenario(selectedRow.scenario_id);
    setScenario(scenario);
    store.mcBatch = null;
    store.run = null;
    await ctx.buildMap(false);
    ctx.refreshEntities();
    setStatus(`Opened ${scenario.name}.`, "ok");
    ctx.setView("create");
  } catch (e) {
    setStatus(`Could not open scenario: ${e.message}`, "err");
  }
}

async function duplicateSelected(ctx) {
  if (!selectedRow) return;
  setStatus("Duplicating\u2026", "busy");
  try {
    const name = `${selectedRow.name} (copy)`;
    const resp = await api.duplicateScenario(selectedRow.scenario_id, { name });
    setValidation([], true);
    setStatus(`Duplicated as ${resp.scenario.name}.`, "ok");
    await renderScenarios(ctx);
  } catch (e) {
    setStatus(`Duplicate failed: ${e.message}`, "err");
  }
}

async function validateSelected() {
  if (!selectedRow) return;
  setStatus("Validating\u2026", "busy");
  try {
    const scenario = await api.getScenario(selectedRow.scenario_id);
    const resp = await api.validateScenario(scenario);
    setValidation(resp.messages || [], resp.ok);
    setStatus(resp.ok ? "Validation OK." : "Validation failed.", resp.ok ? "ok" : "warn");
  } catch (e) {
    setValidation([e.message], false);
    setStatus(`Validate failed: ${e.message}`, "err");
  }
}

function setValidation(messages, ok) {
  const el = document.querySelector("#sc-validation");
  if (!el) return;
  if (ok || !messages || !messages.length) {
    el.innerHTML = '<div class="validation ok">Validation OK</div>';
  } else {
    el.innerHTML = `<div class="validation warn">${messages.map(escapeHtml).join("<br/>")}</div>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
