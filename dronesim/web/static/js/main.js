// App bootstrap: builds the pluggable map engine (custom XYZ or Cesium LLA),
// wires view routing + the engine toggle, and delegates to the per-view modules.

import { MapEngine } from "./map/engine.js";
import { store, setStatus, setScenario, subscribe } from "./state.js";
import { api } from "./api.js";
import { mountCreate } from "./views/create.js";
import { mountReplay } from "./views/replay.js";
import { renderAnalysis } from "./views/analysis.js";
import { renderRuns } from "./views/runs.js";
import { renderScenarios } from "./views/scenarios.js";

const MAP_HOST = "map-host";

const ctx = {
  scene: null,
  api,
  // refresh the map entities from the current scenario (create view owns this)
  refreshEntities: () => ctx.views.create && ctx.views.create.refreshEntities(),
  refreshMapDisplay: () => refreshMapDisplay(),
  buildMap: buildCurrentMap,
  setView,
  switchMapMode: null,
  syncMapModeButtons: null,
  views: {},
};

function setStatusUI() {
  const pill = document.getElementById("status-pill");
  pill.dataset.level = store.status.level;
  document.getElementById("status-text").textContent = store.status.text;
}
subscribe((reason) => {
  if (reason === "status") setStatusUI();
});

function setView(name) {
  store.view = name;
  document.querySelectorAll("#view-switcher button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  const mapView = name === "create" || name === "replay";
  document.getElementById("map-stage").classList.toggle("active", mapView);
  document.getElementById("scenarios-stage").classList.toggle("active", name === "scenarios");
  document.getElementById("analysis-stage").classList.toggle("active", name === "analysis");
  document.getElementById("runs-stage").classList.toggle("active", name === "runs");
  document.getElementById("replay-bar").hidden = name !== "replay";
  document.getElementById("map-hud").hidden = !mapView;
  document.getElementById("map-debug").hidden = !mapView;

  if (name === "create") ctx.views.create.show();
  else if (name === "replay") ctx.views.replay.show();
  else if (name === "scenarios") renderScenarios(ctx);
  else if (name === "analysis") renderAnalysis(ctx);
  else if (name === "runs") renderRuns(ctx);

  if (mapView) setTimeout(() => ctx.scene.resize(), 60);
}

async function buildCurrentMap(fetchRemote = false) {
  if (!store.scenario) return;
  setStatus("Building map\u2026", "busy");
  try {
    const info = await api.buildMap(store.scenario.map, fetchRemote);
    store.map = info;
    await ctx.scene.setMap(info);
    ctx.refreshEntities();
    setStatus(`Map ready (${info.origin}).`, "ok");
  } catch (e) {
    if (e.status === 409) {
      setStatus("No cached map for this area. Enable Fetch remote to download.", "warn");
    } else {
      setStatus(`Map error: ${e.message}`, "err");
    }
  }
}

// Apply whatever map/ground is appropriate for the active engine mode.
async function applyMapForMode(mode) {
  if (mode === "lla") {
    if (store.map) await ctx.scene.setMap(store.map);
    else await buildCurrentMap(false);
  } else {
    ctx.scene.setGround(store.xyz.scale_m);
    if (store.xyz.mapInfo) await ctx.scene.setMap(store.xyz.mapInfo);
  }
}

function refreshMapDisplay() {
  if (store.view === "create") ctx.refreshEntities();
  else if (store.view === "replay" && ctx.views.replay) {
    if (ctx.views.replay.renderScenarioWaypoints) ctx.views.replay.renderScenarioWaypoints();
    if (ctx.views.replay.renderFrame) ctx.views.replay.renderFrame();
  }
}

function syncMapModeButtons() {
  document.querySelectorAll("#map-mode-switch button").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === store.mapMode);
  });
}

function mountMapDebug() {
  const root = document.getElementById("map-debug");
  const toggle = document.getElementById("map-debug-toggle");
  const panel = document.getElementById("map-debug-panel");
  const slider = document.getElementById("debug-wp-size");
  const output = document.getElementById("debug-wp-size-val");
  if (!root || !toggle || !panel || !slider || !output) return;

  const baseSize = store.debug.waypointMarkerSize ?? 12;
  slider.value = String(baseSize);
  output.textContent = String(baseSize);

  function setPanelOpen(open) {
    store.debug.menuOpen = open;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  }

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    setPanelOpen(panel.hidden);
  });

  document.addEventListener("click", (e) => {
    if (!store.debug.menuOpen) return;
    if (root.contains(e.target)) return;
    setPanelOpen(false);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && store.debug.menuOpen) setPanelOpen(false);
  });

  slider.addEventListener("input", () => {
    const size = parseInt(slider.value, 10);
    store.debug.waypointMarkerSize = size;
    output.textContent = String(size);
    refreshMapDisplay();
  });
}

async function switchMapMode(mode) {
  if (mode === store.mapMode || !ctx.scene) return;
  store.mapMode = mode;
  syncMapModeButtons();
  setStatus(mode === "lla" ? "Initializing Cesium\u2026" : "Switching to XYZ engine\u2026", "busy");
  try {
    await ctx.scene.setMode(mode);
    await applyMapForMode(mode);
    if (store.view === "create") {
      ctx.views.create.show();
      ctx.refreshEntities();
    } else if (store.view === "replay") {
      ctx.views.replay.show();
    }
    setStatus("Ready.", "ok");
  } catch (e) {
    setStatus(`Engine switch failed: ${e.message}`, "err");
  }
}

async function bootstrap() {
  ctx.scene = new MapEngine(MAP_HOST, {
    onClick: (evt) => {
      if (store.view === "create" && ctx.views.create) ctx.views.create.handleClick(evt);
    },
  });

  document.querySelectorAll("#view-switcher button").forEach((b) => {
    b.addEventListener("click", () => setView(b.dataset.view));
  });
  ctx.switchMapMode = switchMapMode;
  ctx.syncMapModeButtons = syncMapModeButtons;
  mountMapDebug();

  try {
    store.backends = await api.backends();
  } catch (_e) {
    store.backends = [{ backend_id: "inhouse_mpc_quad", display_name: "In-house MPC Quadcopter" }];
  }

  let scenario;
  try {
    scenario = await api.defaultScenario();
  } catch (e) {
    setStatus(`Failed to load default scenario: ${e.message}`, "err");
    return;
  }
  setScenario(scenario);

  ctx.views.create = mountCreate(ctx);
  ctx.views.replay = mountReplay(ctx);

  // Build the default engine (XYZ) before the first render so entities land.
  await ctx.scene.setMode(store.mapMode);
  syncMapModeButtons();

  setView("create");
  await applyMapForMode(store.mapMode);
  ctx.refreshEntities();
  setStatus("Ready.", "ok");
}

window.addEventListener("resize", () => ctx.scene && ctx.scene.resize());
bootstrap();
