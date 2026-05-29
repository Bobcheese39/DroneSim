// App bootstrap: builds the Cesium scene, wires view routing, and delegates
// to the per-view modules.

import { CesiumScene } from "./map/cesium.js";
import { store, setStatus, setScenario, subscribe } from "./state.js";
import { api } from "./api.js";
import { mountCreate } from "./views/create.js";
import { mountReplay } from "./views/replay.js";
import { renderAnalysis } from "./views/analysis.js";
import { renderRuns } from "./views/runs.js";

const ctx = {
  scene: null,
  api,
  // refresh the map entities from the current scenario (create view owns this)
  refreshEntities: () => ctx.views.create && ctx.views.create.refreshEntities(),
  buildMap: buildCurrentMap,
  setView,
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
  document.getElementById("analysis-stage").classList.toggle("active", name === "analysis");
  document.getElementById("runs-stage").classList.toggle("active", name === "runs");
  document.getElementById("replay-bar").hidden = name !== "replay";
  document.getElementById("map-hud").hidden = !mapView;

  if (name === "create") ctx.views.create.show();
  else if (name === "replay") ctx.views.replay.show();
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

async function bootstrap() {
  ctx.scene = new CesiumScene("cesium", {
    onClick: (evt) => {
      if (store.view === "create" && ctx.views.create) ctx.views.create.handleClick(evt);
    },
  });

  // View switcher
  document.querySelectorAll("#view-switcher button").forEach((b) => {
    b.addEventListener("click", () => setView(b.dataset.view));
  });

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

  setView("create");
  await buildCurrentMap(false);
  setStatus("Ready.", "ok");
}

window.addEventListener("resize", () => ctx.scene && ctx.scene.resize());
bootstrap();
