// Thin fetch + WebSocket wrappers around the DroneSim REST API.

async function jsonOrThrow(resp) {
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (_e) {
    data = { detail: text };
  }
  if (!resp.ok) {
    const err = new Error((data && (data.message || data.detail)) || resp.statusText);
    err.status = resp.status;
    err.data = data;
    throw err;
  }
  return data;
}

async function get(path) {
  return jsonOrThrow(await fetch(path));
}

async function post(path, body) {
  return jsonOrThrow(
    await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    })
  );
}

export const api = {
  backends: () => get("/api/backends"),
  listScenarios: () => get("/api/scenarios"),
  defaultScenario: () => get("/api/scenarios/default"),
  getScenario: (id) => get(`/api/scenarios/${id}`),
  validateScenario: (scenario) => post("/api/scenarios/validate", scenario),
  saveScenario: (scenario) => post("/api/scenarios", scenario),
  duplicateScenario: (id, body) => post(`/api/scenarios/${id}/duplicate`, body),

  buildMap: (mapSpec, fetchRemote) =>
    post("/api/map/build", { map: mapSpec, fetch_remote: !!fetchRemote }),
  mapCaches: () => get("/api/map/caches"),
  uploadMap: async (formData) =>
    jsonOrThrow(await fetch("/api/map/upload", { method: "POST", body: formData })),

  addWaypoint: (body) => post("/api/edit/add-waypoint", body),
  addMarker: (body) => post("/api/edit/add-marker", body),
  moveEntity: (body) => post("/api/edit/move", body),
  deleteEntity: (body) => post("/api/edit/delete", body),
  reorderEntity: (body) => post("/api/edit/reorder", body),

  listRuns: (scenarioId) =>
    get(scenarioId ? `/api/runs?scenario_id=${scenarioId}` : "/api/runs"),
  runResult: (path) => get(`/api/runs/result?path=${encodeURIComponent(path)}`),
  startRun: (body) => post("/api/runs/start", body),
  cancelRun: (runToken) => post("/api/runs/cancel", { run_token: runToken }),
  mcAnalysis: (paths) => post("/api/runs/mc-analysis", { paths }),
  mcReplay: (paths) => post("/api/runs/mc-replay", { paths }),
};

// Open a run-progress WebSocket. Returns the socket; caller wires onmessage.
export function openRunSocket(token, onEvent, onClose) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/run/${token}`);
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data));
    } catch (_e) {
      /* ignore malformed frame */
    }
  };
  if (onClose) ws.onclose = onClose;
  return ws;
}
