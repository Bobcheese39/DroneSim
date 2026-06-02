// Central app state + a tiny pub/sub so views re-render on change.

const listeners = new Set();

export const store = {
  scenario: null, // working ScenarioSpec dict
  map: null, // { key, center_lat, center_lon, bounds, imagery_url, vertical_exaggeration }
  view: "create", // create | replay | analysis | runs
  createInspectorTab: "scenario", // scenario | vehicle
  editMode: "trajectory", // view | trajectory | marker | edit
  selection: { kind: null, index: null }, // waypoint | marker
  pendingAltitude: 5.0,
  run: null, // { run: RunResult, analysis, center }
  mcBatch: null, // { trials: [...], analysis: {...}, replay: {...} | null }
  runToken: null,
  runSocket: null,
  backends: [],
  status: { text: "Ready", level: "ok" },
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function emit(reason) {
  for (const fn of listeners) {
    try {
      fn(reason);
    } catch (e) {
      console.error("listener error", e);
    }
  }
}

export function setStatus(text, level = "ok") {
  store.status = { text, level };
  emit("status");
}

export function setScenario(scenario) {
  store.scenario = scenario;
  emit("scenario");
}

export function setSelection(kind, index) {
  store.selection = { kind, index };
  emit("selection");
}

// Local ENU <-> geographic conversions matching the server (terrain.py).
export function localToLatLon(xM, yM, centerLat, centerLon) {
  const lat = yM / 110540.0 + centerLat;
  const lon = xM / (111320.0 * Math.cos((centerLat * Math.PI) / 180.0)) + centerLon;
  return [lat, lon];
}

export function latLonToLocal(lat, lon, centerLat, centerLon) {
  const x = (lon - centerLon) * 111320.0 * Math.cos((centerLat * Math.PI) / 180.0);
  const y = (lat - centerLat) * 110540.0;
  return [x, y];
}
