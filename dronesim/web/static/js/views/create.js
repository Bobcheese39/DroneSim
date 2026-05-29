// Create view: scenario inspector form, map building, click-to-edit on the
// Cesium scene, selection editing, and save/run.

import { api, openRunSocket } from "../api.js";
import { store, setStatus, setScenario, setSelection, localToLatLon } from "../state.js";

const CSS_COLORS = {
  red: "#ff453a",
  yellow: "#ffd60a",
  green: "#30d158",
  blue: "#0a84ff",
  orange: "#ff9f0a",
};

function cssColor(c) {
  if (!c) return "#ff453a";
  return CSS_COLORS[c] || c;
}

function center() {
  const m = store.scenario.map;
  return { lat: m.center_lat, lon: m.center_lon };
}

function wpToGeo(wp) {
  const c = center();
  if (wp.lat != null && wp.lon != null) {
    return { lat: wp.lat, lon: wp.lon, alt: wp.alt_m != null ? wp.alt_m : wp.z_m || 0 };
  }
  const [lat, lon] = localToLatLon(wp.x_m, wp.y_m, c.lat, c.lon);
  return { lat, lon, alt: wp.z_m != null ? wp.z_m : wp.alt_m || 0 };
}

function markerToGeo(m) {
  const c = center();
  if (m.lat != null && m.lon != null) {
    return { lat: m.lat, lon: m.lon, alt: m.alt_m != null ? m.alt_m : m.z_m || 0 };
  }
  const [lat, lon] = localToLatLon(m.x_m, m.y_m, c.lat, c.lon);
  return { lat, lon, alt: m.z_m != null ? m.z_m : m.alt_m || 0 };
}

export function mountCreate(ctx) {
  const inspector = document.getElementById("inspector");
  const hud = document.getElementById("map-hud");

  function backendOptions(selected) {
    return store.backends
      .map(
        (b) =>
          `<option value="${b.backend_id}" ${b.backend_id === selected ? "selected" : ""}>${b.display_name}</option>`
      )
      .join("");
  }

  function render() {
    const s = store.scenario;
    const rc = s.run_config;
    const sel = store.selection;
    inspector.innerHTML = `
      <div class="insp-section">
        <h3 class="insp-title">Scenario</h3>
        <div class="card">
          <div class="field"><label>Name</label><input type="text" data-k="name" value="${escapeAttr(s.name)}" /></div>
          <div class="field"><label>Description</label><textarea data-k="description" style="font-family:var(--font);">${escapeHtml(s.description || "")}</textarea></div>
        </div>
      </div>

      <div class="insp-section">
        <h3 class="insp-title">Map</h3>
        <div class="card">
          <div class="field-row">
            <div class="field"><label>Center lat</label><input type="number" step="0.0001" data-mk="center_lat" value="${s.map.center_lat}" /></div>
            <div class="field"><label>Center lon</label><input type="number" step="0.0001" data-mk="center_lon" value="${s.map.center_lon}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>Radius (km)</label><input type="number" step="0.1" data-mk="radius_km" value="${s.map.radius_km}" /></div>
            <div class="field"><label>Resolution</label><input type="number" step="50" data-mk="resolution" value="${s.map.resolution}" /></div>
          </div>
          <div class="field"><label>Vertical exaggeration</label><input type="number" step="0.1" data-mk="vertical_exaggeration" value="${s.map.vertical_exaggeration}" /></div>
          <div class="field toggle"><span>Fetch remote imagery</span>
            <label class="switch"><input type="checkbox" id="fetch-remote" /><span class="slider"></span></label>
          </div>
          <button class="btn primary full" id="build-map">Build map</button>
        </div>
      </div>

      <div class="insp-section">
        <h3 class="insp-title">Edit mode</h3>
        <div class="segmented compact" id="edit-mode">
          ${["view", "trajectory", "marker", "edit"]
            .map(
              (m) => `<button data-mode="${m}" class="${store.editMode === m ? "active" : ""}">${cap(m)}</button>`
            )
            .join("")}
        </div>
        <div class="field" style="margin-top:10px;"><label>Placement altitude (m, AGL)</label>
          <input type="number" step="0.5" id="pending-alt" value="${store.pendingAltitude}" /></div>
      </div>

      <div class="insp-section">
        <h3 class="insp-title">Waypoints (${s.waypoints.waypoints.length})</h3>
        <div class="list" id="wp-list">
          ${s.waypoints.waypoints
            .map((wp, i) => {
              const g = wpToGeo(wp);
              return `<div class="list-item ${sel.kind === "waypoint" && sel.index === i ? "selected" : ""}" data-kind="waypoint" data-i="${i}">
                <span class="swatch" style="background:#0a84ff"></span>${wp.label || "WP" + i}
                <span class="meta">${(wp.z_m ?? wp.alt_m ?? 0).toFixed(1)} m</span></div>`;
            })
            .join("")}
        </div>
      </div>

      <div class="insp-section">
        <h3 class="insp-title">Markers (${s.markers.markers.length})</h3>
        <div class="list" id="marker-list">
          ${s.markers.markers
            .map(
              (m, i) =>
                `<div class="list-item ${sel.kind === "marker" && sel.index === i ? "selected" : ""}" data-kind="marker" data-i="${i}">
                <span class="swatch" style="background:${cssColor(m.color)}"></span>${m.label}
                <span class="meta">${(m.z_m ?? m.alt_m ?? 0).toFixed(1)} m</span></div>`
            )
            .join("") || '<div class="list-item">No markers</div>'}
        </div>
        ${selectionPanel(sel)}
      </div>

      <div class="insp-section">
        <h3 class="insp-title">Vehicle &amp; Run</h3>
        <div class="card">
          <div class="field"><label>Backend</label><select data-rk="backend_id">${backendOptions(rc.backend_id)}</select></div>
          <div class="field-row">
            <div class="field"><label>Target alt (m)</label><input type="number" step="0.5" data-rk="target_altitude_m" value="${rc.target_altitude_m}" /></div>
            <div class="field"><label>dt (s)</label><input type="number" step="0.01" data-rk="dt_s" value="${rc.dt_s}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>Max steps</label><input type="number" step="10" data-rk="max_steps" value="${rc.max_steps}" /></div>
            <div class="field"><label>Horizon</label><input type="number" step="1" data-rk="horizon" value="${rc.horizon}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>Lookahead</label><input type="number" step="5" data-rk="lookahead" value="${rc.lookahead}" /></div>
            <div class="field"><label>Seed</label><input type="number" step="1" data-rk="seed" value="${rc.seed ?? 0}" /></div>
          </div>
        </div>
      </div>

      <div class="insp-section">
        <div class="btn-row">
          <button class="btn" id="save-scenario">Save</button>
          <button class="btn primary" id="run-scenario">Run</button>
        </div>
        <div id="validation"></div>
      </div>
    `;
    wire();
    updateHud();
  }

  function selectionPanel(sel) {
    if (sel.kind == null || sel.index == null) return "";
    const list = sel.kind === "waypoint" ? store.scenario.waypoints.waypoints : store.scenario.markers.markers;
    const item = list[sel.index];
    if (!item) return "";
    const x = item.x_m ?? 0;
    const y = item.y_m ?? 0;
    const z = item.z_m ?? item.alt_m ?? 0;
    return `
      <div class="card" style="margin-top:10px;">
        <div class="hint" style="margin-bottom:8px;">Selected ${sel.kind} #${sel.index}: ${item.label || ""}</div>
        <div class="field-row">
          <div class="field"><label>X (m)</label><input type="number" step="0.5" id="nudge-x" value="${num(x)}" /></div>
          <div class="field"><label>Y (m)</label><input type="number" step="0.5" id="nudge-y" value="${num(y)}" /></div>
          <div class="field"><label>Z (m)</label><input type="number" step="0.5" id="nudge-z" value="${num(z)}" /></div>
        </div>
        <button class="btn primary full" id="apply-nudge">Apply XYZ</button>
        <div class="btn-row">
          <button class="btn" id="reorder-up">Move up</button>
          <button class="btn" id="reorder-down">Move down</button>
        </div>
        <div class="btn-row">
          <button class="btn danger full" id="delete-selected">Delete</button>
        </div>
      </div>`;
  }

  function wire() {
    inspector.querySelectorAll("[data-k]").forEach((el) =>
      el.addEventListener("change", () => {
        store.scenario[el.dataset.k] = el.value;
      })
    );
    inspector.querySelectorAll("[data-mk]").forEach((el) =>
      el.addEventListener("change", () => {
        const k = el.dataset.mk;
        store.scenario.map[k] = k === "resolution" ? parseInt(el.value, 10) : parseFloat(el.value);
      })
    );
    inspector.querySelectorAll("[data-rk]").forEach((el) =>
      el.addEventListener("change", () => {
        const k = el.dataset.rk;
        const intKeys = ["max_steps", "horizon", "lookahead", "seed"];
        store.scenario.run_config[k] =
          el.tagName === "SELECT" ? el.value : intKeys.includes(k) ? parseInt(el.value, 10) : parseFloat(el.value);
      })
    );

    inspector.querySelector("#fetch-remote") &&
      (inspector.querySelector("#fetch-remote").checked = false);
    inspector.querySelector("#build-map").addEventListener("click", () => {
      const remote = inspector.querySelector("#fetch-remote").checked;
      ctx.buildMap(remote);
    });

    inspector.querySelectorAll("#edit-mode button").forEach((b) =>
      b.addEventListener("click", () => {
        store.editMode = b.dataset.mode;
        if (store.editMode !== "edit") setSelection(null, null);
        render();
        refreshEntities();
      })
    );

    inspector.querySelector("#pending-alt").addEventListener("change", (e) => {
      store.pendingAltitude = parseFloat(e.target.value) || 0;
    });

    inspector.querySelectorAll(".list-item[data-kind]").forEach((el) => {
      el.addEventListener("click", () => {
        setSelection(el.dataset.kind, parseInt(el.dataset.i, 10));
        render();
        refreshEntities();
      });
      el.addEventListener("dblclick", (e) => {
        e.preventDefault();
        centerSidebarItem(el.dataset.kind, parseInt(el.dataset.i, 10));
      });
    });

    const applyBtn = inspector.querySelector("#apply-nudge");
    if (applyBtn) applyBtn.addEventListener("click", applyNudge);
    const delBtn = inspector.querySelector("#delete-selected");
    if (delBtn) delBtn.addEventListener("click", deleteSelected);
    const upBtn = inspector.querySelector("#reorder-up");
    if (upBtn) upBtn.addEventListener("click", () => reorder("up"));
    const downBtn = inspector.querySelector("#reorder-down");
    if (downBtn) downBtn.addEventListener("click", () => reorder("down"));

    inspector.querySelector("#save-scenario").addEventListener("click", saveScenario);
    inspector.querySelector("#run-scenario").addEventListener("click", runScenario);
  }

  function updateHud() {
    const tips = {
      view: "View mode \u2014 pan/zoom only.",
      trajectory: "Trajectory mode \u2014 click the map to add waypoints.",
      marker: "Marker mode \u2014 click the map to drop annotation markers.",
      edit: "Edit mode \u2014 click waypoints/markers to select and edit them.",
    };
    hud.textContent = tips[store.editMode];
  }

  function centerSidebarItem(kind, index) {
    const list =
      kind === "waypoint" ? store.scenario.waypoints.waypoints : store.scenario.markers.markers;
    const item = list[index];
    if (!item || !ctx.scene) return;
    const geo = kind === "waypoint" ? wpToGeo(item) : markerToGeo(item);
    ctx.scene.centerOn(geo);
  }

  function refreshEntities() {
    const s = store.scenario;
    const sel = store.selection;
    const wps = s.waypoints.waypoints.map((wp, i) => ({
      index: i,
      ...wpToGeo(wp),
      label: wp.label || "WP" + i,
      selected: sel.kind === "waypoint" && sel.index === i,
    }));
    const markers = s.markers.markers.map((m, i) => ({
      index: i,
      ...markerToGeo(m),
      label: m.label,
      color: cssColor(m.color),
      size: m.size,
      visible: m.visible,
      selected: sel.kind === "marker" && sel.index === i,
    }));
    ctx.scene.renderEntities({ waypoints: wps, markers });
    ctx.scene.clearReplay();
  }

  async function handleClick(evt) {
    const mode = store.editMode;
    if (mode === "view") return;

    if (evt.entity_kind === "waypoint" || evt.entity_kind === "marker") {
      if (evt.entity_index != null) {
        setSelection(evt.entity_kind, evt.entity_index);
        render();
        refreshEntities();
        return;
      }
    }
    if (mode === "edit") {
      setSelection(null, null);
      render();
      refreshEntities();
      return;
    }

    try {
      let resp;
      if (mode === "trajectory") {
        resp = await api.addWaypoint({
          scenario: store.scenario,
          lat: evt.lat,
          lon: evt.lon,
          alt_m: store.pendingAltitude,
        });
      } else if (mode === "marker") {
        resp = await api.addMarker({
          scenario: store.scenario,
          lat: evt.lat,
          lon: evt.lon,
          alt_m: store.pendingAltitude,
        });
      } else {
        return;
      }
      setScenario(resp.scenario);
      setSelection(mode === "trajectory" ? "waypoint" : "marker", resp.index);
      render();
      refreshEntities();
      setStatus(`Added ${mode === "trajectory" ? "waypoint" : "marker"} #${resp.index}.`, "ok");
    } catch (e) {
      setStatus(`Edit failed: ${e.message}`, "err");
    }
  }

  async function applyNudge() {
    const sel = store.selection;
    const resp = await api.moveEntity({
      scenario: store.scenario,
      kind: sel.kind,
      index: sel.index,
      x_m: parseFloat(inspector.querySelector("#nudge-x").value),
      y_m: parseFloat(inspector.querySelector("#nudge-y").value),
      z_m: parseFloat(inspector.querySelector("#nudge-z").value),
    });
    setScenario(resp.scenario);
    render();
    refreshEntities();
  }

  async function deleteSelected() {
    const sel = store.selection;
    const resp = await api.deleteEntity({ scenario: store.scenario, kind: sel.kind, index: sel.index });
    setScenario(resp.scenario);
    setSelection(null, null);
    render();
    refreshEntities();
  }

  async function reorder(direction) {
    const sel = store.selection;
    const resp = await api.reorderEntity({
      scenario: store.scenario,
      kind: sel.kind,
      index: sel.index,
      direction,
    });
    setScenario(resp.scenario);
    if (resp.ok) setSelection(sel.kind, resp.new_index);
    render();
    refreshEntities();
  }

  async function saveScenario() {
    setStatus("Saving\u2026", "busy");
    try {
      const resp = await api.saveScenario(store.scenario);
      setScenario(resp.scenario);
      setValidation([], true);
      setStatus("Scenario saved.", "ok");
    } catch (e) {
      const msgs = (e.data && e.data.detail && e.data.detail.messages) || [e.message];
      setValidation(msgs, false);
      setStatus("Save failed (see validation).", "warn");
    }
  }

  async function runScenario() {
    setStatus("Launching run\u2026", "busy");
    try {
      const { run_token } = await api.startRun({ scenario: store.scenario });
      store.runToken = run_token;
      streamRun(run_token);
    } catch (e) {
      const msgs = (e.data && e.data.detail && e.data.detail.messages) || [e.message];
      setValidation(msgs, false);
      setStatus("Run failed to start.", "err");
    }
  }

  function streamRun(token) {
    openRunSocket(token, async (ev) => {
      if (ev.type === "step_progress") {
        setStatus(`Simulating \u2014 step ${ev.step} / ${ev.total}`, "busy");
      } else if (ev.type === "result") {
        setStatus(
          `Run ${ev.success ? "succeeded" : ev.status} \u2014 miss ${fmt(ev.miss_distance_m)} m.`,
          ev.success ? "ok" : "warn"
        );
        if (ev.path) {
          try {
            store.run = await api.runResult(ev.path);
          } catch (_e) {
            /* ignore */
          }
        }
      } else if (ev.type === "error") {
        setStatus(`Run error: ${ev.message}`, "err");
      } else if (ev.type === "done") {
        if (store.run) {
          ctx.setView("replay");
        }
      }
    });
  }

  function setValidation(messages, ok) {
    const el = inspector.querySelector("#validation");
    if (!el) return;
    if (ok || !messages || !messages.length) {
      el.innerHTML = '<div class="validation ok">Validation OK</div>';
    } else {
      el.innerHTML = `<div class="validation warn">${messages.map(escapeHtml).join("<br/>")}</div>`;
    }
  }

  return {
    show: render,
    handleClick,
    refreshEntities,
  };
}

// ---- small format/escape helpers ----
function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function num(v) {
  return typeof v === "number" ? v.toFixed(2) : "0";
}
function fmt(v) {
  return v == null ? "\u2014" : Number(v).toFixed(3);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function escapeAttr(s) {
  return String(s).replace(/"/g, "&quot;");
}
