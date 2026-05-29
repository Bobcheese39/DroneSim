// Replay view: play/pause/scrub a run's trajectory over the Cesium scene.
// Ports the timing logic from the old Panel ReplayController.

import { store, localToLatLon } from "../state.js";

export function mountReplay(ctx) {
  const inspector = document.getElementById("inspector");
  const bar = document.getElementById("replay-bar");

  const state = {
    timeIndex: 0,
    playing: false,
    speed: 1.0,
    showReference: true,
    showClearance: true,
    raf: null,
    lastTs: 0,
    accumulator: 0,
  };

  function runData() {
    return store.run && store.run.run ? store.run.run : null;
  }
  function geoCenter() {
    const c = store.run && store.run.center;
    if (c && c.lat != null) return c;
    return { lat: store.scenario.map.center_lat, lon: store.scenario.map.center_lon };
  }

  function toGeoPath(rows) {
    if (!rows || !rows.length) return null;
    const c = geoCenter();
    return rows.map((p) => {
      const [lat, lon] = localToLatLon(p[0], p[1], c.lat, c.lon);
      return [lat, lon, p.length > 2 ? p[2] : 0];
    });
  }

  function clearanceArray() {
    const a = store.run && store.run.analysis;
    return a && a.clearance_m ? a.clearance_m : null;
  }

  function frameCount() {
    const run = runData();
    return run && run.time_s ? run.time_s.length : 0;
  }

  function renderInspector() {
    const run = runData();
    if (!run) {
      inspector.innerHTML = `
        <div class="insp-section"><h3 class="insp-title">Replay</h3>
        <div class="card"><div class="hint">No run loaded yet. Run a scenario from <strong>Create</strong>, or load one from <strong>Runs</strong>.</div></div></div>`;
      bar.hidden = true;
      return;
    }
    const s = run.summary || {};
    inspector.innerHTML = `
      <div class="insp-section">
        <h3 class="insp-title">Run</h3>
        <div class="card">
          <div class="field"><label>Run ID</label><input type="text" value="${run.run_id}" readonly /></div>
          <div class="field-row">
            <div class="field"><label>Status</label><input type="text" value="${run.status}" readonly /></div>
            <div class="field"><label>Miss (m)</label><input type="text" value="${fmt(s.miss_distance_m)}" readonly /></div>
          </div>
          <div class="field"><label>Steps</label><input type="text" value="${frameCount()}" readonly /></div>
        </div>
      </div>
      <div class="insp-section">
        <h3 class="insp-title">Playback</h3>
        <div class="card">
          <div class="field"><label>Speed</label>
            <select id="rp-speed">
              ${[0.5, 1, 2, 4].map((v) => `<option value="${v}" ${v === state.speed ? "selected" : ""}>${v}x</option>`).join("")}
            </select>
          </div>
          <div class="field toggle"><span>Show reference</span>
            <label class="switch"><input type="checkbox" id="rp-ref" ${state.showReference ? "checked" : ""}/><span class="slider"></span></label></div>
          <div class="field toggle"><span>Highlight low clearance</span>
            <label class="switch"><input type="checkbox" id="rp-clear" ${state.showClearance ? "checked" : ""}/><span class="slider"></span></label></div>
        </div>
      </div>`;
    inspector.querySelector("#rp-speed").addEventListener("change", (e) => (state.speed = parseFloat(e.target.value)));
    inspector.querySelector("#rp-ref").addEventListener("change", (e) => {
      state.showReference = e.target.checked;
      renderFrame();
    });
    inspector.querySelector("#rp-clear").addEventListener("change", (e) => {
      state.showClearance = e.target.checked;
      renderFrame();
    });
    renderBar();
  }

  function renderBar() {
    const n = frameCount();
    bar.hidden = false;
    bar.innerHTML = `
      <button class="icon-btn" id="rp-play">${state.playing ? "\u275a\u275a" : "\u25b6"}</button>
      <input type="range" id="rp-slider" min="0" max="${Math.max(0, n - 1)}" value="${state.timeIndex}" />
      <span class="time-label" id="rp-time">${timeLabel()}</span>`;
    bar.querySelector("#rp-play").addEventListener("click", togglePlay);
    bar.querySelector("#rp-slider").addEventListener("input", (e) => {
      state.timeIndex = parseInt(e.target.value, 10);
      renderFrame();
    });
  }

  function timeLabel() {
    const run = runData();
    if (!run || !run.time_s || !run.time_s.length) return "0.0 s";
    const t = run.time_s[Math.min(state.timeIndex, run.time_s.length - 1)] || 0;
    return `${t.toFixed(1)} / ${run.time_s[run.time_s.length - 1].toFixed(1)} s`;
  }

  function renderFrame() {
    const run = runData();
    if (!run) {
      ctx.scene.clearReplay();
      return;
    }
    const traj = toGeoPath(run.position_m);
    const ref = state.showReference ? toGeoPath(run.reference_position_m) : null;
    const clearance = clearanceArray();
    let low = false;
    if (state.showClearance && clearance) {
      low = clearance.slice(0, state.timeIndex + 1).some((c) => c != null && c < 1.0);
    }
    const drone = traj && traj.length ? traj[Math.min(state.timeIndex, traj.length - 1)] : null;
    ctx.scene.renderReplay({ trajectory: traj, reference: ref, timeIndex: state.timeIndex, drone, lowClearance: low });

    const t = bar.querySelector("#rp-time");
    if (t) t.textContent = timeLabel();
    const slider = bar.querySelector("#rp-slider");
    if (slider) slider.value = state.timeIndex;
  }

  function togglePlay() {
    const n = frameCount();
    if (n < 2) return;
    if (state.playing) {
      stop();
    } else {
      if (state.timeIndex >= n - 1) state.timeIndex = 0;
      state.playing = true;
      state.lastTs = performance.now();
      state.accumulator = 0;
      state.raf = requestAnimationFrame(tick);
      const btn = bar.querySelector("#rp-play");
      if (btn) btn.textContent = "\u275a\u275a";
    }
  }

  function stop() {
    state.playing = false;
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = null;
    const btn = bar.querySelector("#rp-play");
    if (btn) btn.textContent = "\u25b6";
  }

  function tick(ts) {
    if (!state.playing) return;
    const run = runData();
    const n = frameCount();
    const dt = (ts - state.lastTs) / 1000;
    state.lastTs = ts;
    const stepDt = run && run.time_s && run.time_s.length > 1 ? run.time_s[1] - run.time_s[0] : 0.1;
    state.accumulator += (dt * state.speed) / Math.max(stepDt, 1e-3);
    const advance = Math.floor(state.accumulator);
    if (advance >= 1) {
      state.accumulator -= advance;
      state.timeIndex = Math.min(state.timeIndex + advance, n - 1);
      renderFrame();
    }
    if (state.timeIndex >= n - 1) {
      stop();
      return;
    }
    state.raf = requestAnimationFrame(tick);
  }

  async function show() {
    stop();
    if (store.map) await ctx.scene.setMap(store.map);
    state.timeIndex = 0;
    renderInspector();
    renderFrame();
  }

  return { show };
}

function fmt(v) {
  return v == null ? "\u2014" : Number(v).toFixed(3);
}
