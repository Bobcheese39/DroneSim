// Create view: scenario inspector form, map building, click-to-edit on the
// Cesium scene, selection editing, and save/run.

import { api, openRunSocket } from "../api.js";
import { store, setStatus, setScenario, setSelection, waypointMarkerRatio } from "../state.js";

const CSS_COLORS = {
  red: "#ff453a",
  yellow: "#ffd60a",
  green: "#30d158",
  blue: "#0a84ff",
  orange: "#ff9f0a",
};

const JSBSIM_CESSNA_BACKEND = "jsbsim_cessna";
const POINTMASS_QUAD_BACKEND = "pointmass_quad";
const POINTMASS_FW_BACKEND = "pointmass_fixed_wing";

function cssColor(c) {
  if (!c) return "#ff453a";
  return CSS_COLORS[c] || c;
}

// Both engines consume local meters { x, y, z }.
function wpToLocal(wp) {
  return {
    x: wp.x_m ?? 0,
    y: wp.y_m ?? 0,
    z: wp.z_m != null ? wp.z_m : wp.alt_m || 0,
  };
}

function markerToLocal(m) {
  return {
    x: m.x_m ?? 0,
    y: m.y_m ?? 0,
    z: m.z_m != null ? m.z_m : m.alt_m || 0,
  };
}

export function mountCreate(ctx) {
  const inspector = document.getElementById("inspector");
  const hud = document.getElementById("map-hud");
  void loadJsbsimPresets();
  void loadJsbsimAircraft();
  void loadPointmassModels();

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
    if (!s) return;
    ensureRunConfigObj(s);
    ensureVehicleObj(s);
    ensureEnvironmentObj(s);
    ensureMonteCarloObj(s);
    const rc = s.run_config;
    const mc = rc.monte_carlo;
    const sel = store.selection;
    const tab = activeInspectorTab();
    inspector.innerHTML = `
      ${tabSelector(tab)}
      ${tab === "scenario" ? scenarioTabSection(s, sel) : vehicleTabSection(s, rc, mc)}
      ${actionSection(rc)}
    `;
    wire();
    updateHud();
  }

  function activeInspectorTab() {
    return store.createInspectorTab === "vehicle" ? "vehicle" : "scenario";
  }

  function tabSelector(tab) {
    return `
      <div class="insp-section">
        <h3 class="insp-title">Create</h3>
        <div class="segmented compact inspector-tabs" id="inspector-tabs">
          <button data-tab="scenario" class="${tab === "scenario" ? "active" : ""}">Scenario</button>
          <button data-tab="vehicle" class="${tab === "vehicle" ? "active" : ""}">Vehicle</button>
        </div>
      </div>
    `;
  }

  function scenarioTabSection(s, sel) {
    return `
      <div class="insp-section">
        <h3 class="insp-title">Scenario</h3>
        <div class="card">
          <div class="field"><label>Name</label><input type="text" data-k="name" value="${escapeAttr(s.name)}" /></div>
          <div class="field"><label>Description</label><textarea data-k="description" style="font-family:var(--font);">${escapeHtml(s.description || "")}</textarea></div>
          <div class="field">
            <label>Map engine</label>
            <nav class="segmented compact" id="map-mode-switch" title="3D engine">
              <button type="button" data-mode="xyz" class="${store.mapMode === "xyz" ? "active" : ""}">XYZ</button>
              <button type="button" data-mode="lla" class="${store.mapMode === "lla" ? "active" : ""}">LLA</button>
            </nav>
          </div>
        </div>
      </div>

      <div class="insp-section">
        <h3 class="insp-title">Map</h3>
        ${store.mapMode === "xyz" ? mapSectionXyz() : mapSectionLla(s)}
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
    `;
  }

  function vehicleTabSection(s, rc, mc) {
    const vehicle = s.vehicle;
    const params = vehicle.parameters;
    const aero = params.aero;
    const env = s.environment;
    const wind = Array.isArray(env.wind_mps) ? env.wind_mps : [0, 0, 0];
    const fixedWing = isFixedWingVehicle(s, rc);
    const pointMass = isPointMassVehicle(s, rc);
    return `
      <div class="insp-section">
        <h3 class="insp-title">Run profile</h3>
        <div class="card">
          <div class="hint" style="margin-bottom:8px;">
            ${
              fixedWing
                ? "JSBSim Cessna uses fixed-wing waypoint guidance and normalized flight-control commands."
                : pointMass
                ? "3DOF point-mass model: kinematic guidance — no rotational dynamics, oscillation-free for any positive gains."
                : "Advanced fidelity options currently apply to In-house MPC backend."
            }
          </div>
          <div class="field"><label>${labelWithTip("Backend", "Simulation backend used when you launch a run.")}</label><select data-rk="backend_id">${backendOptions(rc.backend_id)}</select></div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Target altitude (m)", "Default mission altitude used by the tracker and initial spawn height.")}</label><input type="number" step="0.5" data-rk="target_altitude_m" value="${safeNum(rc.target_altitude_m, 5)}" /></div>
            <div class="field"><label>${labelWithTip("dt (s)", "Control sample period. Smaller values increase fidelity and compute cost.")}</label><input type="number" step="0.01" data-rk="dt_s" value="${safeNum(rc.dt_s, 0.1)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Max steps", "Hard cap on simulated control steps before a run is marked complete.")}</label><input type="number" step="10" data-rk="max_steps" value="${safeNum(rc.max_steps, 250)}" /></div>
            <div class="field"><label>${labelWithTip(
              "Waypoint threshold (m)",
              fixedWing
                ? "Final approach tolerance (horizontal and vertical). Leg-to-leg capture uses Waypoint capture radius in the Fixed-Wing section."
                : "Distance used to mark waypoint completion and advance to the next segment."
            )}</label><input type="number" step="0.01" data-rk="waypoint_threshold_m" value="${safeNum(rc.waypoint_threshold_m, fixedWing ? 50 : 0.25)}" /></div>
          </div>
          ${
            fixedWing || pointMass
              ? ""
              : `<div class="field-row">
                  <div class="field"><label>${labelWithTip("Horizon", "MPC prediction horizon in control steps.")}</label><input type="number" step="1" data-rk="horizon" value="${safeNum(rc.horizon, 20)}" /></div>
                  <div class="field"><label>${labelWithTip("Lookahead", "Spline lookahead length used by the waypoint tracker.")}</label><input type="number" step="1" data-rk="lookahead" value="${safeNum(rc.lookahead, 60)}" /></div>
                </div>`
          }
          <div class="field"><label>${labelWithTip("Seed", "Optional base seed for deterministic sampling. Leave blank for backend default behavior.")}</label><input type="number" step="1" data-rk="seed" value="${rc.seed == null ? "" : rc.seed}" placeholder="auto" /></div>
        </div>
      </div>

      ${fixedWing ? fixedWingSection(vehicle) : pointMass ? pointMassSection(vehicle, rc) : quadFidelitySection(rc)}

      ${(fixedWing || pointMass) ? "" : quadVehicleSections(params, aero)}

      <div class="insp-section vehicle-detail">
        <details open>
          <summary>Environment</summary>
          <div class="card">
            <div class="field-row">
              <div class="field"><label>${labelWithTip("Wind X (m/s)", "Constant east-west wind component in local frame.")}</label><input type="number" step="0.1" data-wind-i="0" value="${safeNum(wind[0], 0)}" /></div>
              <div class="field"><label>${labelWithTip("Wind Y (m/s)", "Constant north-south wind component in local frame.")}</label><input type="number" step="0.1" data-wind-i="1" value="${safeNum(wind[1], 0)}" /></div>
              <div class="field"><label>${labelWithTip("Wind Z (m/s)", "Constant vertical wind component in local frame.")}</label><input type="number" step="0.1" data-wind-i="2" value="${safeNum(wind[2], 0)}" /></div>
            </div>
            <div class="field-row">
              <div class="field"><label>${labelWithTip("Gust σ (m/s)", "Random gust intensity for Ornstein-Uhlenbeck wind model.")}</label><input type="number" step="0.01" data-ek="gust_std_mps" value="${safeNum(env.gust_std_mps, 0)}" /></div>
              <div class="field"><label>${labelWithTip("Gust decorrelation (s)", "Time constant controlling how quickly random gusts change.")}</label><input type="number" step="0.1" data-ek="gust_decorrelation_s" value="${safeNum(env.gust_decorrelation_s, 2)}" /></div>
            </div>
            <div class="field"><label>${labelWithTip("Air density (kg/m³)", "Air density used to scale aerodynamic drag in extended dynamics.")}</label><input type="number" step="0.001" data-ek="air_density_kg_m3" value="${safeNum(env.air_density_kg_m3, 1.225)}" /></div>
            <div class="field toggle">
              <span class="toggle-label">${labelWithTip("Terrain collision check", "Ends a run when altitude drops below terrain + clearance. Requires cached terrain map.")}</span>
              <label class="switch"><input type="checkbox" data-ek="terrain_collision_enabled" ${env.terrain_collision_enabled ? "checked" : ""} /><span class="slider"></span></label>
            </div>
            <div class="field"><label>${labelWithTip("Terrain collision clearance (m)", "Minimum altitude above terrain before collision is reported.")}</label><input type="number" step="0.1" data-ek="terrain_collision_offset_m" value="${safeNum(env.terrain_collision_offset_m, 0.5)}" /></div>
          </div>
        </details>
      </div>

      <div class="insp-section mc-section vehicle-detail">
        <details ${mcOpen()}>
          <summary>Monte Carlo</summary>
          <div class="card">
            <div class="hint" style="margin-bottom:8px;">
              Monte Carlo mode activates when Trials is greater than 1.
            </div>
            ${mcField("n_trials", "Trials", mc.n_trials ?? 1, 1, "Number of randomized trials in the batch.")}
            <div class="field-row">
              ${mcField("workers", "Workers", mc.workers ?? 1, 1, "Parallel worker count used by the run manager.")}
              ${mcField("base_seed", "Base seed", mc.base_seed ?? 0, 1, "Per-trial seeds are base_seed + trial_index.")}
            </div>
            <div class="field-row">
              ${mcField("init_pos_std", "Init pos σ (m)", mc.init_pos_std ?? 0, 0.01, "Standard deviation for initial position perturbation.")}
              ${mcField("init_vel_std", "Init vel σ (m/s)", mc.init_vel_std ?? 0, 0.01, "Standard deviation for initial velocity perturbation.")}
            </div>
            <div class="field-row">
              ${mcField("init_att_std", "Init att σ (rad)", mc.init_att_std ?? 0, 0.001, "Standard deviation for initial attitude perturbation.")}
              ${mcField("force_noise_std", "Force noise σ", mc.force_noise_std ?? 0, 0.01, "Control-force perturbation applied each simulation step.")}
            </div>
            <div class="field-row">
              ${mcField("mass_jitter_pct", "Mass jitter (%)", mc.mass_jitter_pct ?? 0, 0.1, "Percent mass randomization applied per trial.")}
              ${mcField("inertia_jitter_pct", "Inertia jitter (%)", mc.inertia_jitter_pct ?? 0, 0.1, "Percent inertia randomization applied per trial.")}
            </div>
          </div>
        </details>
      </div>
    `;
  }

  function quadFidelitySection(rc) {
    return `
      <div class="insp-section">
        <h3 class="insp-title">Solver &amp; Fidelity</h3>
        <div class="card">
          <div class="field-row">
            <div class="field">
              <label>${labelWithTip("Integration method", "Euler reproduces legacy behavior; RK4 improves numerical accuracy on extended dynamics.")}</label>
              <select data-rk="integration_method">
                ${optionHtml("euler", rc.integration_method)}
                ${optionHtml("rk4", rc.integration_method)}
              </select>
            </div>
            <div class="field">
              <label>${labelWithTip("Fidelity mode", "Auto enables extended runtime when advanced knobs are non-default. Legacy forces vendor path.")}</label>
              <select data-rk="fidelity_mode">
                ${optionHtml("auto", rc.fidelity_mode)}
                ${optionHtml("legacy", rc.fidelity_mode)}
                ${optionHtml("extended", rc.fidelity_mode)}
              </select>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function pointMassSection(vehicle, rc) {
    const params = vehicle.parameters || {};
    const isQuad = vehicle.backend_id === POINTMASS_QUAD_BACKEND || vehicle.model_type === "pointmass_quad";
    const activeModel = (store.scenario?.metadata?.pointmass_model || "").trim();
    const modelOptions = (store.pointmassModels || [])
      .filter((m) => isQuad ? m.model_type === "pointmass_quad" : m.model_type === "pointmass_fixed_wing")
      .map(
        (m) =>
          `<option value="${escapeAttr(m.id)}" ${m.id === activeModel ? "selected" : ""}>${escapeHtml(m.label)}</option>`
      )
      .join("");
    const quadFields = isQuad ? `
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Mass (kg)", "Vehicle mass used to compute thrust output for display.")}</label><input type="number" step="0.1" min="0.1" data-vk="parameters.mass" value="${safeNum(params.mass, 5)}" /></div>
        <div class="field"><label>${labelWithTip("Max accel (m/s²)", "Hard cap on commanded acceleration magnitude.")}</label><input type="number" step="0.1" min="0.1" data-vk="parameters.max_accel_mps2" value="${safeNum(params.max_accel_mps2, 5)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Max speed (m/s)", "Speed cap applied after each integration step.")}</label><input type="number" step="0.5" min="0.1" data-vk="parameters.max_speed_mps" value="${safeNum(params.max_speed_mps, 10)}" /></div>
        <div class="field"><label>${labelWithTip("Capture radius (m)", "Horizontal distance to count a waypoint as reached.")}</label><input type="number" step="0.1" min="0" data-vk="parameters.waypoint_capture_radius_m" value="${safeNum(params.waypoint_capture_radius_m, 0.5)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("kp_pos", "Position proportional gain. Increase for faster response.")}</label><input type="number" step="0.05" min="0" data-vk="parameters.kp_pos" value="${safeNum(params.kp_pos, 1.2)}" /></div>
        <div class="field"><label>${labelWithTip("kd_pos", "Velocity damping gain. Increase to reduce overshoot.")}</label><input type="number" step="0.05" min="0" data-vk="parameters.kd_pos" value="${safeNum(params.kd_pos, 1.4)}" /></div>
      </div>` : `
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Cruise speed (m/s)", "Target airspeed throughout the mission.")}</label><input type="number" step="0.5" min="1" data-vk="parameters.cruise_speed_mps" value="${safeNum(params.cruise_speed_mps, 40)}" /></div>
        <div class="field"><label>${labelWithTip("Capture radius (m)", "Horizontal distance to advance to next waypoint.")}</label><input type="number" step="1" min="0" data-vk="parameters.waypoint_capture_radius_m" value="${safeNum(params.waypoint_capture_radius_m, 75)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Min speed (m/s)", "Lower speed bound.")}</label><input type="number" step="0.5" min="1" data-vk="parameters.min_speed_mps" value="${safeNum(params.min_speed_mps, 25)}" /></div>
        <div class="field"><label>${labelWithTip("Max speed (m/s)", "Upper speed bound.")}</label><input type="number" step="0.5" min="1" data-vk="parameters.max_speed_mps" value="${safeNum(params.max_speed_mps, 70)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Max bank (deg)", "Maximum bank angle for turns.")}</label><input type="number" step="1" min="0" max="89" data-vk="parameters.max_bank_deg" value="${safeNum(params.max_bank_deg, 30)}" /></div>
        <div class="field"><label>${labelWithTip("Turn rate limit (deg/s)", "Cap on heading rate of change.")}</label><input type="number" step="0.5" min="0" data-vk="parameters.turn_rate_limit_deg_s" value="${safeNum(params.turn_rate_limit_deg_s, 10)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Max climb (deg)", "Maximum flight-path climb angle.")}</label><input type="number" step="0.5" min="0" max="45" data-vk="parameters.max_climb_deg" value="${safeNum(params.max_climb_deg, 8)}" /></div>
        <div class="field"><label>${labelWithTip("Max descent (deg)", "Maximum flight-path descent angle.")}</label><input type="number" step="0.5" min="0" max="45" data-vk="parameters.max_descent_deg" value="${safeNum(params.max_descent_deg, 8)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Heading gain", "Proportional gain from heading error to turn command.")}</label><input type="number" step="0.1" min="0" data-vk="parameters.heading_gain" value="${safeNum(params.heading_gain, 1.5)}" /></div>
        <div class="field"><label>${labelWithTip("Altitude gain", "Proportional gain from altitude error to flight-path command.")}</label><input type="number" step="0.005" min="0" data-vk="parameters.altitude_gain" value="${safeNum(params.altitude_gain, 0.03)}" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label>${labelWithTip("Climb rate limit (m/s)", "Max vertical rate of change.")}</label><input type="number" step="0.1" min="0" data-vk="parameters.climb_rate_limit_mps" value="${safeNum(params.climb_rate_limit_mps, 4)}" /></div>
      </div>`;
    return `
      <div class="insp-section">
        <h3 class="insp-title">${isQuad ? "3DOF Quadcopter" : "3DOF Fixed-Wing"} Parameters</h3>
        <div class="card">
          <div class="field">
            <label>${labelWithTip("Model preset", "Apply a preset parameter set. You can further adjust values below.")}</label>
            <select data-pointmass-model>
              <option value="" ${activeModel ? "" : "selected"}>Custom</option>
              ${modelOptions}
            </select>
          </div>
          ${quadFields}
        </div>
      </div>
    `;
  }

  function fixedWingSection(vehicle) {
    const params = vehicle.parameters || {};
    const controller = vehicle.controller || {};
    const altRef = String(params.altitude_reference || "agl").toLowerCase() === "msl" ? "msl" : "agl";
    const engineOn = params.engine_running !== false && params.engine_running !== "false";
    const autoTrim = params.auto_trim_elevator === true || params.auto_trim_elevator === "true";
    const activePreset = (store.scenario?.metadata?.jsbsim_preset || "").trim();
    const presetOptions = (store.jsbsimPresets || [])
      .map(
        (p) =>
          `<option value="${escapeAttr(p.id)}" ${p.id === activePreset ? "selected" : ""}>${escapeHtml(p.label)}</option>`
      )
      .join("");
    const { catalogId: activeAircraft, isCustom: aircraftIsCustom } = resolveActiveJsbsimAircraft(
      params,
      store.scenario?.metadata
    );
    const aircraftOptions = (store.jsbsimAircraft || [])
      .map(
        (a) =>
          `<option value="${escapeAttr(a.id)}" ${a.id === activeAircraft ? "selected" : ""}>${escapeHtml(a.label)}</option>`
      )
      .join("");
    return `
      <div class="insp-section">
        <h3 class="insp-title">Fixed-Wing Vehicle</h3>
        <div class="card">
          <div class="field">
            <label>${labelWithTip("Aircraft model", "Pick a prebuilt JSBSim aircraft with baseline autopilot tuning, or Custom to enter a model name manually.")}</label>
            <select data-jsbsim-aircraft>
              <option value="" ${aircraftIsCustom ? "selected" : ""}>Custom</option>
              ${aircraftOptions}
            </select>
          </div>
          <div class="field" ${aircraftIsCustom ? "" : 'style="display:none;"'} data-jsbsim-aircraft-custom-wrap>
            <label>${labelWithTip("Custom aircraft name", "JSBSim folder/model name passed to load_model (e.g. c172p, pa28).")}</label>
            <input type="text" data-jsbsim-aircraft-custom data-vtext="parameters.aircraft" value="${escapeAttr(params.aircraft || params.aircraft_xml || "c172p")}" />
          </div>
          <div class="field">
            <label>${labelWithTip("Flight profile preset", "Apply run settings and controller tuning for a flight profile. Pick aircraft first, then profile. Choose Custom to edit manually.")}</label>
            <select data-jsbsim-preset>
              <option value="" ${activePreset ? "" : "selected"}>Custom</option>
              ${presetOptions}
            </select>
          </div>
          <div class="field-row">
            <div class="field">
              <label>${labelWithTip("Altitude reference", "Whether waypoint and spawn heights are above ground level (AGL) or mean sea level (MSL).")}</label>
              <select data-vtext="parameters.altitude_reference">
                <option value="agl" ${altRef === "agl" ? "selected" : ""}>AGL (above terrain)</option>
                <option value="msl" ${altRef === "msl" ? "selected" : ""}>MSL (absolute)</option>
              </select>
            </div>
            <div class="field"><label>${labelWithTip("Initial IAS (m/s)", "Calibrated airspeed at simulation start.")}</label><input type="number" step="0.5" data-vk="parameters.initial_ias_mps" value="${safeNum(params.initial_ias_mps, 40)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Initial heading (deg)", "Leave blank to auto-align with the first leg (WP0 toward WP1).")}</label><input type="number" step="0.1" data-vtext="parameters.initial_heading_deg" value="${params.initial_heading_deg != null && params.initial_heading_deg !== "" ? escapeAttr(params.initial_heading_deg) : ""}" placeholder="auto" /></div>
            <div class="field"><label>${labelWithTip("Initial flight-path (deg)", "Climb/descent angle at start. 0 is level.")}</label><input type="number" step="0.5" data-vk="parameters.initial_flight_path_deg" value="${safeNum(params.initial_flight_path_deg, 0)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Initial pitch (deg)", "Pitch attitude at start.")}</label><input type="number" step="0.5" data-vk="parameters.initial_pitch_deg" value="${safeNum(params.initial_pitch_deg, 0)}" /></div>
            <div class="field"><label>${labelWithTip("Initial roll (deg)", "Roll attitude at start.")}</label><input type="number" step="0.5" data-vk="parameters.initial_roll_deg" value="${safeNum(params.initial_roll_deg, 0)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Cruise speed (m/s)", "Target airspeed (clamped to 15–70 m/s in the backend).")}</label><input type="number" step="0.5" data-vk="controller.cruise_speed_mps" value="${safeNum(controller.cruise_speed_mps, 40)}" /></div>
            <div class="field"><label>${labelWithTip("Waypoint capture radius (m)", "Horizontal distance to advance to the next waypoint.")}</label><input type="number" step="1" data-vk="controller.waypoint_capture_radius_m" value="${safeNum(controller.waypoint_capture_radius_m, 75)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Max bank (deg)", "Maximum intended bank for heading control.")}</label><input type="number" step="1" data-vk="controller.max_bank_deg" value="${safeNum(controller.max_bank_deg, 25)}" /></div>
            <div class="field"><label>${labelWithTip("Min AGL (m)", "Minimum altitude above terrain before protective climb.")}</label><input type="number" step="1" data-vk="controller.min_agl_m" value="${safeNum(controller.min_agl_m, 10)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Base throttle", "Nominal throttle command before speed correction.")}</label><input type="number" step="0.01" min="0" max="1" data-vk="controller.base_throttle" value="${safeNum(controller.base_throttle, 0.65)}" /></div>
            <div class="field"><label>${labelWithTip("Heading gain", "Aileron response to heading error.")}</label><input type="number" step="0.1" data-vk="controller.heading_gain" value="${safeNum(controller.heading_gain, 1.5)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Altitude gain", "Scales commanded flight-path angle from altitude error.")}</label><input type="number" step="0.001" data-vk="controller.altitude_gain" value="${safeNum(controller.altitude_gain, 0.012)}" /></div>
            <div class="field"><label>${labelWithTip("Pitch gain", "Optional override for vertical tracking; defaults from altitude gain.")}</label><input type="number" step="0.01" data-vk="controller.pitch_gain" value="${safeNum(controller.pitch_gain, controller.altitude_gain != null ? controller.altitude_gain * 80 : 0.96)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Throttle gain", "Throttle response to airspeed error.")}</label><input type="number" step="0.001" data-vk="controller.throttle_gain" value="${safeNum(controller.throttle_gain, 0.02)}" /></div>
            <div class="field"><label>${labelWithTip("Elevator trim", "Baseline elevator command.")}</label><input type="number" step="0.01" min="-1" max="1" data-vk="controller.elevator_trim" value="${safeNum(controller.elevator_trim, 0)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Climb-rate gain", "Altitude error to commanded climb rate (m/s per m). Defaults to altitude gain.")}</label><input type="number" step="0.001" data-vk="controller.climb_rate_gain" value="${safeNum(controller.climb_rate_gain, controller.altitude_gain != null ? controller.altitude_gain : 0.012)}" /></div>
            <div class="field"><label>${labelWithTip("Climb-rate limit (m/s)", "Caps commanded vertical speed.")}</label><input type="number" step="0.1" data-vk="controller.climb_rate_limit_mps" value="${safeNum(controller.climb_rate_limit_mps, 4)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Elevator gain", "Climb-rate error to elevator command.")}</label><input type="number" step="0.01" data-vk="controller.elevator_gain" value="${safeNum(controller.elevator_gain, 0.12)}" /></div>
            <div class="field"><label>${labelWithTip("Gamma rate limit (deg/s)", "Limits how fast commanded climb rate can change.")}</label><input type="number" step="0.1" data-vk="controller.gamma_rate_limit_deg_s" value="${safeNum(controller.gamma_rate_limit_deg_s, 4)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Max sink rate (m/s)", "Triggers extra climb command when exceeded.")}</label><input type="number" step="0.5" data-vk="controller.max_sink_rate_mps" value="${safeNum(controller.max_sink_rate_mps, 5)}" /></div>
            <div class="field"><label>${labelWithTip("Max climb (deg)", "Steep-climb protection limit (legacy angle cap).")}</label><input type="number" step="0.5" data-vk="controller.max_climb_deg" value="${safeNum(controller.max_climb_deg, 8)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Max descent (deg)", "Steep-descent protection limit.")}</label><input type="number" step="0.5" data-vk="controller.max_descent_deg" value="${safeNum(controller.max_descent_deg, 8)}" /></div>
            <div class="field"><label>${labelWithTip("JSBSim root/path", "Optional root directory for JSBSim aircraft, engine, and systems data. Leave blank for package defaults.")}</label><input type="text" data-vtext="parameters.jsbsim_root" value="${escapeAttr(params.jsbsim_root || "")}" placeholder="auto" /></div>
          </div>
        </div>
      </div>
      <div class="insp-section vehicle-detail">
        <details>
          <summary>Fixed-Wing Advanced</summary>
          <div class="card">
            <div class="field-row">
              <div class="field"><label>${labelWithTip("Aircraft data path", "Optional aircraft XML directory for load_model_with_paths.")}</label><input type="text" data-vtext="parameters.aircraft_path" value="${escapeAttr(params.aircraft_path || "")}" placeholder="auto" /></div>
              <div class="field"><label>${labelWithTip("Engine data path", "Optional engine XML directory.")}</label><input type="text" data-vtext="parameters.engine_path" value="${escapeAttr(params.engine_path || "")}" placeholder="auto" /></div>
            </div>
            <div class="field-row">
              <div class="field"><label>${labelWithTip("Systems data path", "Optional systems XML directory.")}</label><input type="text" data-vtext="parameters.systems_path" value="${escapeAttr(params.systems_path || "")}" placeholder="auto" /></div>
              <div class="field"><label>${labelWithTip("Flap command", "Normalized flap deflection (0–1).")}</label><input type="number" step="0.01" min="0" max="1" data-vk="parameters.flap_cmd_norm" value="${safeNum(params.flap_cmd_norm, 0)}" /></div>
            </div>
            <div class="field-row">
              <div class="field toggle">
                <span class="toggle-label">${labelWithTip("Engine running at start", "Starts the JSBSim piston engine before the run loop.")}</span>
                <label class="switch"><input type="checkbox" data-vehicle-bool="parameters.engine_running" ${engineOn ? "checked" : ""} /><span class="slider"></span></label>
              </div>
              <div class="field toggle">
                <span class="toggle-label">${labelWithTip("Auto elevator trim at IC", "Search elevator trim after engine start for near-level flight.")}</span>
                <label class="switch"><input type="checkbox" data-vehicle-bool="parameters.auto_trim_elevator" ${autoTrim ? "checked" : ""} /><span class="slider"></span></label>
              </div>
            </div>
            <div class="field"><label>${labelWithTip("IC trim search steps", "Iterations for auto elevator trim (advanced).")}</label><input type="number" step="1" data-vk="parameters.ic_trim_steps" value="${safeNum(params.ic_trim_steps, 24)}" /></div>
          </div>
        </details>
      </div>
    `;
  }

  function quadVehicleSections(params, aero) {
    return `
      <div class="insp-section">
        <h3 class="insp-title">Vehicle Mass &amp; Inertia</h3>
        <div class="card">
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Mass (kg)", "Vehicle mass used by translational dynamics and Monte Carlo mass jitter.")}</label><input type="number" step="0.01" data-vk="parameters.mass" value="${safeNum(params.mass, 5)}" /></div>
            <div class="field"><label>${labelWithTip("Ix (kg·m²)", "Roll-axis moment of inertia.")}</label><input type="number" step="0.001" data-vk="parameters.Ix" value="${safeNum(params.Ix, 1)}" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>${labelWithTip("Iy (kg·m²)", "Pitch-axis moment of inertia.")}</label><input type="number" step="0.001" data-vk="parameters.Iy" value="${safeNum(params.Iy, 1)}" /></div>
            <div class="field"><label>${labelWithTip("Iz (kg·m²)", "Yaw-axis moment of inertia.")}</label><input type="number" step="0.001" data-vk="parameters.Iz" value="${safeNum(params.Iz, 1.5)}" /></div>
          </div>
        </div>
      </div>

      <div class="insp-section vehicle-detail">
        <details open>
          <summary>Aerodynamics</summary>
          <div class="card">
            <div class="field-row">
              <div class="field"><label>${labelWithTip("Linear drag coefficient", "Velocity-proportional drag term for extended in-house dynamics.")}</label><input type="number" step="0.001" data-vk="parameters.aero.cd_linear" value="${safeNum(aero.cd_linear, 0)}" /></div>
              <div class="field"><label>${labelWithTip("Quadratic drag coefficient", "Speed-squared drag term for extended in-house dynamics.")}</label><input type="number" step="0.001" data-vk="parameters.aero.cd_quadratic" value="${safeNum(aero.cd_quadratic, 0)}" /></div>
            </div>
            <div class="field"><label>${labelWithTip("Reference area (m²)", "Frontal reference area used to scale aerodynamic drag force.")}</label><input type="number" step="0.001" data-vk="parameters.aero.reference_area_m2" value="${safeNum(aero.reference_area_m2, 0.1)}" /></div>
          </div>
        </details>
      </div>
    `;
  }

  function actionSection(rc) {
    return `
      <div class="insp-section">
        <div class="btn-row">
          <button class="btn" id="save-scenario">Save</button>
          <button class="btn" id="cancel-run" hidden>Cancel</button>
          <button class="btn primary" id="run-scenario">${runButtonLabel(rc)}</button>
        </div>
        <div id="validation"></div>
      </div>
    `;
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
    inspector.querySelectorAll("#inspector-tabs button").forEach((btn) =>
      btn.addEventListener("click", () => {
        store.createInspectorTab = btn.dataset.tab === "vehicle" ? "vehicle" : "scenario";
        render();
      })
    );

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
        const intKeys = ["max_steps", "horizon", "lookahead"];
        const textKeys = ["backend_id", "integration_method", "fidelity_mode"];
        if (textKeys.includes(k)) {
          store.scenario.run_config[k] = el.value;
          if (k === "backend_id") {
            applyVehiclePresetForBackend(store.scenario, el.value);
            render();
          }
          return;
        }
        if (k === "seed") {
          const raw = String(el.value || "").trim();
          store.scenario.run_config.seed = raw === "" ? null : parseIntOr(raw, store.scenario.run_config.seed ?? 0);
          return;
        }
        store.scenario.run_config[k] = intKeys.includes(k)
          ? parseIntOr(el.value, store.scenario.run_config[k])
          : parseFloatOr(el.value, store.scenario.run_config[k]);
      })
    );
    inspector.querySelectorAll("[data-vk]").forEach((el) =>
      el.addEventListener("change", () => {
        const path = el.dataset.vk;
        const prior = getByPath(store.scenario.vehicle, path, 0);
        setByPath(store.scenario.vehicle, path, parseFloatOr(el.value, prior));
      })
    );
    inspector.querySelectorAll("[data-vtext]").forEach((el) =>
      el.addEventListener("change", () => {
        setByPath(store.scenario.vehicle, el.dataset.vtext, String(el.value || "").trim());
      })
    );
    inspector.querySelectorAll("[data-vehicle-bool]").forEach((el) =>
      el.addEventListener("change", () => {
        setByPath(store.scenario.vehicle, el.dataset.vehicleBool, !!el.checked);
      })
    );
    inspector.querySelectorAll("[data-jsbsim-aircraft]").forEach((el) =>
      el.addEventListener("change", async () => {
        const aircraftId = String(el.value || "").trim();
        if (!aircraftId) {
          if (store.scenario.metadata) delete store.scenario.metadata.jsbsim_aircraft;
          render();
          return;
        }
        try {
          const merged = await api.applyJsbsimAircraft(store.scenario, aircraftId);
          mergeScenarioFromPreset(store.scenario, merged);
          ensureVehicleObj(store.scenario);
          setStatus(`Applied JSBSim aircraft: ${aircraftId}`, "ok");
          render();
        } catch (e) {
          setStatus(`Aircraft preset failed: ${e.message}`, "err");
        }
      })
    );
    inspector.querySelectorAll("[data-jsbsim-preset]").forEach((el) =>
      el.addEventListener("change", async () => {
        const presetId = String(el.value || "").trim();
        if (!presetId) {
          if (store.scenario.metadata) delete store.scenario.metadata.jsbsim_preset;
          return;
        }
        try {
          const merged = await api.applyJsbsimPreset(store.scenario, presetId);
          mergeScenarioFromPreset(store.scenario, merged);
          ensureVehicleObj(store.scenario);
          setStatus(`Applied JSBSim preset: ${presetId}`, "ok");
          render();
        } catch (e) {
          setStatus(`Preset failed: ${e.message}`, "err");
        }
      })
    );
    inspector.querySelectorAll("[data-pointmass-model]").forEach((el) =>
      el.addEventListener("change", async () => {
        const modelId = String(el.value || "").trim();
        if (!modelId) {
          if (store.scenario.metadata) delete store.scenario.metadata.pointmass_model;
          render();
          return;
        }
        try {
          const merged = await api.applyPointmassModel(store.scenario, modelId);
          mergeScenarioFromPreset(store.scenario, merged);
          ensureRunConfigObj(store.scenario);
          setStatus(`Applied 3DOF preset: ${modelId}`, "ok");
          render();
        } catch (e) {
          setStatus(`Point-mass preset failed: ${e.message}`, "err");
        }
      })
    );
    inspector.querySelectorAll("[data-ek]").forEach((el) =>
      el.addEventListener("change", () => {
        const k = el.dataset.ek;
        if (el.type === "checkbox") {
          store.scenario.environment[k] = !!el.checked;
        } else {
          store.scenario.environment[k] = parseFloatOr(el.value, store.scenario.environment[k]);
        }
      })
    );
    inspector.querySelectorAll("[data-wind-i]").forEach((el) =>
      el.addEventListener("change", () => {
        ensureEnvironmentObj(store.scenario);
        const idx = parseInt(el.dataset.windI, 10);
        if (!Number.isFinite(idx)) return;
        store.scenario.environment.wind_mps[idx] = parseFloatOr(el.value, store.scenario.environment.wind_mps[idx]);
      })
    );
    inspector.querySelectorAll("[data-mck]").forEach((el) => {
      if (!store.scenario.run_config.monte_carlo) {
        store.scenario.run_config.monte_carlo = {};
      }
      el.addEventListener("change", () => {
        const k = el.dataset.mck;
        const intKeys = ["n_trials", "workers", "base_seed"];
        store.scenario.run_config.monte_carlo[k] = intKeys.includes(k)
          ? parseIntOr(el.value, store.scenario.run_config.monte_carlo[k] ?? 0)
          : parseFloatOr(el.value, store.scenario.run_config.monte_carlo[k] ?? 0);
      });
    });

    const fetchRemote = inspector.querySelector("#fetch-remote");
    if (fetchRemote) fetchRemote.checked = false;
    const buildMapBtn = inspector.querySelector("#build-map");
    if (buildMapBtn) buildMapBtn.addEventListener("click", () => {
      const remote = inspector.querySelector("#fetch-remote").checked;
      ctx.buildMap(remote);
    });

    const scaleInput = inspector.querySelector("#xyz-scale");
    if (scaleInput)
      scaleInput.addEventListener("change", () => {
        store.xyz.scale_m = parseFloatOr(scaleInput.value, store.xyz.scale_m);
        if (ctx.scene.setGround) ctx.scene.setGround(store.xyz.scale_m);
      });
    inspector.querySelectorAll("[data-scale-preset]").forEach((b) =>
      b.addEventListener("click", () => {
        store.xyz.scale_m = parseFloat(b.dataset.scalePreset);
        if (ctx.scene.setGround) ctx.scene.setGround(store.xyz.scale_m);
        render();
      })
    );
    const loadMapBtn = inspector.querySelector("#xyz-load-map");
    if (loadMapBtn) loadMapBtn.addEventListener("click", () => openMapSourceModal(ctx));
    const clearMapBtn = inspector.querySelector("#xyz-clear-map");
    if (clearMapBtn)
      clearMapBtn.addEventListener("click", () => {
        store.xyz.mapInfo = null;
        if (ctx.scene.clearMap) ctx.scene.clearMap();
        ctx.refreshEntities();
        setStatus("Custom map cleared.", "ok");
        render();
      });

    inspector.querySelectorAll("#map-mode-switch button").forEach((b) =>
      b.addEventListener("click", () => ctx.switchMapMode(b.dataset.mode))
    );

    inspector.querySelectorAll("#edit-mode button").forEach((b) =>
      b.addEventListener("click", () => {
        store.editMode = b.dataset.mode;
        if (store.editMode !== "edit") setSelection(null, null);
        render();
        refreshEntities();
      })
    );

    const pendingAlt = inspector.querySelector("#pending-alt");
    if (pendingAlt)
      pendingAlt.addEventListener("change", (e) => {
        store.pendingAltitude = parseFloatOr(e.target.value, 0);
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

    const saveBtn = inspector.querySelector("#save-scenario");
    if (saveBtn) saveBtn.addEventListener("click", saveScenario);
    const runBtn = inspector.querySelector("#run-scenario");
    if (runBtn) runBtn.addEventListener("click", runScenario);
    const cancelBtn = inspector.querySelector("#cancel-run");
    if (cancelBtn) cancelBtn.addEventListener("click", cancelRun);
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
    const pos = kind === "waypoint" ? wpToLocal(item) : markerToLocal(item);
    ctx.scene.centerOn(pos);
  }

  function refreshEntities() {
    const s = store.scenario;
    const sel = store.selection;
    const ratio = waypointMarkerRatio();
    const wps = s.waypoints.waypoints.map((wp, i) => {
      const selected = sel.kind === "waypoint" && sel.index === i;
      return {
        index: i,
        ...wpToLocal(wp),
        label: wp.label || "WP" + i,
        selected,
        pixelSize: Math.round((selected ? 18 : 12) * ratio),
        radius: (selected ? 1.0 : 0.7) * ratio,
      };
    });
    const markers = s.markers.markers.map((m, i) => ({
      index: i,
      ...markerToLocal(m),
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
          x_m: evt.x,
          y_m: evt.y,
          z_m: store.pendingAltitude,
        });
      } else if (mode === "marker") {
        resp = await api.addMarker({
          scenario: store.scenario,
          x_m: evt.x,
          y_m: evt.y,
          z_m: store.pendingAltitude,
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
    ensureMonteCarlo();
    const nTrials = store.scenario.run_config.monte_carlo?.n_trials ?? 1;
    const isMc = nTrials > 1;
    setStatus(isMc ? "Launching Monte Carlo\u2026" : "Launching run\u2026", "busy");
    store.mcBatch = isMc ? { trials: [], analysis: null } : null;
    store.run = null;
    try {
      const { run_token } = await api.startRun({ scenario: store.scenario });
      store.runToken = run_token;
      showCancel(true);
      streamRun(run_token, isMc);
    } catch (e) {
      const msgs = (e.data && e.data.detail && e.data.detail.messages) || [e.message];
      setValidation(msgs, false);
      setStatus("Run failed to start.", "err");
      showCancel(false);
    }
  }

  async function cancelRun() {
    if (!store.runToken) return;
    try {
      await api.cancelRun(store.runToken);
      setStatus("Cancelling run\u2026", "busy");
    } catch (e) {
      setStatus(`Cancel failed: ${e.message}`, "err");
    }
  }

  function showCancel(visible) {
    const btn = inspector.querySelector("#cancel-run");
    if (btn) btn.hidden = !visible;
  }

  function streamRun(token, isMc) {
    if (store.runSocket) {
      try {
        store.runSocket.close();
      } catch (_e) {
        /* ignore */
      }
    }
    store.runSocket = openRunSocket(
      token,
      async (ev) => {
        if (ev.type === "step_progress") {
          setStatus(`Simulating \u2014 step ${ev.step} / ${ev.total}`, "busy");
        } else if (ev.type === "trial_progress") {
          setStatus(`Monte Carlo \u2014 trial ${ev.done} / ${ev.total}`, "busy");
        } else if (ev.type === "result") {
          if (isMc && store.mcBatch) {
            store.mcBatch.trials.push(ev);
            setStatus(
              `Trial ${(ev.trial_index ?? 0) + 1} ${ev.success ? "succeeded" : ev.status} \u2014 miss ${fmt(ev.miss_distance_m)} m.`,
              ev.success ? "ok" : "warn"
            );
          } else {
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
          }
        } else if (ev.type === "error") {
          const prefix = ev.trial_index != null ? `Trial ${ev.trial_index + 1}: ` : "";
          setStatus(`${prefix}Run error: ${ev.message}`, "err");
        } else if (ev.type === "done") {
          showCancel(false);
          store.runSocket = null;
          if (ev.mode === "monte_carlo" && store.mcBatch?.trials?.length) {
            await finishMonteCarlo();
          } else if (store.run) {
            ctx.setView("replay");
          }
        }
      },
      () => {
        showCancel(false);
        store.runSocket = null;
      }
    );
  }

  async function finishMonteCarlo() {
    const paths = store.mcBatch.trials.map((t) => t.path).filter(Boolean);
    if (!paths.length) {
      setStatus("Monte Carlo finished but no runs were saved.", "warn");
      return;
    }
    setStatus("Building Monte Carlo analysis\u2026", "busy");
    try {
      const analysis = await api.mcAnalysis(paths);
      store.mcBatch.analysis = analysis;
      store.mcBatch.trials = store.mcBatch.trials.map((t, i) => ({
        ...t,
        ...(analysis.trials[i] || {}),
      }));
      setStatus(`Monte Carlo complete (${paths.length} trials).`, "ok");
      ctx.setView("analysis");
    } catch (e) {
      setStatus(`MC analysis failed: ${e.message}`, "err");
    }
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

// ---- map section builders (mode-aware) ----
function mapSectionLla(s) {
  return `
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
    </div>`;
}

function mapSectionXyz() {
  const scale = store.xyz.scale_m;
  const loaded = !!store.xyz.mapInfo;
  const presets = [1, 10, 100, 1000];
  return `
    <div class="card">
      <div class="hint" style="margin-bottom:8px;">Generic XYZ space (meters). Center is the origin.</div>
      <div class="field"><label>Scale / radius (m)</label>
        <input type="number" step="1" min="1" id="xyz-scale" value="${scale}" ${loaded ? "disabled" : ""} /></div>
      <div class="segmented compact" style="margin-bottom:10px;">
        ${presets
          .map(
            (p) =>
              `<button data-scale-preset="${p}" class="${p === scale ? "active" : ""}" ${loaded ? "disabled" : ""}>${p >= 1000 ? p / 1000 + "km" : p + "m"}</button>`
          )
          .join("")}
      </div>
      <button class="btn primary full" id="xyz-load-map">${loaded ? "Replace tile map\u2026" : "Load tile map\u2026"}</button>
      ${loaded ? '<button class="btn full" id="xyz-clear-map" style="margin-top:8px;">Clear map</button>' : ""}
      ${loaded ? '<div class="validation ok" style="margin-top:8px;">Custom tile map loaded.</div>' : ""}
    </div>`;
}

// ---- map source modal (server cache + upload) ----
async function applyXyzMap(ctx, mapInfo) {
  store.xyz.mapInfo = mapInfo;
  if (ctx.scene.setMap) await ctx.scene.setMap(mapInfo);
  ctx.refreshEntities();
}

function openMapSourceModal(ctx) {
  const existing = document.getElementById("map-source-modal");
  if (existing) existing.remove();
  const overlay = document.createElement("div");
  overlay.id = "map-source-modal";
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h3 class="insp-title" style="margin:0;">Load tile map</h3>
        <button class="icon-btn" data-close aria-label="Close">\u2715</button>
      </div>
      <div class="segmented compact" id="ms-tabs">
        <button data-ms-tab="cache" class="active">Server cache</button>
        <button data-ms-tab="upload">Upload</button>
      </div>
      <div class="modal-body" id="ms-body"></div>
    </div>`;
  document.body.appendChild(overlay);

  const body = overlay.querySelector("#ms-body");
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  overlay.querySelector("[data-close]").addEventListener("click", close);

  function setTab(tab) {
    overlay.querySelectorAll("#ms-tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.msTab === tab)
    );
    if (tab === "cache") renderCacheTab();
    else renderUploadTab();
  }
  overlay.querySelectorAll("#ms-tabs button").forEach((b) =>
    b.addEventListener("click", () => setTab(b.dataset.msTab))
  );

  async function renderCacheTab() {
    body.innerHTML = '<div class="hint">Loading cached maps\u2026</div>';
    let caches = [];
    try {
      caches = await api.mapCaches();
    } catch (e) {
      body.innerHTML = `<div class="validation warn">Could not list caches: ${escapeHtml(e.message)}</div>`;
      return;
    }
    if (!caches.length) {
      body.innerHTML = '<div class="hint">No cached maps found in maps/cache.</div>';
      return;
    }
    body.innerHTML = `<div class="list">${caches
      .map(
        (c, i) =>
          `<div class="list-item" data-cache-i="${i}">
            <span class="swatch" style="background:var(--accent)"></span>${escapeHtml(c.key)}
            <span class="meta">${c.radius_km != null ? c.radius_km + " km" : ""} ${c.resolution ? "n" + c.resolution : ""}</span>
          </div>`
      )
      .join("")}</div>`;
    body.querySelectorAll("[data-cache-i]").forEach((el) =>
      el.addEventListener("click", async () => {
        const c = caches[parseInt(el.dataset.cacheI, 10)];
        setStatus("Loading cached map\u2026", "busy");
        try {
          const info = await api.buildMap(
            {
              center_lat: c.center_lat,
              center_lon: c.center_lon,
              radius_km: c.radius_km,
              resolution: c.resolution,
            },
            false
          );
          await applyXyzMap(ctx, info);
          setStatus("Custom tile map loaded.", "ok");
          close();
          ctx.views.create.show();
        } catch (e) {
          setStatus(`Map load failed: ${e.message}`, "err");
        }
      })
    );
  }

  function renderUploadTab() {
    body.innerHTML = `
      <div class="field"><label>Imagery image (PNG/JPG)</label><input type="file" id="ms-img" accept="image/*" /></div>
      <div class="field"><label>Elevation (optional: .npy or grayscale image)</label><input type="file" id="ms-elev" accept=".npy,image/*" /></div>
      <div class="field"><label>Map width (m)</label><input type="number" step="1" min="1" id="ms-scale" value="${store.xyz.scale_m * 2}" /></div>
      <button class="btn primary full" id="ms-upload">Upload &amp; load</button>
      <div id="ms-upload-status"></div>`;
    body.querySelector("#ms-upload").addEventListener("click", async () => {
      const imgEl = body.querySelector("#ms-img");
      const elevEl = body.querySelector("#ms-elev");
      const statusEl = body.querySelector("#ms-upload-status");
      if (!imgEl.files || !imgEl.files[0]) {
        statusEl.innerHTML = '<div class="validation warn">Choose an imagery image first.</div>';
        return;
      }
      const fd = new FormData();
      fd.append("imagery", imgEl.files[0]);
      if (elevEl.files && elevEl.files[0]) fd.append("elevation", elevEl.files[0]);
      fd.append("scale_m", String(parseFloatOr(body.querySelector("#ms-scale").value, store.xyz.scale_m * 2)));
      setStatus("Uploading map\u2026", "busy");
      try {
        const info = await api.uploadMap(fd);
        await applyXyzMap(ctx, info);
        setStatus("Custom tile map loaded.", "ok");
        close();
        ctx.views.create.show();
      } catch (e) {
        statusEl.innerHTML = `<div class="validation warn">${escapeHtml(e.message)}</div>`;
        setStatus(`Upload failed: ${e.message}`, "err");
      }
    });
  }

  setTab("cache");
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

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseFloatOr(value, fallback = 0) {
  const n = parseFloat(value);
  return Number.isFinite(n) ? n : safeNum(fallback, 0);
}

function parseIntOr(value, fallback = 0) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : parseInt(safeNum(fallback, 0), 10);
}

function optionHtml(value, selected) {
  const labels = {
    euler: "Euler",
    rk4: "RK4",
    auto: "Auto",
    legacy: "Legacy",
    extended: "Extended",
  };
  const text = labels[value] || value;
  return `<option value="${value}" ${value === selected ? "selected" : ""}>${text}</option>`;
}

function labelWithTip(label, tip) {
  if (!tip) return escapeHtml(label);
  return `<span class="label-with-tip"><span>${escapeHtml(label)}</span><span class="tip-badge" tabindex="0" role="button" aria-label="${escapeAttr(
    tip
  )}" data-tip="${escapeAttr(tip)}">?</span></span>`;
}

function getByPath(obj, path, fallback) {
  if (!obj || !path) return fallback;
  const value = path.split(".").reduce((acc, key) => (acc == null ? undefined : acc[key]), obj);
  return value == null ? fallback : value;
}

function setByPath(obj, path, value) {
  if (!obj || !path) return;
  const keys = path.split(".");
  let cursor = obj;
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i];
    if (cursor[key] == null || typeof cursor[key] !== "object") cursor[key] = {};
    cursor = cursor[key];
  }
  cursor[keys[keys.length - 1]] = value;
}

function ensureRunConfigObj(scenario) {
  if (!scenario) return;
  if (!scenario.run_config || typeof scenario.run_config !== "object") {
    scenario.run_config = {};
  }
  const rc = scenario.run_config;
  if (!rc.backend_id) rc.backend_id = scenario.vehicle?.backend_id || "inhouse_mpc_quad";
  if (!Number.isFinite(Number(rc.dt_s))) rc.dt_s = 0.1;
  if (!Number.isFinite(Number(rc.max_steps))) rc.max_steps = 250;
  if (!Number.isFinite(Number(rc.target_altitude_m))) rc.target_altitude_m = 5.0;
  if (!Number.isFinite(Number(rc.horizon))) rc.horizon = 20;
  if (!Number.isFinite(Number(rc.lookahead))) rc.lookahead = 60;
  if (!Number.isFinite(Number(rc.waypoint_threshold_m))) rc.waypoint_threshold_m = 0.25;
  if (rc.seed === undefined) rc.seed = null;
  if (!rc.integration_method) rc.integration_method = "euler";
  if (!rc.fidelity_mode) rc.fidelity_mode = "auto";
}

function isFixedWingVehicle(scenario, runConfig = null) {
  const backendId = runConfig?.backend_id || scenario?.run_config?.backend_id || scenario?.vehicle?.backend_id;
  return backendId === JSBSIM_CESSNA_BACKEND || scenario?.vehicle?.model_type === "fixed_wing";
}

function isPointMassVehicle(scenario, runConfig = null) {
  const backendId = runConfig?.backend_id || scenario?.run_config?.backend_id || scenario?.vehicle?.backend_id;
  return (
    backendId === POINTMASS_QUAD_BACKEND ||
    backendId === POINTMASS_FW_BACKEND ||
    scenario?.vehicle?.model_type === "pointmass_quad" ||
    scenario?.vehicle?.model_type === "pointmass_fixed_wing"
  );
}

function mergeScenarioFromPreset(target, merged) {
  if (!target || !merged) return;
  if (merged.vehicle) {
    target.vehicle = deepMergeObjects(target.vehicle || {}, merged.vehicle);
  }
  if (merged.run_config) {
    target.run_config = deepMergeObjects(target.run_config || {}, merged.run_config);
  }
  if (merged.metadata) {
    target.metadata = { ...(target.metadata || {}), ...merged.metadata };
  }
}

function deepMergeObjects(base, overlay) {
  const out = { ...base };
  for (const [key, value] of Object.entries(overlay || {})) {
    if (value && typeof value === "object" && !Array.isArray(value) && typeof out[key] === "object" && out[key]) {
      out[key] = deepMergeObjects(out[key], value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

async function loadJsbsimPresets() {
  try {
    const body = await api.jsbsimPresets();
    store.jsbsimPresets = body.presets || [];
  } catch (_e) {
    store.jsbsimPresets = [];
  }
}

async function loadJsbsimAircraft() {
  try {
    const body = await api.jsbsimAircraft();
    store.jsbsimAircraft = body.aircraft || [];
  } catch (_e) {
    store.jsbsimAircraft = [];
  }
}

async function loadPointmassModels() {
  try {
    const body = await api.pointmassModels();
    store.pointmassModels = body.models || [];
  } catch (_e) {
    store.pointmassModels = [];
  }
}

function resolveActiveJsbsimAircraft(params, metadata) {
  const catalogId = String(metadata?.jsbsim_aircraft || "").trim();
  if (catalogId) return { catalogId, isCustom: false };
  const modelName = String(params?.aircraft || params?.aircraft_xml || "").trim();
  const match = (store.jsbsimAircraft || []).find(
    (a) => a.jsbsim_model === modelName || a.id === modelName
  );
  if (match) return { catalogId: match.id, isCustom: false };
  return { catalogId: "", isCustom: true };
}

function applyVehiclePresetForBackend(scenario, backendId) {
  if (!scenario) return;
  if (!scenario.vehicle || typeof scenario.vehicle !== "object") scenario.vehicle = {};
  const vehicle = scenario.vehicle;
  if (!vehicle.parameters || typeof vehicle.parameters !== "object") vehicle.parameters = {};
  if (!vehicle.controller || typeof vehicle.controller !== "object") vehicle.controller = {};

  vehicle.backend_id = backendId;
  if (backendId === JSBSIM_CESSNA_BACKEND) {
    vehicle.model_type = "fixed_wing";
    if (!vehicle.model_id || vehicle.model_id === "jsbsim_c172") vehicle.model_id = "jsbsim_c172p";
    if (!vehicle.display_name || vehicle.display_name === "JSBSim Cessna 172") vehicle.display_name = "Cessna 172P";
    if (!vehicle.parameters.aircraft && !vehicle.parameters.aircraft_xml) vehicle.parameters.aircraft = "c172p";
    if (!vehicle.parameters.altitude_reference) vehicle.parameters.altitude_reference = "agl";
    if (!Number.isFinite(Number(vehicle.parameters.initial_ias_mps))) vehicle.parameters.initial_ias_mps = 40.0;
    if (vehicle.parameters.engine_running === undefined) vehicle.parameters.engine_running = true;
    if (!Number.isFinite(Number(vehicle.parameters.base_throttle))) vehicle.parameters.base_throttle = 0.65;
    if (!vehicle.controller.type || vehicle.controller.type === "mpc") vehicle.controller.type = "waypoint_autopilot";
    const cruise = Number(vehicle.controller.cruise_speed_mps);
    if (!Number.isFinite(cruise) || cruise < 15 || cruise > 70) vehicle.controller.cruise_speed_mps = 40.0;
    if (!Number.isFinite(Number(vehicle.controller.max_bank_deg))) vehicle.controller.max_bank_deg = 25.0;
    if (!Number.isFinite(Number(vehicle.controller.base_throttle))) vehicle.controller.base_throttle = 0.65;
    if (!Number.isFinite(Number(vehicle.controller.heading_gain))) vehicle.controller.heading_gain = 1.5;
    if (!Number.isFinite(Number(vehicle.controller.altitude_gain))) vehicle.controller.altitude_gain = 0.012;
    if (!Number.isFinite(Number(vehicle.controller.throttle_gain))) vehicle.controller.throttle_gain = 0.02;
    if (!Number.isFinite(Number(vehicle.controller.waypoint_capture_radius_m))) {
      vehicle.controller.waypoint_capture_radius_m = 75.0;
    }
    if (!Number.isFinite(Number(vehicle.controller.min_agl_m))) vehicle.controller.min_agl_m = 10.0;
    if (!Number.isFinite(Number(vehicle.controller.max_sink_rate_mps))) vehicle.controller.max_sink_rate_mps = 5.0;
    if (!Number.isFinite(Number(vehicle.controller.max_climb_deg))) vehicle.controller.max_climb_deg = 8.0;
    if (!Number.isFinite(Number(vehicle.controller.max_descent_deg))) vehicle.controller.max_descent_deg = 8.0;
    if (!Number.isFinite(Number(vehicle.controller.elevator_trim))) vehicle.controller.elevator_trim = 0.0;
    if (!Number.isFinite(Number(vehicle.controller.climb_rate_limit_mps))) {
      vehicle.controller.climb_rate_limit_mps = 4.0;
    }
    if (!Number.isFinite(Number(vehicle.controller.elevator_gain))) vehicle.controller.elevator_gain = 0.12;
    vehicle.controller.elevator_sign = -1.0;
    if (!Number.isFinite(Number(vehicle.controller.gamma_rate_limit_deg_s))) {
      vehicle.controller.gamma_rate_limit_deg_s = 4.0;
    }
    if (!Number.isFinite(Number(vehicle.parameters.ic_trim_steps))) vehicle.parameters.ic_trim_steps = 24;
    if (!scenario.run_config || typeof scenario.run_config !== "object") scenario.run_config = {};
    const rc = scenario.run_config;
    const fixedWingMinAltM = 100.0;
    if (!Number.isFinite(Number(rc.target_altitude_m)) || Number(rc.target_altitude_m) < 30.0) {
      rc.target_altitude_m = fixedWingMinAltM;
    }
    if (scenario.waypoints && typeof scenario.waypoints === "object") {
      if (!Number.isFinite(Number(scenario.waypoints.default_alt_m)) || Number(scenario.waypoints.default_alt_m) < 30.0) {
        scenario.waypoints.default_alt_m = fixedWingMinAltM;
      }
      const wps = scenario.waypoints.waypoints;
      if (Array.isArray(wps)) {
        for (const wp of wps) {
          if (!wp || typeof wp !== "object") continue;
          const zVal = wp.z_m != null ? Number(wp.z_m) : Number(wp.alt_m);
          if (!Number.isFinite(zVal) || zVal < 30.0) {
            wp.z_m = fixedWingMinAltM;
            wp.alt_m = fixedWingMinAltM;
          }
        }
      }
    }
    if (!Number.isFinite(Number(rc.dt_s)) || Number(rc.dt_s) > 0.1) rc.dt_s = 0.05;
    if (!Number.isFinite(Number(rc.waypoint_threshold_m)) || Number(rc.waypoint_threshold_m) < 1) {
      rc.waypoint_threshold_m = 50.0;
    }
    return;
  }

  if (backendId === POINTMASS_QUAD_BACKEND) {
    vehicle.model_type = "pointmass_quad";
    if (!vehicle.model_id || !String(vehicle.model_id).startsWith("pointmass_quad")) {
      vehicle.model_id = "pointmass_quad_default";
    }
    if (!vehicle.display_name || vehicle.display_name === "JSBSim Cessna 172" || vehicle.display_name === "Cessna 172P") {
      vehicle.display_name = "3DOF Quadcopter";
    }
    vehicle.controller.type = "pointmass_pd";
    if (!Number.isFinite(Number(vehicle.parameters.mass))) vehicle.parameters.mass = 5.0;
    if (!Number.isFinite(Number(vehicle.parameters.max_accel_mps2))) vehicle.parameters.max_accel_mps2 = 5.0;
    if (!Number.isFinite(Number(vehicle.parameters.max_speed_mps))) vehicle.parameters.max_speed_mps = 10.0;
    if (!Number.isFinite(Number(vehicle.parameters.kp_pos))) vehicle.parameters.kp_pos = 1.2;
    if (!Number.isFinite(Number(vehicle.parameters.kd_pos))) vehicle.parameters.kd_pos = 1.4;
    if (!Number.isFinite(Number(vehicle.parameters.waypoint_capture_radius_m))) {
      vehicle.parameters.waypoint_capture_radius_m = 0.5;
    }
    if (!scenario.run_config || typeof scenario.run_config !== "object") scenario.run_config = {};
    const qrc = scenario.run_config;
    if (!Number.isFinite(Number(qrc.dt_s)) || Number(qrc.dt_s) > 0.5) qrc.dt_s = 0.1;
    if (!Number.isFinite(Number(qrc.max_steps)) || Number(qrc.max_steps) < 10) qrc.max_steps = 500;
    return;
  }

  if (backendId === POINTMASS_FW_BACKEND) {
    vehicle.model_type = "pointmass_fixed_wing";
    if (!vehicle.model_id || !String(vehicle.model_id).startsWith("pointmass_fw")) {
      vehicle.model_id = "pointmass_fw_default";
    }
    if (!vehicle.display_name || vehicle.display_name === "JSBSim Cessna 172" || vehicle.display_name === "Cessna 172P") {
      vehicle.display_name = "3DOF Fixed-Wing";
    }
    vehicle.controller.type = "pointmass_waypoint";
    if (!Number.isFinite(Number(vehicle.parameters.cruise_speed_mps))) vehicle.parameters.cruise_speed_mps = 40.0;
    if (!Number.isFinite(Number(vehicle.parameters.min_speed_mps))) vehicle.parameters.min_speed_mps = 25.0;
    if (!Number.isFinite(Number(vehicle.parameters.max_speed_mps))) vehicle.parameters.max_speed_mps = 70.0;
    if (!Number.isFinite(Number(vehicle.parameters.max_bank_deg))) vehicle.parameters.max_bank_deg = 30.0;
    if (!Number.isFinite(Number(vehicle.parameters.max_climb_deg))) vehicle.parameters.max_climb_deg = 8.0;
    if (!Number.isFinite(Number(vehicle.parameters.max_descent_deg))) vehicle.parameters.max_descent_deg = 8.0;
    if (!Number.isFinite(Number(vehicle.parameters.turn_rate_limit_deg_s))) vehicle.parameters.turn_rate_limit_deg_s = 10.0;
    if (!Number.isFinite(Number(vehicle.parameters.climb_rate_limit_mps))) vehicle.parameters.climb_rate_limit_mps = 4.0;
    if (!Number.isFinite(Number(vehicle.parameters.heading_gain))) vehicle.parameters.heading_gain = 1.5;
    if (!Number.isFinite(Number(vehicle.parameters.altitude_gain))) vehicle.parameters.altitude_gain = 0.03;
    if (!Number.isFinite(Number(vehicle.parameters.waypoint_capture_radius_m))) {
      vehicle.parameters.waypoint_capture_radius_m = 75.0;
    }
    if (!scenario.run_config || typeof scenario.run_config !== "object") scenario.run_config = {};
    const fwrc = scenario.run_config;
    const fwMinAltM = 100.0;
    if (!Number.isFinite(Number(fwrc.target_altitude_m)) || Number(fwrc.target_altitude_m) < 30.0) {
      fwrc.target_altitude_m = fwMinAltM;
    }
    if (!Number.isFinite(Number(fwrc.dt_s)) || Number(fwrc.dt_s) > 0.1) fwrc.dt_s = 0.05;
    if (!Number.isFinite(Number(fwrc.max_steps)) || Number(fwrc.max_steps) < 100) fwrc.max_steps = 2000;
    if (!Number.isFinite(Number(fwrc.waypoint_threshold_m)) || Number(fwrc.waypoint_threshold_m) < 1) {
      fwrc.waypoint_threshold_m = 75.0;
    }
    if (scenario.waypoints && typeof scenario.waypoints === "object") {
      if (!Number.isFinite(Number(scenario.waypoints.default_alt_m)) || Number(scenario.waypoints.default_alt_m) < 30.0) {
        scenario.waypoints.default_alt_m = fwMinAltM;
      }
      const wps = scenario.waypoints.waypoints;
      if (Array.isArray(wps)) {
        for (const wp of wps) {
          if (!wp || typeof wp !== "object") continue;
          const zVal = wp.z_m != null ? Number(wp.z_m) : Number(wp.alt_m);
          if (!Number.isFinite(zVal) || zVal < 30.0) {
            wp.z_m = fwMinAltM;
            wp.alt_m = fwMinAltM;
          }
        }
      }
    }
    return;
  }

  if (vehicle.model_type === "fixed_wing" || vehicle.model_type === "pointmass_quad" || vehicle.model_type === "pointmass_fixed_wing" || !vehicle.model_type) {
    vehicle.model_type = "quadcopter";
  }
  if (!vehicle.model_id || String(vehicle.model_id).startsWith("jsbsim_") || String(vehicle.model_id).startsWith("pointmass_")) {
    vehicle.model_id = backendId === "pybullet_quad" ? "pybullet_quad" : "inhouse_mpc_quad";
  }
  if (!vehicle.display_name || vehicle.display_name === "JSBSim Cessna 172" || vehicle.display_name === "Cessna 172P") {
    vehicle.display_name = backendId === "pybullet_quad" ? "PyBullet / PyFlyt Quadcopter" : "In-house MPC Quadcopter";
  }
  if (!vehicle.controller.type || vehicle.controller.type === "waypoint_autopilot" || vehicle.controller.type === "pointmass_pd" || vehicle.controller.type === "pointmass_waypoint") vehicle.controller.type = "mpc";
  if (!Number.isFinite(Number(vehicle.controller.horizon))) vehicle.controller.horizon = 20;
  if (!Number.isFinite(Number(vehicle.controller.lookahead))) vehicle.controller.lookahead = 60;
}

function ensureVehicleObj(scenario) {
  if (!scenario) return;
  if (!scenario.vehicle || typeof scenario.vehicle !== "object") scenario.vehicle = {};
  const backendId = scenario.run_config?.backend_id || scenario.vehicle.backend_id || "inhouse_mpc_quad";
  applyVehiclePresetForBackend(scenario, backendId);
  const vehicle = scenario.vehicle;
  if (!vehicle.parameters || typeof vehicle.parameters !== "object") vehicle.parameters = {};
  if (!vehicle.controller || typeof vehicle.controller !== "object") vehicle.controller = {};
  if (isFixedWingVehicle(scenario) || isPointMassVehicle(scenario)) return;
  const params = vehicle.parameters;
  if (!Number.isFinite(Number(params.mass))) params.mass = 5.0;
  if (!Number.isFinite(Number(params.Ix))) params.Ix = 1.0;
  if (!Number.isFinite(Number(params.Iy))) params.Iy = 1.0;
  if (!Number.isFinite(Number(params.Iz))) params.Iz = 1.5;
  if (!params.aero || typeof params.aero !== "object") params.aero = {};
  if (!Number.isFinite(Number(params.aero.cd_linear))) params.aero.cd_linear = 0.0;
  if (!Number.isFinite(Number(params.aero.cd_quadratic))) params.aero.cd_quadratic = 0.0;
  if (!Number.isFinite(Number(params.aero.reference_area_m2))) params.aero.reference_area_m2 = 0.1;
}

function ensureEnvironmentObj(scenario) {
  if (!scenario) return;
  if (!scenario.environment || typeof scenario.environment !== "object") scenario.environment = {};
  const env = scenario.environment;
  if (!Array.isArray(env.wind_mps)) env.wind_mps = [0, 0, 0];
  env.wind_mps = [0, 1, 2].map((i) => safeNum(env.wind_mps[i], 0));
  if (!Number.isFinite(Number(env.gust_std_mps))) env.gust_std_mps = 0.0;
  if (!Number.isFinite(Number(env.gust_decorrelation_s))) env.gust_decorrelation_s = 2.0;
  if (env.terrain_collision_enabled == null) env.terrain_collision_enabled = false;
  if (!Number.isFinite(Number(env.terrain_collision_offset_m))) env.terrain_collision_offset_m = 0.5;
  if (!Number.isFinite(Number(env.air_density_kg_m3))) env.air_density_kg_m3 = 1.225;
}

function ensureMonteCarloObj(scenario) {
  ensureRunConfigObj(scenario);
  const defaults = {
    enabled: false,
    n_trials: 1,
    workers: 1,
    base_seed: 0,
    init_pos_std: 0,
    init_vel_std: 0,
    init_att_std: 0,
    force_noise_std: 0,
    mass_jitter_pct: 0,
    inertia_jitter_pct: 0,
  };
  scenario.run_config.monte_carlo = {
    ...defaults,
    ...(scenario.run_config.monte_carlo || {}),
  };
}

function ensureMonteCarlo() {
  ensureMonteCarloObj(store.scenario);
}

function mcOpen() {
  const mc = store.scenario?.run_config?.monte_carlo;
  return mc && (mc.n_trials ?? 1) > 1 ? "open" : "";
}

function mcField(key, label, value, step, tip = "") {
  return `<div class="field"><label>${labelWithTip(label, tip)}</label><input type="number" step="${step}" data-mck="${key}" value="${value ?? 0}" /></div>`;
}

function runButtonLabel(rc) {
  const n = rc?.monte_carlo?.n_trials ?? 1;
  return n > 1 ? "Run Monte Carlo" : "Run";
}
