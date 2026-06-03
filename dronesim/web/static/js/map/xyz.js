// Custom XYZ 3D engine built on Three.js.
//
// Renders a generic local-meters space: an optional textured/displaced terrain
// (or a reference grid when no map is loaded), waypoint/marker points with
// labels, and replay polylines + a drone marker. Mirrors the public interface
// of CesiumScene but speaks local meters everywhere.
//
// Coordinate convention: scenario x_m = east, y_m = north, z_m = up.
// Three.js world: x = east, y = up, z = -north  ->  v(x,y,z) = (x_m, z_m, -y_m)

import { replayDroneRadius } from "../state.js";
import { localToThree, replayAxisArrows } from "./attitude.js";

const WP_COLOR = "#0a84ff";
const SEL_COLOR = "#ffd60a";
const MARKER_COLOR = "#ff453a";
const REF_COLOR = "#ff9f0a";
const TRAJ_COLOR = "#30d158";
const LOW_COLOR = "#ff453a";

const ZOOM_BASE = 0.85;
const ZOOM_MIN = 0.12;
const ZOOM_MAX = 1.0;
const WHEEL_ZOOM_SMOOTH = 0.22;
const WHEEL_ZOOM_EPS = 1e-4;

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export class CustomScene {
  constructor(containerId, { onClick, three } = {}) {
    this.container = document.getElementById(containerId);
    this.onClick = onClick;
    this.three = three;
    this.scaleM = 100;
    this.extentM = null; // { width, height } when a map is loaded
    this._raf = null;
    this._disposed = false;
    this._init();
  }

  _init() {
    const { THREE, OrbitControls, CSS2DRenderer } = this.three;
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h);
    renderer.domElement.classList.add("xyz-canvas");
    this.container.appendChild(renderer.domElement);
    this.renderer = renderer;

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(w, h);
    labelRenderer.domElement.classList.add("xyz-labels");
    this.container.appendChild(labelRenderer.domElement);
    this.labelRenderer = labelRenderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(cssVar("--bg", "#0b0b0d"));
    this.scene = scene;

    const camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 1_000_000);
    this.camera = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    this.controls = controls;
    controls.addEventListener("change", () => this._updateCameraClipping());

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(1, 2, 1.5);
    scene.add(sun);

    this.mapGroup = new THREE.Group();
    this.groundGroup = new THREE.Group();
    this.entityGroup = new THREE.Group();
    this.replayGroup = new THREE.Group();
    scene.add(this.mapGroup, this.groundGroup, this.entityGroup, this.replayGroup);

    // Large invisible plane at y=0 used as the click-picking fallback.
    const pickGeo = new THREE.PlaneGeometry(2_000_000, 2_000_000);
    pickGeo.rotateX(-Math.PI / 2);
    this.pickPlane = new THREE.Mesh(
      pickGeo,
      new THREE.MeshBasicMaterial({ visible: false })
    );
    scene.add(this.pickPlane);

    this.raycaster = new THREE.Raycaster();
    this._terrainMesh = null;
    this._pendingWheelZoom = 0;

    this._buildGround(this.scaleM);
    this._frameCamera(this.scaleM);
    this._wireInput();

    const animate = () => {
      if (this._disposed) return;
      this._raf = requestAnimationFrame(animate);
      this._applySmoothWheelZoom();
      this._updateZoomSpeed();
      controls.update();
      this._updateCameraClipping();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();
  }

  // ---- camera / orbit limits ----
  _sceneSpan() {
    const extW = this.extentM?.width ?? 0;
    const extH = this.extentM?.height ?? 0;
    return Math.max(extW, extH, this.scaleM * 2, 10);
  }

  _applyControlLimits() {
    const span = this._sceneSpan();
    const { THREE } = this.three;
    this.controls.minDistance = Math.max(2, span * 0.02);
    this.controls.maxDistance = Math.max(span * 6, 500);
    this.controls.minPolarAngle = Math.PI / 8;
    this.controls.maxPolarAngle = Math.PI / 2.2;
    this._updateZoomSpeed();

    const offset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    const dist = offset.length();
    if (dist > 0 && dist < this.controls.minDistance) {
      offset.multiplyScalar(this.controls.minDistance / dist);
      this.camera.position.copy(this.controls.target).add(offset);
    }
    this.controls.update();
  }

  _updateZoomSpeed() {
    const span = this._sceneSpan();
    const dist = this.camera.position.distanceTo(this.controls.target);
    const ratio = span / Math.max(dist, span * 0.02);
    this.controls.zoomSpeed = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, ZOOM_BASE * ratio));
  }

  _normalizeWheelDelta(e) {
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 16;
    else if (e.deltaMode === 2) dy *= 100;
    const dpr = Math.max(window.devicePixelRatio || 1, 1);
    return Math.min(3, Math.abs(dy) / (100 * dpr));
  }

  _applySmoothWheelZoom() {
    if (Math.abs(this._pendingWheelZoom) < WHEEL_ZOOM_EPS) return;
    const chunk = this._pendingWheelZoom * WHEEL_ZOOM_SMOOTH;
    this._pendingWheelZoom -= chunk;

    this._updateZoomSpeed();
    const scale = Math.pow(0.95, this.controls.zoomSpeed * Math.abs(chunk));

    const { THREE } = this.three;
    const offset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    const dist = offset.length();
    if (dist < 1e-6) return;

    let newDist = chunk < 0 ? dist * scale : dist / scale;
    newDist = THREE.MathUtils.clamp(
      newDist,
      this.controls.minDistance,
      this.controls.maxDistance
    );
    offset.multiplyScalar(newDist / dist);
    this.camera.position.copy(this.controls.target).add(offset);
    this._updateCameraClipping();
  }

  _updateCameraClipping() {
    const dist = this.camera.position.distanceTo(this.controls.target);
    const span = this._sceneSpan();
    this.camera.near = Math.max(0.05, Math.min(dist * 0.02, span * 0.05));
    this.camera.far = Math.max(dist * 12, span * 8, 1000);
    this.camera.updateProjectionMatrix();
  }

  // ---- coordinate helpers ----
  _v(x, y, z) {
    const { THREE } = this.three;
    return new THREE.Vector3(x, z, -y);
  }

  // ---- ground / grid ----
  _buildGround(scaleM) {
    const { THREE } = this.three;
    this._clearGroup(this.groundGroup);
    const span = Math.max(scaleM * 2, 2);
    const divisions = 20;
    const grid = new THREE.GridHelper(
      span,
      divisions,
      new THREE.Color(cssVar("--accent", "#0a84ff")),
      new THREE.Color(cssVar("--hairline", "#38383a"))
    );
    grid.material.opacity = 0.35;
    grid.material.transparent = true;
    this.groundGroup.add(grid);

    const planeGeo = new THREE.PlaneGeometry(span, span);
    planeGeo.rotateX(-Math.PI / 2);
    const plane = new THREE.Mesh(
      planeGeo,
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(cssVar("--surface-2", "#1c1c1e")),
        roughness: 0.95,
        metalness: 0.0,
        transparent: true,
        opacity: 0.25,
      })
    );
    plane.position.y = -0.02;
    this.groundGroup.add(plane);
  }

  setGround(scaleM) {
    if (Number.isFinite(scaleM) && scaleM > 0) this.scaleM = scaleM;
    if (this.extentM) return; // a map controls the scene extent
    this._buildGround(this.scaleM);
    this._frameCamera(this.scaleM);
  }

  _frameCamera(sizeM) {
    const d = Math.max(sizeM, 5) * 1.6;
    this.camera.position.set(d * 0.6, d * 0.7, d * 0.9);
    this.controls.target.set(0, 0, 0);
    this._applyControlLimits();
    this._updateCameraClipping();
  }

  // ---- map / terrain ----
  async setMap(mapInfo) {
    this._clearGroup(this.mapGroup);
    this._terrainMesh = null;
    if (!mapInfo || !mapInfo.heightmap_url || !mapInfo.heightmap || !mapInfo.extent_m) {
      this.extentM = null;
      this.groundGroup.visible = true;
      this._buildGround(this.scaleM);
      this._frameCamera(this.scaleM);
      return;
    }
    const hm = mapInfo.heightmap;
    const extent = mapInfo.extent_m;
    let heights;
    try {
      const resp = await fetch(mapInfo.heightmap_url);
      if (!resp.ok) throw new Error(`Heightmap fetch failed (${resp.status})`);
      heights = this._decodeHeights(await resp.arrayBuffer(), hm);
    } catch (_e) {
      heights = new Float32Array(hm.width * hm.height);
    }
    const mesh = this._buildTerrainMesh(heights, hm, extent, mapInfo.imagery_url);
    this.mapGroup.add(mesh);
    this._terrainMesh = mesh;
    this.extentM = { width: extent.width, height: extent.height };
    this.groundGroup.visible = false;
    this._frameCamera(Math.max(extent.width, extent.height));
  }

  clearMap() {
    this._clearGroup(this.mapGroup);
    this._terrainMesh = null;
    this.extentM = null;
    this.groundGroup.visible = true;
    this._buildGround(this.scaleM);
    this._frameCamera(this.scaleM);
  }

  _decodeHeights(buffer, hm) {
    const encoded = new Uint16Array(buffer);
    const heights = new Float32Array(encoded.length);
    for (let i = 0; i < encoded.length; i++) {
      heights[i] = encoded[i] * hm.height_scale + hm.height_offset;
    }
    return heights;
  }

  _buildTerrainMesh(heights, hm, extent, imageryUrl) {
    const { THREE } = this.three;
    const cols = hm.width;
    const rows = hm.height;
    const W = extent.width;
    const H = extent.height;
    const positions = new Float32Array(cols * rows * 3);
    const uvs = new Float32Array(cols * rows * 2);

    for (let r = 0; r < rows; r++) {
      const north = H / 2 - (r / (rows - 1)) * H; // row 0 = north edge
      for (let c = 0; c < cols; c++) {
        const east = -W / 2 + (c / (cols - 1)) * W;
        const idx = r * cols + c;
        positions[idx * 3] = east; // world x
        positions[idx * 3 + 1] = heights[idx]; // world y (up)
        positions[idx * 3 + 2] = -north; // world z
        uvs[idx * 2] = c / (cols - 1);
        uvs[idx * 2 + 1] = 1 - r / (rows - 1);
      }
    }

    const indices = [];
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const a = r * cols + c;
        const b = r * cols + c + 1;
        const d = (r + 1) * cols + c;
        const e = (r + 1) * cols + c + 1;
        indices.push(a, d, b, b, d, e);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.95,
      metalness: 0.0,
    });
    if (imageryUrl) {
      new THREE.TextureLoader().load(imageryUrl, (tex) => {
        if (tex.colorSpace !== undefined) tex.colorSpace = THREE.SRGBColorSpace;
        material.map = tex;
        material.needsUpdate = true;
      });
    }
    return new THREE.Mesh(geo, material);
  }

  // ---- entities ----
  renderEntities({ waypoints = [], markers = [], pending = null, waypointStyle = null } = {}) {
    this._clearGroup(this.entityGroup);
    waypoints.forEach((wp, i) => {
      const style = wp.style || waypointStyle || "sphere";
      this._addPoint(wp, {
        kind: "waypoint",
        index: wp.index != null ? wp.index : i,
        color: wp.selected ? SEL_COLOR : WP_COLOR,
        radius: wp.radius ?? (wp.selected ? 1.0 : 0.7),
        label: wp.label || `WP${i}`,
        style,
      });
    });
    markers.forEach((m, i) => {
      if (m.visible === false) return;
      this._addPoint(m, {
        kind: "marker",
        index: m.index != null ? m.index : i,
        color: m.selected ? SEL_COLOR : m.color || MARKER_COLOR,
        radius: m.selected ? 1.0 : 0.7,
        label: m.label || `Marker${i}`,
      });
    });
    if (pending) {
      this._addPoint(pending, {
        kind: "pending",
        index: null,
        color: pending.color || SEL_COLOR,
        radius: 0.9,
        label: `${pending.label || "Marker"} (pending)`,
      });
    }
  }

  _addPoint(pos, { kind, index, color, radius, label, style = "sphere" }) {
    const { THREE, CSS2DObject } = this.three;
    const scale = this._pointScale();
    const rScale = style === "dot" ? 0.25 : 1;
    const geo = new THREE.SphereGeometry(radius * scale * rScale, style === "dot" ? 8 : 16, style === "dot" ? 6 : 12);
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(this._v(pos.x, pos.y, pos.z));
    mesh.userData = { kind, index };
    this.entityGroup.add(mesh);

    if (label) {
      const el = document.createElement("div");
      el.className = "xyz-label";
      el.textContent = label;
      const labelObj = new CSS2DObject(el);
      labelObj.position.set(0, radius * scale + scale, 0);
      mesh.add(labelObj);
    }
  }

  _pointScale() {
    const span = this.extentM ? Math.max(this.extentM.width, this.extentM.height) : this.scaleM;
    return Math.max(span / 120, 0.25);
  }

  // ---- replay ----
  renderReplay({
    trajectory = null,
    reference = null,
    timeIndex = null,
    drone = null,
    attitude = null,
    velocity = null,
    acceleration = null,
    vehicleMarkerStyle = "sphere",
    markerAxes = "attitude",
    lowClearance = false,
    layers = null,
  } = {}) {
    this._clearGroup(this.replayGroup);

    if (reference && reference.length >= 2) {
      this._addLine(reference, REF_COLOR, 2);
    }

    if (layers && layers.length) {
      for (const layer of layers) {
        this._addTrajectoryLayer(layer.trajectory, layer.timeIndex, layer.color, layer.lowClearance);
        if (layer.drone) {
          this._addDrone(layer.drone, layer.color || WP_COLOR, {
            vehicleMarkerStyle: layer.vehicleMarkerStyle ?? vehicleMarkerStyle,
          });
        }
      }
      return;
    }

    if (trajectory && trajectory.length >= 2) {
      this._addTrajectoryLayer(trajectory, timeIndex, lowClearance ? LOW_COLOR : TRAJ_COLOR, lowClearance);
    }
    if (drone) {
      this._addDrone(drone, WP_COLOR, {
        vehicleMarkerStyle,
        markerAxes,
        attitude,
        velocity,
        acceleration,
      });
    }
  }

  _addTrajectoryLayer(trajectory, timeIndex, color, lowClearance = false) {
    if (!trajectory || trajectory.length < 2) return;
    const end = timeIndex != null ? Math.min(timeIndex + 1, trajectory.length) : trajectory.length;
    const slice = trajectory.slice(0, end);
    if (slice.length < 2) return;
    this._addLine(slice, lowClearance ? LOW_COLOR : color || TRAJ_COLOR, 3);
  }

  _addLine(points, color, _width) {
    const { THREE } = this.three;
    const verts = points.map((p) => this._v(p[0], p[1], p[2]));
    const geo = new THREE.BufferGeometry().setFromPoints(verts);
    const mat = new THREE.LineBasicMaterial({ color: new THREE.Color(color) });
    this.replayGroup.add(new THREE.Line(geo, mat));
  }

  _addDrone(
    drone,
    color,
    {
      vehicleMarkerStyle = "sphere",
      markerAxes = "attitude",
      attitude = null,
      velocity = null,
      acceleration = null,
    } = {}
  ) {
    const { THREE } = this.three;
    const scale = this._pointScale();
    const group = new THREE.Group();
    group.position.copy(this._v(drone[0], drone[1], drone[2]));

    const rScale = vehicleMarkerStyle === "dot" ? 0.35 : 1;
    const segs = vehicleMarkerStyle === "dot" ? 8 : 16;
    const geo = new THREE.SphereGeometry(replayDroneRadius() * scale * rScale, segs, vehicleMarkerStyle === "dot" ? 6 : 12);
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
    group.add(new THREE.Mesh(geo, mat));

    const headLen = Math.max(scale * 0.5, 0.15);
    const headWidth = Math.max(scale * 0.35, 0.1);
    const arrows = replayAxisArrows(markerAxes, { attitude, velocity, acceleration });
    for (const axis of arrows) {
      const [tx, ty, tz] = localToThree(axis.dir);
      const dir = new THREE.Vector3(tx, ty, tz).normalize();
      const origin = new THREE.Vector3(0, 0, 0);
      const arrow = new THREE.ArrowHelper(
        dir,
        origin,
        axis.length,
        new THREE.Color(axis.color).getHex(),
        headLen,
        headWidth
      );
      group.add(arrow);
    }

    this.replayGroup.add(group);
  }

  clearReplay() {
    this._clearGroup(this.replayGroup);
  }

  centerOn({ x = 0, y = 0, z = 0 } = {}) {
    const target = this._v(x, y, z);
    this.controls.target.copy(target);
    const span = this._sceneSpan();
    const d = Math.max(span, 5) * 0.4;
    this.camera.position.set(target.x + d * 0.6, target.y + d * 0.7, target.z + d * 0.9);
    this._applyControlLimits();
    this._updateCameraClipping();
  }

  // ---- input / picking ----
  _wireInput() {
    const el = this.renderer.domElement;
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        e.stopImmediatePropagation();
        if (!e.deltaY) return;
        this._pendingWheelZoom += Math.sign(e.deltaY) * this._normalizeWheelDelta(e);
      },
      { passive: false, capture: true }
    );
    let downPos = null;
    el.addEventListener("pointerdown", (e) => {
      downPos = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("pointerup", (e) => {
      if (!downPos) return;
      const moved = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);
      downPos = null;
      if (moved > 5) return; // treat as orbit/pan drag, not a click
      this._handleClick(e);
    });
  }

  _handleClick(e) {
    if (!this.onClick) return;
    const { THREE } = this.three;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(ndc, this.camera);

    const entityHits = this.raycaster.intersectObjects(this.entityGroup.children, false);
    for (const hit of entityHits) {
      const ud = hit.object.userData || {};
      if (ud.kind === "waypoint" || ud.kind === "marker") {
        this.onClick({ x: 0, y: 0, z: 0, entity_kind: ud.kind, entity_index: ud.index });
        return;
      }
    }

    const targets = this._terrainMesh ? [this._terrainMesh, this.pickPlane] : [this.pickPlane];
    const hits = this.raycaster.intersectObjects(targets, false);
    if (!hits.length) return;
    const p = hits[0].point;
    this.onClick({
      x: p.x, // east
      y: -p.z, // north
      z: p.y, // up
      entity_kind: "terrain",
      entity_index: null,
    });
  }

  // ---- lifecycle ----
  resize() {
    if (!this.renderer) return;
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
  }

  _clearGroup(group) {
    if (!group) return;
    for (let i = group.children.length - 1; i >= 0; i--) {
      const child = group.children[i];
      group.remove(child);
      this._disposeObject(child);
    }
  }

  _disposeObject(obj) {
    obj.traverse?.((node) => {
      if (node.geometry) node.geometry.dispose();
      if (node.material) {
        const mats = Array.isArray(node.material) ? node.material : [node.material];
        mats.forEach((m) => {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      }
      if (node.element && node.element.parentNode) {
        node.element.parentNode.removeChild(node.element);
      }
    });
  }

  dispose() {
    this._disposed = true;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this._clearGroup(this.entityGroup);
    this._clearGroup(this.replayGroup);
    this._clearGroup(this.mapGroup);
    this._clearGroup(this.groundGroup);
    if (this.controls) this.controls.dispose();
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }
    if (this.labelRenderer && this.labelRenderer.domElement.parentNode) {
      this.labelRenderer.domElement.parentNode.removeChild(this.labelRenderer.domElement);
    }
  }
}
