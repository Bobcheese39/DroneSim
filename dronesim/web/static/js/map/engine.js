// Engine manager: owns the active 3D scene and swaps between the custom XYZ
// (Three.js) engine and the Cesium LLA engine. Both scenes implement the same
// local-meters interface, so the rest of the app talks to MapEngine without
// caring which engine is active.

import { loadThree, loadCesium } from "./loader.js";

export class MapEngine {
  constructor(containerId, { onClick, onModeChange } = {}) {
    this.containerId = containerId;
    this.onClick = onClick;
    this.onModeChange = onModeChange;
    this.scene = null;
    this.mode = null;
    this._switching = false;
  }

  get ready() {
    return !!this.scene;
  }

  async setMode(mode) {
    if (mode === this.mode || this._switching) return;
    this._switching = true;
    try {
      if (this.scene) {
        try {
          this.scene.dispose();
        } catch (_e) {
          /* ignore */
        }
        this.scene = null;
      }
      const container = document.getElementById(this.containerId);
      if (container) container.innerHTML = "";

      if (mode === "lla") {
        await loadCesium();
        const { CesiumScene } = await import("./cesium.js");
        this.scene = new CesiumScene(this.containerId, { onClick: this.onClick });
      } else {
        const three = await loadThree();
        const { CustomScene } = await import("./xyz.js");
        this.scene = new CustomScene(this.containerId, { onClick: this.onClick, three });
      }
      this.mode = mode;
      if (this.onModeChange) await this.onModeChange(mode);
    } finally {
      this._switching = false;
    }
  }

  async setMap(mapInfo) {
    if (this.scene && this.scene.setMap) return this.scene.setMap(mapInfo);
  }

  clearMap() {
    if (this.scene && this.scene.clearMap) this.scene.clearMap();
  }

  setGround(scaleM) {
    if (this.scene && this.scene.setGround) this.scene.setGround(scaleM);
  }

  renderEntities(payload) {
    if (this.scene) this.scene.renderEntities(payload);
  }

  renderReplay(payload) {
    if (this.scene) this.scene.renderReplay(payload);
  }

  clearReplay() {
    if (this.scene) this.scene.clearReplay();
  }

  centerOn(pos) {
    if (this.scene && this.scene.centerOn) this.scene.centerOn(pos);
  }

  resize() {
    if (this.scene && this.scene.resize) this.scene.resize();
  }
}
