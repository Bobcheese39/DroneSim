// Native CesiumJS scene wrapper.
//
// Ported from the old Panel ReactiveHTML bridge (dronesim/gui/viewers/
// cesium_viewer.py) but freed of the param.String shuttling, pending-payload
// queueing, and shadow-DOM root resolution. Cesium is loaded globally from the
// CDN <script> in index.html.

export class CesiumScene {
  constructor(containerId, { onClick } = {}) {
    this.container = document.getElementById(containerId);
    this.onClick = onClick;
    this.viewer = null;
    this.satelliteLayer = null;
    this.entityRegistry = new Map();
    this.trajectoryEntity = null;
    this.referenceEntity = null;
    this.droneEntity = null;
    this.trajectoryEntities = [];
    this.droneEntities = [];
    this.pendingEntity = null;
    this.currentBounds = null;
    this._terrainData = null;
    this._init();
  }

  _init() {
    if (typeof Cesium === "undefined") {
      this.container.innerHTML =
        '<div class="cesium-error">CesiumJS failed to load (network or CDN blocked).</div>';
      return;
    }
    if (typeof Cesium.Ion !== "undefined") {
      Cesium.Ion.defaultAccessToken = undefined; // avoid Ion network calls
    }
    const viewer = new Cesium.Viewer(this.container, {
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      selectionIndicator: false,
      infoBox: false,
      baseLayer: new Cesium.ImageryLayer(
        new Cesium.UrlTemplateImageryProvider({
          url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          tilingScheme: new Cesium.WebMercatorTilingScheme(),
          maximumLevel: 19,
          credit: "Map data \u00a9 OpenStreetMap contributors",
        })
      ),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    });
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewer.scene.skyAtmosphere.show = true;
    this.viewer = viewer;

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((click) => this._handleClick(click), Cesium.ScreenSpaceEventType.LEFT_CLICK);
    this._handler = handler;
  }

  _handleClick(click) {
    if (!this.viewer || !this.onClick) return;
    const scene = this.viewer.scene;
    const picked = scene.pick(click.position);
    let entityKind = "terrain";
    let entityIndex = null;
    if (Cesium.defined(picked) && picked.id && picked.id.dronesim_kind) {
      entityKind = picked.id.dronesim_kind;
      entityIndex = picked.id.dronesim_index;
    }
    let cartesian = null;
    try {
      cartesian = scene.pickPosition(click.position);
    } catch (_e) {
      cartesian = null;
    }
    if (!Cesium.defined(cartesian)) {
      const ray = this.viewer.camera.getPickRay(click.position);
      if (ray) cartesian = scene.globe.pick(ray, scene);
    }
    if (!Cesium.defined(cartesian)) return;
    const carto = Cesium.Cartographic.fromCartesian(cartesian);
    this.onClick({
      lat: Cesium.Math.toDegrees(carto.latitude),
      lon: Cesium.Math.toDegrees(carto.longitude),
      height: carto.height,
      entity_kind: entityKind,
      entity_index: entityIndex,
    });
  }

  setImagery(mapInfo) {
    if (!this.viewer || !mapInfo || !mapInfo.bounds) return;
    const b = mapInfo.bounds;
    const rect = Cesium.Rectangle.fromDegrees(b.west, b.south, b.east, b.north);
    if (this.satelliteLayer) {
      this.viewer.imageryLayers.remove(this.satelliteLayer, true);
      this.satelliteLayer = null;
    }
    if (mapInfo.imagery_url) {
      const layer = Cesium.ImageryLayer.fromProviderAsync(
        Cesium.SingleTileImageryProvider.fromUrl(mapInfo.imagery_url, { rectangle: rect })
      );
      this.viewer.imageryLayers.add(layer);
      this.satelliteLayer = layer;
      this.satelliteLayer.alpha = 0.95;
    }
    const key = `${b.west},${b.south},${b.east},${b.north}`;
    if (this.currentBounds !== key) {
      this.viewer.camera.flyTo({
        destination: rect,
        duration: 0.0,
        orientation: { pitch: Cesium.Math.toRadians(-45), heading: 0 },
      });
      this.currentBounds = key;
    }
  }

  _rectanglesIntersect(a, b) {
    return !(a.east <= b.west || a.west >= b.east || a.north <= b.south || a.south >= b.north);
  }

  _decodeHeightmapBuffer(buffer, hm) {
    const encoded = new Uint16Array(buffer);
    const heights = new Float32Array(encoded.length);
    for (let i = 0; i < encoded.length; i++) {
      heights[i] = encoded[i] * hm.height_scale + hm.height_offset;
    }
    return heights;
  }

  _sampleTerrainHeight(lonDeg, latDeg) {
    const td = this._terrainData;
    if (!td) return 0;
    const { west, south, east, north } = td.bounds;
    const lon = Math.max(west, Math.min(east, lonDeg));
    const lat = Math.max(south, Math.min(north, latDeg));
    const cols = td.width;
    const rows = td.height;
    const colF = ((lon - west) / (east - west)) * (cols - 1);
    const rowF = ((north - lat) / (north - south)) * (rows - 1);
    const col0 = Math.floor(colF);
    const row0 = Math.floor(rowF);
    const col1 = Math.min(col0 + 1, cols - 1);
    const row1 = Math.min(row0 + 1, rows - 1);
    const tx = colF - col0;
    const ty = rowF - row0;
    const h00 = td.heights[row0 * cols + col0];
    const h10 = td.heights[row0 * cols + col1];
    const h01 = td.heights[row1 * cols + col0];
    const h11 = td.heights[row1 * cols + col1];
    const h0 = h00 * (1 - tx) + h10 * tx;
    const h1 = h01 * (1 - tx) + h11 * tx;
    return h0 * (1 - ty) + h1 * ty;
  }

  async setTerrain(mapInfo) {
    if (!this.viewer) return;
    const globe = this.viewer.scene.globe;
    if (!mapInfo || !mapInfo.heightmap_url || !mapInfo.heightmap || !mapInfo.bounds) {
      this._terrainData = null;
      this.viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
      globe.depthTestAgainstTerrain = false;
      return;
    }
    const resp = await fetch(mapInfo.heightmap_url);
    if (!resp.ok) {
      throw new Error(`Heightmap fetch failed (${resp.status})`);
    }
    const hm = mapInfo.heightmap;
    const b = mapInfo.bounds;
    const heights = this._decodeHeightmapBuffer(await resp.arrayBuffer(), hm);
    this._terrainData = {
      heights,
      width: hm.width,
      height: hm.height,
      bounds: b,
    };

    const tileW = 64;
    const tileH = 64;
    const mapRect = Cesium.Rectangle.fromDegrees(b.west, b.south, b.east, b.north);
    const tilingScheme = new Cesium.GeographicTilingScheme();
    const sampleHeight = (lonDeg, latDeg) => this._sampleTerrainHeight(lonDeg, latDeg);

    this.viewer.terrainProvider = new Cesium.CustomHeightmapTerrainProvider({
      width: tileW,
      height: tileH,
      tilingScheme,
      callback: (x, y, level) => {
        const rect = tilingScheme.tileXYToRectangle(x, y, level);
        if (!this._rectanglesIntersect(rect, mapRect)) return undefined;

        const out = new Float32Array(tileW * tileH);
        for (let row = 0; row < tileH; row++) {
          const v = (row + 0.5) / tileH;
          const lat = Cesium.Math.toDegrees(Cesium.Math.lerp(rect.north, rect.south, v));
          for (let col = 0; col < tileW; col++) {
            const u = (col + 0.5) / tileW;
            const lon = Cesium.Math.toDegrees(Cesium.Math.lerp(rect.west, rect.east, u));
            out[row * tileW + col] = sampleHeight(lon, lat);
          }
        }
        return out;
      },
    });
    globe.depthTestAgainstTerrain = true;
  }

  async setMap(mapInfo) {
    await this.setTerrain(mapInfo);
    this.setImagery(mapInfo);
  }

  renderEntities({ waypoints = [], markers = [], pending = null } = {}) {
    if (!this.viewer) return;
    this.entityRegistry.forEach((ent) => this.viewer.entities.remove(ent));
    this.entityRegistry.clear();

    waypoints.forEach((wp, i) => {
      const ent = this.viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(wp.lon, wp.lat, wp.alt),
        point: {
          pixelSize: wp.selected ? 18 : 12,
          color: Cesium.Color.fromCssColorString(wp.selected ? "#ffd60a" : "#0a84ff"),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: this._labelGraphics(wp.label || `WP${i}`),
      });
      ent.dronesim_kind = "waypoint";
      ent.dronesim_index = wp.index != null ? wp.index : i;
      this.entityRegistry.set(`wp_${i}`, ent);
    });

    markers.forEach((m, i) => {
      if (m.visible === false) return;
      const ent = this.viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(m.lon, m.lat, m.alt),
        point: {
          pixelSize: (m.size || 10) + (m.selected ? 8 : 0),
          color: Cesium.Color.fromCssColorString(m.selected ? "#ffd60a" : m.color || "#ff453a"),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: this._labelGraphics(m.label || `Marker${i}`),
      });
      ent.dronesim_kind = "marker";
      ent.dronesim_index = m.index != null ? m.index : i;
      this.entityRegistry.set(`m_${i}`, ent);
    });

    if (this.pendingEntity) {
      this.viewer.entities.remove(this.pendingEntity);
      this.pendingEntity = null;
    }
    if (pending) {
      this.pendingEntity = this.viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(pending.lon, pending.lat, pending.alt),
        point: {
          pixelSize: 16,
          color: Cesium.Color.fromCssColorString(pending.color || "#ffd60a"),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: this._labelGraphics(`${pending.label || "Marker"} (pending)`),
      });
    }
  }

  renderReplay({
    trajectory = null,
    reference = null,
    timeIndex = null,
    drone = null,
    lowClearance = false,
    layers = null,
  } = {}) {
    if (!this.viewer) return;
    this._clearReplayEntities();

    if (reference && reference.length >= 2) {
      const positions = [];
      reference.forEach((p) => positions.push(p[1], p[0], p[2]));
      this.referenceEntity = this.viewer.entities.add({
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(positions),
          width: 3,
          material: Cesium.Color.fromCssColorString("#ff9f0a"),
        },
      });
    }

    if (layers && layers.length) {
      for (const layer of layers) {
        this._addTrajectoryLayer(layer.trajectory, layer.timeIndex, layer.color, layer.lowClearance);
        this._addDroneLayer(layer.drone, layer.color);
      }
      return;
    }

    if (trajectory && trajectory.length >= 2) {
      this._addTrajectoryLayer(trajectory, timeIndex, lowClearance ? "#ff453a" : "#30d158", lowClearance);
    }
    if (drone) {
      this._addDroneLayer(drone, "#0a84ff");
    }
  }

  _clearReplayEntities() {
    [this.trajectoryEntity, this.referenceEntity, this.droneEntity].forEach((ent) => {
      if (ent) this.viewer.entities.remove(ent);
    });
    this.trajectoryEntity = this.referenceEntity = this.droneEntity = null;
    for (const ent of this.trajectoryEntities) {
      if (ent) this.viewer.entities.remove(ent);
    }
    for (const ent of this.droneEntities) {
      if (ent) this.viewer.entities.remove(ent);
    }
    this.trajectoryEntities = [];
    this.droneEntities = [];
  }

  _addTrajectoryLayer(trajectory, timeIndex, color, lowClearance = false) {
    if (!trajectory || trajectory.length < 2) return;
    const end = timeIndex != null ? Math.min(timeIndex + 1, trajectory.length) : trajectory.length;
    const positions = [];
    for (let i = 0; i < end; i++) positions.push(trajectory[i][1], trajectory[i][0], trajectory[i][2]);
    const lineColor = lowClearance ? "#ff453a" : color || "#30d158";
    if (positions.length >= 6) {
      const ent = this.viewer.entities.add({
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(positions),
          width: 4,
          material: Cesium.Color.fromCssColorString(lineColor),
        },
      });
      this.trajectoryEntities.push(ent);
    }
  }

  _addDroneLayer(drone, color) {
    if (!drone) return;
    const ent = this.viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(drone[1], drone[0], drone[2]),
      point: {
        pixelSize: 14,
        color: Cesium.Color.fromCssColorString(color || "#0a84ff"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    this.droneEntities.push(ent);
  }

  clearReplay() {
    this.renderReplay({});
  }

  centerOn({ lat, lon, alt = 0 }) {
    if (!this.viewer) return;
    const position = Cesium.Cartesian3.fromDegrees(lon, lat, alt);
    const sphere = new Cesium.BoundingSphere(position, 50);
    this.viewer.camera.flyToBoundingSphere(sphere, {
      duration: 0.6,
      offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-45), 300),
    });
  }

  _labelGraphics(text) {
    return {
      text,
      font: "12px -apple-system, system-ui, sans-serif",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      pixelOffset: new Cesium.Cartesian2(0, -18),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    };
  }

  resize() {
    if (this.viewer) {
      try {
        this.viewer.resize();
      } catch (_e) {
        /* noop */
      }
    }
  }
}
