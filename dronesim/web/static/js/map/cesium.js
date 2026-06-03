// Native CesiumJS scene wrapper.
//
// Implements the same local-meters interface as the custom XYZ engine: callers
// pass positions in local meters { x, y, z } and this adapter converts to/from
// WGS84 lat/lon internally using the active map center. Cesium is loaded lazily
// (see map/loader.js) and is only initialized when LLA mode is selected.

import { localToLatLon, latLonToLocal, replayDronePixelSize, replayDroneRadius } from "../state.js";
import { replayAxisArrows } from "./attitude.js";

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
    this.center = { lat: 0, lon: 0 };
    this._init();
  }

  // ---- local-meters <-> geographic helpers ----
  _carto(x, y, z = 0) {
    const [lat, lon] = localToLatLon(x, y, this.center.lat, this.center.lon);
    return Cesium.Cartesian3.fromDegrees(lon, lat, z);
  }

  _degreesArray(points) {
    // points: array of [x, y, z] meters -> flat [lon, lat, height, ...]
    const out = [];
    for (const p of points) {
      const [lat, lon] = localToLatLon(p[0], p[1], this.center.lat, this.center.lon);
      out.push(lon, lat, p.length > 2 ? p[2] : 0);
    }
    return out;
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
    const lat = Cesium.Math.toDegrees(carto.latitude);
    const lon = Cesium.Math.toDegrees(carto.longitude);
    const [x, y] = latLonToLocal(lat, lon, this.center.lat, this.center.lon);
    this.onClick({
      x,
      y,
      z: carto.height,
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
    if (mapInfo && mapInfo.center_lat != null && mapInfo.center_lon != null) {
      this.center = { lat: mapInfo.center_lat, lon: mapInfo.center_lon };
    }
    await this.setTerrain(mapInfo);
    this.setImagery(mapInfo);
  }

  clearMap() {
    if (!this.viewer) return;
    if (this.satelliteLayer) {
      this.viewer.imageryLayers.remove(this.satelliteLayer, true);
      this.satelliteLayer = null;
    }
    this._terrainData = null;
    this.viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
    this.viewer.scene.globe.depthTestAgainstTerrain = false;
  }

  renderEntities({ waypoints = [], markers = [], pending = null, waypointStyle = null } = {}) {
    if (!this.viewer) return;
    this.entityRegistry.forEach((ent) => this.viewer.entities.remove(ent));
    this.entityRegistry.clear();

    waypoints.forEach((wp, i) => {
      const style = wp.style || waypointStyle || "dot";
      const color = wp.selected ? "#ffd60a" : "#0a84ff";
      const pos = this._carto(wp.x, wp.y, wp.z);
      const label = this._labelGraphics(wp.label || `WP${i}`);
      let ent;
      if (style === "sphere") {
        const r = Math.max(wp.radius ?? 0.7, 0.3);
        ent = this.viewer.entities.add({
          position: pos,
          ellipsoid: {
            radii: new Cesium.Cartesian3(r, r, r),
            material: Cesium.Color.fromCssColorString(color),
            outline: true,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 1,
          },
          label,
        });
      } else {
        ent = this.viewer.entities.add({
          position: pos,
          point: {
            pixelSize: wp.pixelSize ?? (wp.selected ? 18 : 12),
            color: Cesium.Color.fromCssColorString(color),
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label,
        });
      }
      ent.dronesim_kind = "waypoint";
      ent.dronesim_index = wp.index != null ? wp.index : i;
      this.entityRegistry.set(`wp_${i}`, ent);
    });

    markers.forEach((m, i) => {
      if (m.visible === false) return;
      const ent = this.viewer.entities.add({
        position: this._carto(m.x, m.y, m.z),
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
        position: this._carto(pending.x, pending.y, pending.z),
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
    attitude = null,
    velocity = null,
    acceleration = null,
    vehicleMarkerStyle = "dot",
    markerAxes = "attitude",
    lowClearance = false,
    layers = null,
  } = {}) {
    if (!this.viewer) return;
    this._clearReplayEntities();

    if (reference && reference.length >= 2) {
      this.referenceEntity = this.viewer.entities.add({
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(this._degreesArray(reference)),
          width: 3,
          material: Cesium.Color.fromCssColorString("#ff9f0a"),
        },
      });
    }

    if (layers && layers.length) {
      for (const layer of layers) {
        this._addTrajectoryLayer(layer.trajectory, layer.timeIndex, layer.color, layer.lowClearance);
        this._addDroneLayer(layer.drone, layer.color, {
          vehicleMarkerStyle: layer.vehicleMarkerStyle ?? vehicleMarkerStyle,
        });
      }
      return;
    }

    if (trajectory && trajectory.length >= 2) {
      this._addTrajectoryLayer(trajectory, timeIndex, lowClearance ? "#ff453a" : "#30d158", lowClearance);
    }
    if (drone) {
      this._addDroneLayer(drone, "#0a84ff", {
        vehicleMarkerStyle,
        markerAxes,
        attitude,
        velocity,
        acceleration,
      });
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
    const positions = this._degreesArray(trajectory.slice(0, end));
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

  _addDroneLayer(
    drone,
    color,
    {
      vehicleMarkerStyle = "dot",
      markerAxes = "attitude",
      attitude = null,
      velocity = null,
      acceleration = null,
    } = {}
  ) {
    if (!drone) return;
    const pos = this._carto(drone[0], drone[1], drone[2]);
    const css = color || "#0a84ff";
    let ent;
    if (vehicleMarkerStyle === "sphere") {
      const r = Math.max(replayDroneRadius(), 0.5);
      ent = this.viewer.entities.add({
        position: pos,
        ellipsoid: {
          radii: new Cesium.Cartesian3(r, r, r),
          material: Cesium.Color.fromCssColorString(css),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 1,
        },
      });
    } else {
      ent = this.viewer.entities.add({
        position: pos,
        point: {
          pixelSize: replayDronePixelSize(),
          color: Cesium.Color.fromCssColorString(css),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }
    this.droneEntities.push(ent);

    const px = drone[0];
    const py = drone[1];
    const pz = drone.length > 2 ? drone[2] : 0;
    const arrows = replayAxisArrows(markerAxes, { attitude, velocity, acceleration });
    for (const axis of arrows) {
      const [dx, dy, dz] = axis.dir;
      const len = axis.length;
      const end = this._carto(px + dx * len, py + dy * len, pz + dz * len);
      const arrowEnt = this.viewer.entities.add({
        polyline: {
          positions: [this._carto(px, py, pz), end],
          width: 3,
          material: Cesium.Color.fromCssColorString(axis.color),
        },
      });
      this.droneEntities.push(arrowEnt);
    }
  }

  clearReplay() {
    this.renderReplay({});
  }

  centerOn({ x = 0, y = 0, z = 0 } = {}) {
    if (!this.viewer) return;
    const position = this._carto(x, y, z);
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

  dispose() {
    if (this._handler) {
      try {
        this._handler.destroy();
      } catch (_e) {
        /* noop */
      }
      this._handler = null;
    }
    if (this.viewer) {
      try {
        this.viewer.destroy();
      } catch (_e) {
        /* noop */
      }
      this.viewer = null;
    }
  }
}
