// Lazy loaders for the two 3D engine libraries.
//
// Three.js is pulled in as ES modules (resolved via the <script type="importmap">
// in index.html) the first time the XYZ engine is built. Cesium is a classic
// UMD global injected on demand so it is only initialized when the user toggles
// to LLA mode.

let _threePromise = null;
let _cesiumPromise = null;

export function loadThree() {
  if (_threePromise) return _threePromise;
  _threePromise = (async () => {
    const THREE = await import("three");
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    const { CSS2DRenderer, CSS2DObject } = await import(
      "three/addons/renderers/CSS2DRenderer.js"
    );
    return { THREE, OrbitControls, CSS2DRenderer, CSS2DObject };
  })();
  return _threePromise;
}

function injectScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-lazy-src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "true") resolve();
      else {
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)));
      }
      return;
    }
    const el = document.createElement("script");
    el.src = src;
    el.async = true;
    el.dataset.lazySrc = src;
    el.addEventListener("load", () => {
      el.dataset.loaded = "true";
      resolve();
    });
    el.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)));
    document.head.appendChild(el);
  });
}

function injectStylesheet(href) {
  if (document.querySelector(`link[data-lazy-href="${href}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  link.dataset.lazyHref = href;
  document.head.appendChild(link);
}

const CESIUM_VERSION = "1.118";
const CESIUM_BASE = `https://cesium.com/downloads/cesiumjs/releases/${CESIUM_VERSION}/Build/Cesium`;

export function loadCesium() {
  if (_cesiumPromise) return _cesiumPromise;
  _cesiumPromise = (async () => {
    injectStylesheet(`${CESIUM_BASE}/Widgets/widgets.css`);
    await injectScript(`${CESIUM_BASE}/Cesium.js`);
    if (typeof window.Cesium === "undefined") {
      throw new Error("Cesium failed to load (network or CDN blocked).");
    }
    return window.Cesium;
  })();
  return _cesiumPromise;
}
