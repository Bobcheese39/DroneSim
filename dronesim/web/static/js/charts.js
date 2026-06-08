// uPlot-based line charts with an Apple-minimal look.
// uPlot is loaded globally from the CDN <script> in index.html.

import { loadThree } from "./map/loader.js";
import { localToThree } from "./map/attitude.js";

const PALETTE = ["#0a84ff", "#30d158", "#ff9f0a", "#ff453a", "#bf5af2", "#64d2ff"];

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return v ? v.trim() : fallback;
}

// series: [{ label, values: number[]|null[], color? }]
export function makeLineChart(container, { title, x, series, height = 220, yLabel = "" }) {
  container.innerHTML = "";
  if (typeof uPlot === "undefined") {
    container.innerHTML = '<div class="chart-error">uPlot failed to load.</div>';
    return null;
  }
  const grid = cssVar("--hairline", "#d2d2d7");
  const text = cssVar("--text-secondary", "#6e6e73");
  const data = [x, ...series.map((s) => s.values)];
  const opts = {
    title,
    width: container.clientWidth || 600,
    height,
    cursor: { y: false, points: { size: 6 } },
    legend: { live: false },
    scales: { x: { time: false } },
    axes: [
      {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        ticks: { stroke: grid, width: 1 },
        font: "11px -apple-system, system-ui, sans-serif",
        label: "time (s)",
        labelFont: "11px -apple-system, system-ui, sans-serif",
      },
      {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        ticks: { stroke: grid, width: 1 },
        font: "11px -apple-system, system-ui, sans-serif",
        label: yLabel,
        labelFont: "11px -apple-system, system-ui, sans-serif",
      },
    ],
    series: [
      {},
      ...series.map((s, i) => ({
        label: s.label,
        stroke: s.color || PALETTE[i % PALETTE.length],
        width: 1.75,
        points: { show: false },
      })),
    ],
  };
  const plot = new uPlot(opts, data, container);
  return plot;
}

function enuPointsToThree(points) {
  const out = [];
  for (const pt of points || []) {
    if (!pt || pt.length < 3) continue;
    const x = Number(pt[0]);
    const y = Number(pt[1]);
    const z = Number(pt[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    out.push(localToThree([x, y, z]));
  }
  return out;
}

function fitCameraToPoints(camera, controls, points, THREE) {
  if (!points.length) return;
  const box = new THREE.Box3();
  for (const [x, y, z] of points) {
    box.expandByPoint(new THREE.Vector3(x, y, z));
  }
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 1) * 0.75;
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.4, center.y + radius, center.z + radius * 1.4);
  camera.near = Math.max(radius / 200, 0.01);
  camera.far = Math.max(radius * 200, 1000);
  camera.updateProjectionMatrix();
  controls.update();
}

/** Interactive Three.js XYZ trajectory plot for the Analysis tab. */
export async function makeTrajectory3dChart(container, { title, trajectory, height = 420 }) {
  container.innerHTML = "";
  const host = document.createElement("div");
  host.className = "trajectory-3d-host";
  host.style.height = `${height}px`;
  container.appendChild(host);

  if (title) {
    const heading = document.createElement("div");
    heading.className = "chart-title";
    heading.textContent = title;
    container.insertBefore(heading, host);
  }

  const positionPts = enuPointsToThree(trajectory?.position_m);
  if (positionPts.length < 2) {
    host.innerHTML = '<div class="empty">Not enough trajectory data.</div>';
    return { destroy() {} };
  }

  let disposed = false;
  let rafId = null;
  const disposables = [];

  try {
    const { THREE, OrbitControls, CSS2DRenderer, CSS2DObject } = await loadThree();
    if (disposed) return { destroy() {} };

    const width = host.clientWidth || container.clientWidth || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    host.appendChild(renderer.domElement);
    disposables.push(renderer);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(cssVar("--bg", "#0b0b0d"));

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1_000_000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    disposables.push(controls);

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(120, 200, 80);
    scene.add(sun);

    const gridSize = 200;
    const grid = new THREE.GridHelper(
      gridSize,
      20,
      new THREE.Color(cssVar("--accent", "#0a84ff")),
      new THREE.Color(cssVar("--hairline", "#38383a"))
    );
    scene.add(grid);

    const allPoints = [...positionPts];
    const refPts = enuPointsToThree(trajectory?.reference_position_m);
    if (refPts.length >= 2) {
      allPoints.push(...refPts);
      const refGeo = new THREE.BufferGeometry().setFromPoints(
        refPts.map(([x, y, z]) => new THREE.Vector3(x, y, z))
      );
      const refMat = new THREE.LineBasicMaterial({ color: new THREE.Color("#6e6e73") });
      scene.add(new THREE.Line(refGeo, refMat));
    }

    const trajGeo = new THREE.BufferGeometry().setFromPoints(
      positionPts.map(([x, y, z]) => new THREE.Vector3(x, y, z))
    );
    const trajMat = new THREE.LineBasicMaterial({ color: new THREE.Color("#30d158") });
    scene.add(new THREE.Line(trajGeo, trajMat));

    const wpPts = enuPointsToThree(trajectory?.waypoints_m);
    for (let i = 0; i < wpPts.length; i++) {
      const [x, y, z] = wpPts[i];
      allPoints.push([x, y, z]);
      const geo = new THREE.SphereGeometry(1.2, 12, 10);
      const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color("#0a84ff") });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, y, z);
      scene.add(mesh);

      const labelEl = document.createElement("div");
      labelEl.className = "trajectory-wp-label";
      labelEl.textContent = `WP${i}`;
      const label = new CSS2DObject(labelEl);
      label.position.set(0, 2.5, 0);
      mesh.add(label);
    }
    allPoints.push(...wpPts);

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.inset = "0";
    labelRenderer.domElement.style.pointerEvents = "none";
    host.appendChild(labelRenderer.domElement);
    disposables.push(labelRenderer);

    fitCameraToPoints(camera, controls, allPoints, THREE);

    const onResize = () => {
      const w = host.clientWidth || width;
      const h = height;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      labelRenderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);
    disposables.push({ dispose: () => window.removeEventListener("resize", onResize) });

    const animate = () => {
      if (disposed) return;
      rafId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    return {
      destroy() {
        disposed = true;
        if (rafId != null) cancelAnimationFrame(rafId);
        for (const item of disposables) {
          if (item?.dispose) item.dispose();
          else if (item?.domElement?.parentNode) item.domElement.parentNode.removeChild(item.domElement);
        }
        container.innerHTML = "";
      },
    };
  } catch (err) {
    host.innerHTML = `<div class="chart-error">3D chart failed: ${err.message}</div>`;
    return { destroy() {} };
  }
}

/** Bar chart for miss-distance histogram bins. bins: edge array, counts: bar heights. */
export function makeHistogramChart(container, { title, bins, counts, height = 220, yLabel = "Trials" }) {
  container.innerHTML = "";
  if (typeof uPlot === "undefined") {
    container.innerHTML = '<div class="chart-error">uPlot failed to load.</div>';
    return null;
  }
  if (!bins || bins.length < 2 || !counts || !counts.length) {
    container.innerHTML = '<div class="empty">No histogram data.</div>';
    return null;
  }
  const grid = cssVar("--hairline", "#d2d2d7");
  const text = cssVar("--text-secondary", "#6e6e73");
  const accent = cssVar("--accent", "#0a84ff");
  const centers = [];
  const widths = [];
  for (let i = 0; i < counts.length; i++) {
    centers.push((bins[i] + bins[i + 1]) / 2);
    widths.push(bins[i + 1] - bins[i]);
  }
  const opts = {
    title,
    width: container.clientWidth || 600,
    height,
    cursor: { y: false },
    legend: { show: false },
    scales: { x: { time: false } },
    axes: [
      {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        font: "11px -apple-system, system-ui, sans-serif",
        label: "Miss distance (m)",
        labelFont: "11px -apple-system, system-ui, sans-serif",
      },
      {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        font: "11px -apple-system, system-ui, sans-serif",
        label: yLabel,
        labelFont: "11px -apple-system, system-ui, sans-serif",
      },
    ],
    series: [
      {},
      {
        paths: uPlot.paths.bars({ size: [0.6, 100] }),
        fill: accent,
        stroke: accent,
      },
    ],
  };
  return new uPlot(opts, [centers, counts], container);
}

/**
 * Envelope chart: faint trial lines + mean + shaded ±1σ band.
 * env: { time_s, mean, std, trials: number[][] }
 */
export function makeEnvelopeChart(container, { title, env, height = 220, yLabel = "" }) {
  container.innerHTML = "";
  if (typeof uPlot === "undefined") {
    container.innerHTML = '<div class="chart-error">uPlot failed to load.</div>';
    return null;
  }
  const { time_s: x, mean, std, trials } = env;
  if (!x || !x.length || !mean || !mean.length) {
    container.innerHTML = '<div class="empty">No envelope data.</div>';
    return null;
  }
  const grid = cssVar("--hairline", "#d2d2d7");
  const text = cssVar("--text-secondary", "#6e6e73");
  const upper = mean.map((m, i) => m + (std[i] ?? 0));
  const lower = mean.map((m, i) => m - (std[i] ?? 0));
  const trialData = trials || [];
  const nTrials = trialData.length;
  const upperIdx = 1 + nTrials;
  const lowerIdx = 2 + nTrials;
  const data = [x, ...trialData, upper, lower, mean];
  const series = [
    {},
    ...trialData.map((_, i) => ({
      label: `trial ${i}`,
      stroke: cssVar("--text-secondary", "#6e6e73"),
      width: 1,
      points: { show: false },
      alpha: 0.2,
    })),
    { label: "+1\u03c3", stroke: "transparent", points: { show: false } },
    {
      label: "\u22121\u03c3",
      stroke: "transparent",
      fill: cssVar("--accent", "#0a84ff"),
      fillTo: upperIdx,
      points: { show: false },
    },
    {
      label: "mean",
      stroke: cssVar("--amber", "#ff9f0a"),
      width: 2.5,
      points: { show: false },
    },
  ];
  const opts = {
    title,
    width: container.clientWidth || 600,
    height,
    cursor: { y: false, points: { size: 6 } },
    legend: { live: false },
    scales: { x: { time: false } },
    axes: [
      {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        font: "11px -apple-system, system-ui, sans-serif",
        label: "time (s)",
        labelFont: "11px -apple-system, system-ui, sans-serif",
      },
      {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        font: "11px -apple-system, system-ui, sans-serif",
        label: yLabel,
        labelFont: "11px -apple-system, system-ui, sans-serif",
      },
    ],
    series,
    bands: [[upperIdx, lowerIdx]],
  };
  return new uPlot(opts, data, container);
}

export function tableHtml(rows, columns) {
  if (!rows || !rows.length) return '<div class="empty">No data.</div>';
  const head = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows
    .map(
      (r) =>
        `<tr>${columns
          .map((c) => `<td>${formatCell(r[c.key])}</td>`)
          .join("")}</tr>`
    )
    .join("");
  return `<table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function formatCell(v) {
  if (v === null || v === undefined) return "\u2014";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}
