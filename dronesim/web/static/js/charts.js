// uPlot-based line charts with an Apple-minimal look.
// uPlot is loaded globally from the CDN <script> in index.html.

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
