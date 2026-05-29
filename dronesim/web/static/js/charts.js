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
