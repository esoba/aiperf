import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { MetricSelector } from '../components/metric-selector.js';
import { ChartWrapper } from '../components/chart-wrapper.js';

const CHART_COLORS = [
  palette.mauve,
  palette.blue,
  palette.green,
  palette.peach,
  palette.pink,
  palette.teal,
  palette.sapphire,
  palette.yellow,
  palette.flamingo,
  palette.lavender,
];

function formatDate(iso) {
  if (!iso) return '---';
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric', year: '2-digit' });
}

function formatValue(value, unit) {
  if (value == null) return '---';
  const formatted = typeof value === 'number' ? value.toFixed(2) : value;
  return unit ? `${formatted} ${unit}` : String(formatted);
}

// Feature 7: Percentile heatmap cell
// For latency metrics: lower = better (green), higher = worse (red)
// For throughput metrics: higher = better (green), lower = worse (red)
function heatmapColor(value, allValues, isLowerBetter) {
  if (value == null || allValues.length === 0) return palette.surface0;
  const min = Math.min(...allValues.filter(v => v != null));
  const max = Math.max(...allValues.filter(v => v != null));
  if (min === max) return palette.surface0;

  // Normalize to 0-1 range
  let normalized = (value - min) / (max - min);
  if (isLowerBetter) normalized = 1 - normalized;

  // Interpolate: green (good) -> yellow (mid) -> red (bad)
  if (normalized >= 0.5) {
    // Good half: blend green and yellow
    const t = (normalized - 0.5) * 2;
    return blendColors(palette.yellow, palette.green, t);
  } else {
    // Bad half: blend red and yellow
    const t = normalized * 2;
    return blendColors(palette.red, palette.yellow, t);
  }
}

function blendColors(c1, c2, t) {
  const r1 = parseInt(c1.slice(1, 3), 16);
  const g1 = parseInt(c1.slice(3, 5), 16);
  const b1 = parseInt(c1.slice(5, 7), 16);
  const r2 = parseInt(c2.slice(1, 3), 16);
  const g2 = parseInt(c2.slice(3, 5), 16);
  const b2 = parseInt(c2.slice(5, 7), 16);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

function PercentileHeatmap({ entry, allEntries, metric }) {
  const pVals = entry.percentiles ?? {};
  const p50 = pVals.p50 ?? null;
  const p90 = pVals.p90 ?? null;
  const p99 = pVals.p99 ?? null;

  // If no percentile data, skip
  if (p50 == null && p90 == null && p99 == null) return null;

  // Determine if lower is better based on metric name
  const isLowerBetter = !metric.includes('throughput');

  // Collect all values across entries for normalization
  const allP50 = allEntries.map(e => e.percentiles?.p50).filter(v => v != null);
  const allP90 = allEntries.map(e => e.percentiles?.p90).filter(v => v != null);
  const allP99 = allEntries.map(e => e.percentiles?.p99).filter(v => v != null);

  const cells = [
    { label: 'p50', value: p50, allValues: allP50 },
    { label: 'p90', value: p90, allValues: allP90 },
    { label: 'p99', value: p99, allValues: allP99 },
  ];

  return html`
    <div style="display: flex; gap: 2px; align-items: center">
      ${cells.map(cell => {
        if (cell.value == null) {
          return html`
            <div
              key=${cell.label}
              title="${cell.label}: ---"
              style=${'width: 24px; height: 18px; border-radius: 2px; background: ' + palette.surface0}
            />
          `;
        }
        const bg = heatmapColor(cell.value, cell.allValues, isLowerBetter);
        return html`
          <div
            key=${cell.label}
            title="${cell.label}: ${cell.value.toFixed(1)}"
            style=${'width: 24px; height: 18px; border-radius: 2px; background: ' + bg + '88; border: 1px solid ' + bg}
          />
        `;
      })}
    </div>
  `;
}

export function Leaderboard() {
  const [selected, setSelected] = useState({ metric: 'request_throughput', stat: 'avg' });
  const [model, setModel] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getLeaderboard(selected.metric, selected.stat)
      .then((resp) => {
        if (!cancelled) setData(resp);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selected.metric, selected.stat]);

  const entries = data?.entries ?? [];

  const filtered = entries.filter((e) => {
    if (model && !(e.model ?? '').toLowerCase().includes(model.toLowerCase())) return false;
    if (endpoint && !(e.endpoint ?? '').toLowerCase().includes(endpoint.toLowerCase())) return false;
    return true;
  });

  const unit = filtered[0]?.unit ?? '';

  const top10 = filtered.slice(0, 10);
  const chartData = {
    labels: top10.map((e) => e.job_id ?? ''),
    datasets: [
      {
        label: selected.metric,
        data: top10.map((e) => e.value ?? 0),
        backgroundColor: top10.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'),
        borderColor: top10.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
        title: {
          display: true,
          text: unit || selected.metric,
          color: palette.overlay1,
          font: { size: 11 },
        },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
      },
    },
  };

  // Check if any entries have percentile data
  const hasPercentiles = filtered.some(e => e.percentiles && (e.percentiles.p50 != null || e.percentiles.p90 != null || e.percentiles.p99 != null));

  return html`
    <div class="leaderboard">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Leaderboard</span>
      </div>

      <!-- Controls -->
      <div class="card" style="margin-bottom: var(--space-4); display: flex; align-items: center; gap: var(--space-6); flex-wrap: wrap">
        <${MetricSelector} value=${selected} onSelect=${setSelected} />
        <div style="display: flex; gap: var(--space-3); align-items: center; flex-wrap: wrap">
          <div style="display: flex; align-items: center; gap: var(--space-2)">
            <label class="metric-selector-label">Model</label>
            <input
              class="metric-selector-select"
              type="text"
              placeholder="Filter by model..."
              value=${model}
              oninput=${(e) => setModel(e.target.value)}
              style="min-width: 160px"
            />
          </div>
          <div style="display: flex; align-items: center; gap: var(--space-2)">
            <label class="metric-selector-label">Endpoint</label>
            <input
              class="metric-selector-select"
              type="text"
              placeholder="Filter by endpoint..."
              value=${endpoint}
              oninput=${(e) => setEndpoint(e.target.value)}
              style="min-width: 160px"
            />
          </div>
        </div>
      </div>

      ${error && html`
        <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
          Failed to load leaderboard: ${error}
        </div>
      `}

      ${loading && html`
        <div class="card" style="text-align: center; padding: var(--space-8); margin-bottom: var(--space-4)">
          <span class="text-dim">Loading...</span>
        </div>
      `}

      ${!loading && !error && filtered.length === 0 && html`
        <div class="card empty-state" style="margin-bottom: var(--space-4)">
          <p class="text-dim">No results found. Complete some benchmarks and try again.</p>
        </div>
      `}

      ${!loading && filtered.length > 0 && html`
        <!-- Bar chart -->
        <div class="card" style="margin-bottom: var(--space-4)">
          <div class="card-title">Top ${top10.length} -- ${selected.metric} (${selected.stat})</div>
          <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${Math.max(200, top10.length * 32)} />
        </div>

        <!-- Feature 7: Heatmap legend (only if percentile data exists) -->
        ${hasPercentiles && html`
          <div class="card" style="margin-bottom: var(--space-4); padding: var(--space-3) var(--space-4)">
            <div style="display: flex; align-items: center; gap: var(--space-4); font-size: var(--font-size-xs); color: ${palette.overlay0}; flex-wrap: wrap">
              <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em">Percentile Heatmap</span>
              <div style="display: flex; align-items: center; gap: var(--space-2)">
                <div style=${'width: 12px; height: 12px; border-radius: 2px; background: ' + palette.green + '88; border: 1px solid ' + palette.green} />
                <span>Good</span>
              </div>
              <div style="display: flex; align-items: center; gap: var(--space-2)">
                <div style=${'width: 12px; height: 12px; border-radius: 2px; background: ' + palette.yellow + '88; border: 1px solid ' + palette.yellow} />
                <span>Mid</span>
              </div>
              <div style="display: flex; align-items: center; gap: var(--space-2)">
                <div style=${'width: 12px; height: 12px; border-radius: 2px; background: ' + palette.red + '88; border: 1px solid ' + palette.red} />
                <span>Poor</span>
              </div>
              <span style="margin-left: var(--space-2)">Cells: p50 | p90 | p99</span>
            </div>
          </div>
        `}

        <!-- Ranked table -->
        <div class="card">
          <div class="card-title">All Results</div>
          <div style="overflow-x: auto">
            <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
              <thead>
                <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">#</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Job</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Namespace</th>
                  <th style="text-align: right; padding: var(--space-2) var(--space-3)">Value</th>
                  ${hasPercentiles && html`
                    <th style="text-align: center; padding: var(--space-2) var(--space-3)">p50/p90/p99</th>
                  `}
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Model</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Endpoint</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Date</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.map((entry, idx) => {
                  const rank = idx + 1;
                  const isTop3 = rank <= 3;
                  const rowColor = rank === 1
                    ? palette.yellow
                    : rank === 2
                    ? palette.subtext1
                    : rank === 3
                    ? palette.peach
                    : null;

                  return html`
                    <tr
                      key=${entry.job_id}
                      style=${'border-bottom: 1px solid var(--surface0);' + (isTop3 ? ' background: ' + rowColor + '0a;' : '')}
                    >
                      <td style=${'padding: var(--space-2) var(--space-3); font-weight: 600;' + (isTop3 ? ' color: ' + rowColor : ' color: var(--overlay0)')}>
                        ${rank}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); font-family: var(--font-mono); font-size: var(--font-size-xs)">
                        ${entry.job_id ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                        ${entry.namespace ?? '---'}
                      </td>
                      <td style=${'padding: var(--space-2) var(--space-3); text-align: right; font-weight: 600;' + (isTop3 ? ' color: ' + rowColor : '')}>
                        ${formatValue(entry.value, entry.unit)}
                      </td>
                      ${hasPercentiles && html`
                        <td style="padding: var(--space-2) var(--space-3); text-align: center">
                          <${PercentileHeatmap}
                            entry=${entry}
                            allEntries=${filtered}
                            metric=${selected.metric}
                          />
                        </td>
                      `}
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                        ${entry.model ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0); font-size: var(--font-size-xs); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
                        ${entry.endpoint ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--overlay0)">
                        ${formatDate(entry.start_time)}
                      </td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          </div>
        </div>
      `}
    </div>
  `;
}
