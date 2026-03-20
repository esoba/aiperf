import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, clusterInfo } from '../lib/state.js';
import { phaseColor, colors, palette } from '../lib/theme.js';
import { navigate } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper } from '../components/chart-wrapper.js';

function formatElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function bestThroughput(jobList) {
  let best = null;
  let bestName = null;
  for (const job of jobList) {
    const phase = (job.phase ?? '').toLowerCase();
    if (phase !== 'completed' && phase !== 'succeeded') continue;
    const val = job.throughputRps ?? null;
    if (val != null && (best === null || val > best)) {
      best = val;
      bestName = job.name;
    }
  }
  return { value: best, name: bestName };
}

function gpuCount(cluster) {
  return cluster?.gpus ?? cluster?.gpuCount ?? cluster?.gpu_count ?? null;
}

// Assign a stable color to a model name via simple hash
const MODEL_COLORS = [
  palette.blue, palette.mauve, palette.green, palette.peach,
  palette.pink, palette.teal, palette.sapphire, palette.yellow,
  palette.flamingo, palette.lavender, palette.red, palette.sky,
];

function modelColor(model) {
  if (!model) return palette.overlay0;
  let hash = 0;
  for (let i = 0; i < model.length; i++) {
    hash = ((hash << 5) - hash + model.charCodeAt(i)) | 0;
  }
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length];
}

// Feature 1+2: Throughput vs Latency scatter plot with quadrant labels
function ThroughputLatencyScatter({ completedJobs }) {
  if (!completedJobs || completedJobs.length === 0) return null;

  const points = completedJobs.filter(
    j => j.throughputRps != null && j.latencyP99Ms != null,
  );
  if (points.length === 0) return null;

  // Group by model for coloring
  const modelGroups = {};
  for (const job of points) {
    const m = job.model ?? 'unknown';
    if (!modelGroups[m]) modelGroups[m] = [];
    modelGroups[m].push(job);
  }

  const datasets = Object.entries(modelGroups).map(([model, jobs]) => ({
    label: model,
    data: jobs.map(j => ({
      x: j.throughputRps,
      y: j.latencyP99Ms,
      jobName: j.name,
    })),
    backgroundColor: modelColor(model) + 'cc',
    borderColor: modelColor(model),
    borderWidth: 1,
    pointRadius: 6,
    pointHoverRadius: 9,
  }));

  const chartData = { datasets };

  const chartOptions = {
    plugins: {
      legend: {
        display: Object.keys(modelGroups).length > 1,
        labels: { color: palette.overlay0, font: { size: 11 } },
      },
      tooltip: {
        callbacks: {
          label: ctx => {
            const pt = ctx.raw;
            return ` ${pt.jobName}: ${pt.x.toFixed(1)} req/s, ${pt.y.toFixed(0)} ms p99`;
          },
        },
      },
      // Quadrant labels plugin
      quadrantLabels: {
        enabled: true,
      },
    },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Throughput (req/s)', color: palette.overlay1, font: { size: 11 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
      y: {
        type: 'linear',
        title: { display: true, text: 'Latency P99 (ms)', color: palette.overlay1, font: { size: 11 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
    },
  };

  // Register a custom Chart.js plugin for quadrant labels (inline)
  const quadrantPlugin = {
    id: 'quadrantLabels',
    afterDraw(chart) {
      const { ctx, chartArea: { left, right, top, bottom } } = chart;
      const midX = (left + right) / 2;
      const midY = (top + bottom) / 2;

      ctx.save();
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.fillStyle = palette.overlay0 + '60';
      ctx.textAlign = 'center';

      // Top-right: best
      ctx.fillText('High Throughput, Low Latency', (midX + right) / 2, top + 16);
      // Bottom-left: worst
      ctx.fillText('Low Throughput, High Latency', (left + midX) / 2, bottom - 8);

      ctx.restore();
    },
  };

  // We need to register the plugin before the chart is created.
  // ChartWrapper creates the chart on mount, so we register it globally once.
  if (window.Chart && !window._quadrantPluginRegistered) {
    window.Chart.register(quadrantPlugin);
    window._quadrantPluginRegistered = true;
  }

  return html`
    <div class="card" style="margin-bottom: var(--space-6)">
      <div class="card-title">Throughput vs Latency (Completed Jobs)</div>
      <${ChartWrapper}
        type="scatter"
        data=${chartData}
        options=${chartOptions}
        height=${280}
      />
    </div>
  `;
}

// Feature 10: Status summary bar
function StatusSummaryBar({ allJobs, cluster, best }) {
  const running = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'running' || p === 'initializing' || p === 'pending';
  }).length;
  const completed = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'completed' || p === 'succeeded';
  }).length;
  const failed = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'failed' || p === 'error';
  }).length;

  const nodes = cluster?.nodes ?? cluster?.nodeCount ?? cluster?.node_count ?? '?';
  const gpus = cluster?.gpus ?? cluster?.gpuCount ?? cluster?.gpu_count ?? '?';

  const parts = [];
  parts.push(`Cluster: ${nodes} node${nodes !== 1 ? 's' : ''}, ${gpus} GPU${gpus !== 1 ? 's' : ''}`);
  parts.push(`${completed} completed, ${running} running, ${failed} failed`);
  if (best.value != null && best.name) {
    parts.push(`Best: ${best.value.toFixed(1)} req/s (${best.name})`);
  }

  return html`
    <div
      style=${'display: flex; align-items: center; gap: var(--space-4); padding: var(--space-2) var(--space-4); margin-bottom: var(--space-4); background: ' + palette.surface0 + '40; border-radius: var(--radius-md); font-size: var(--font-size-sm); color: ' + palette.subtext0 + '; flex-wrap: wrap; border: 1px solid ' + palette.surface0}
    >
      ${parts.map((part, i) => html`
        <span key=${i}>
          ${i > 0 && html`<span style=${'color: ' + palette.overlay0 + '; margin-right: var(--space-4)'}>|</span>`}
          ${part}
        </span>
      `)}
    </div>
  `;
}

export function Dashboard() {
  const [localJobs, setLocalJobs] = useState(jobs.value);
  const [cluster, setCluster] = useState(clusterInfo.value);
  const [clusterError, setClusterError] = useState(false);
  const [historyData, setHistoryData] = useState(null);

  useEffect(() => {
    const ac = new AbortController();

    poll(
      async () => {
        const data = await api.listJobs();
        const list = data?.jobs ?? [];
        jobs.value = list;
        setLocalJobs(list);
      },
      5000,
      ac.signal,
    );

    poll(
      async () => {
        try {
          const data = await api.getCluster();
          clusterInfo.value = data;
          setCluster(data);
          setClusterError(false);
        } catch (_e) {
          setClusterError(true);
        }
      },
      5000,
      ac.signal,
    );

    // Fetch history sparkline once on mount
    api
      .getHistory('request_throughput', 'avg')
      .then((data) => setHistoryData(data))
      .catch(() => {});

    return () => ac.abort();
  }, []);

  const allJobs = localJobs;
  const running = allJobs.filter((j) => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'running' || p === 'initializing' || p === 'pending';
  });
  const completed = allJobs.filter((j) => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'completed' || p === 'succeeded';
  });
  const failed = allJobs.filter((j) => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'failed' || p === 'error';
  });

  const activeJobs = allJobs.filter((j) => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'running' || p === 'pending' || p === 'initializing';
  });

  const gpus = gpuCount(cluster);
  const best = bestThroughput(allJobs);

  // Build sparkline chart data from history response
  const sparklineData = (() => {
    const points = historyData?.points ?? historyData?.data ?? historyData ?? [];
    if (!Array.isArray(points) || points.length === 0) return null;
    const labels = points.map((p) => p.label ?? p.time ?? '');
    const values = points.map((p) => p.value ?? p.avg ?? 0);
    return {
      labels,
      datasets: [
        {
          data: values,
          borderColor: palette.blue,
          backgroundColor: palette.blue + '22',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    };
  })();

  const sparklineOptions = {
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { display: false },
    },
    elements: { point: { radius: 0 } },
  };

  return html`
    <div class="dashboard">
      ${clusterError && html`
        <div class="cluster-warning-banner">
          Cluster endpoint unavailable — data may be stale.
        </div>
      `}

      <!-- Feature 10: Status Summary Bar -->
      <${StatusSummaryBar} allJobs=${allJobs} cluster=${cluster} best=${best} />

      <div class="kpi-row">
        <${KpiCard}
          label="Running"
          value=${running.length}
          color=${colors.phaseRunning}
        />
        <${KpiCard}
          label="Completed"
          value=${completed.length}
          color=${colors.phaseCompleted}
        />
        <${KpiCard}
          label="GPUs"
          value=${gpus ?? '---'}
          color=${palette.mauve}
        />
        <${KpiCard}
          label="Best Throughput"
          value=${best.value != null ? best.value.toFixed(1) : '---'}
          unit=${best.value != null ? 'req/s' : ''}
          color=${palette.yellow}
        />
      </div>

      ${sparklineData && html`
        <div class="card" style="margin-bottom: var(--space-6)">
          <div class="card-title">Throughput History</div>
          <${ChartWrapper}
            type="line"
            data=${sparklineData}
            options=${sparklineOptions}
            height=${120}
          />
        </div>
      `}

      <!-- Feature 1+2: Throughput vs Latency Scatter -->
      <${ThroughputLatencyScatter} completedJobs=${completed} />

      <div class="section-header">
        <span class="section-title">Active Jobs</span>
        <span class="text-dim" style="font-size: var(--font-size-sm)">
          ${activeJobs.length} job${activeJobs.length !== 1 ? 's' : ''}
        </span>
      </div>

      ${activeJobs.length === 0
        ? html`
          <div class="empty-state card">
            <p class="text-dim">No active jobs. Start a benchmark with <code>aiperf kube run</code>.</p>
          </div>
        `
        : html`
          <div class="active-jobs-list">
            ${activeJobs.map((job) => {
              const phase = job.phase ?? 'Unknown';
              const pct = Math.round(job.progressPercent ?? 0);
              const color = phaseColor(phase);
              const startTime = job.startTime;
              const elapsed = startTime
                ? formatElapsed(Date.now() - new Date(startTime).getTime())
                : null;

              return html`
                <div
                  key=${job.namespace + '/' + job.name}
                  class="active-job-row card"
                  onclick=${() => navigate('/jobs/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
                  style="cursor: pointer; margin-bottom: var(--space-3)"
                >
                  <div class="active-job-header">
                    <span class="active-job-name">${job.name}</span>
                    <span class="phase-badge" style=${'background: ' + color + '22; color: ' + color + '; border-color: ' + color + '44'}>
                      ${phase}
                    </span>
                  </div>
                  <div class="active-job-meta text-dim" style="font-size: var(--font-size-sm); margin: var(--space-1) 0">
                    ${job.model ?? '---'}
                    ${elapsed && html` · ${elapsed}`}
                    ${job.throughputRps != null && html` · ${job.throughputRps.toFixed(1)} req/s`}
                  </div>
                  ${pct > 0 && html`
                    <div class="progress-track" style="margin-top: var(--space-2)">
                      <div class="progress-fill" style=${'width: ' + pct + '%; background: ' + color} />
                    </div>
                  `}
                </div>
              `;
            })}
          </div>
        `
      }

      ${failed.length > 0 && html`
        <div class="section-header" style="margin-top: var(--space-6)">
          <span class="section-title" style="color: ${colors.phaseFailed}">Failed Jobs</span>
          <span class="text-dim" style="font-size: var(--font-size-sm)">${failed.length}</span>
        </div>
        <div class="active-jobs-list">
          ${failed.map((job) => {
            const color = phaseColor(job.phase ?? 'Failed');
            return html`
              <div
                key=${job.namespace + '/' + job.name}
                class="active-job-row card"
                onclick=${() => navigate('/jobs/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
                style="cursor: pointer; margin-bottom: var(--space-3); border-color: ${color}44"
              >
                <div class="active-job-header">
                  <span class="active-job-name">${job.name}</span>
                  <span class="phase-badge" style=${'background: ' + color + '22; color: ' + color + '; border-color: ' + color + '44'}>
                    ${job.phase ?? 'Failed'}
                  </span>
                </div>
                ${job.error && html`
                  <div style="font-size: var(--font-size-sm); color: ${colors.error}; margin-top: var(--space-1)">
                    ${job.error}
                  </div>
                `}
              </div>
            `;
          })}
        </div>
      `}

      <!-- Job History (completed jobs) -->
      ${completed.length > 0 && html`
        <div class="section-header" style="margin-top: var(--space-6)">
          <span class="section-title">Job History</span>
          <span class="text-dim" style="font-size: var(--font-size-sm)">${completed.length} completed</span>
        </div>
        <div class="job-table-wrapper">
          <table class="job-table">
            <thead>
              <tr>
                <th class="job-table-th">Name</th>
                <th class="job-table-th">Model</th>
                <th class="job-table-th">Throughput</th>
                <th class="job-table-th">Latency P99</th>
                <th class="job-table-th">Age</th>
              </tr>
            </thead>
            <tbody>
              ${completed.map((job) => {
                const age = job.created ? (() => {
                  const ms = Date.now() - new Date(job.created).getTime();
                  const s = Math.floor(ms / 1000);
                  if (s < 60) return s + 's';
                  const m = Math.floor(s / 60);
                  if (m < 60) return m + 'm';
                  const h = Math.floor(m / 60);
                  if (h < 24) return h + 'h';
                  return Math.floor(h / 24) + 'd';
                })() : '---';
                return html`
                  <tr
                    key=${job.namespace + '/' + job.name}
                    class="job-table-row"
                    onclick=${() => navigate('/jobs/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
                    style="cursor: pointer"
                  >
                    <td class="job-table-td job-table-name">${job.name}</td>
                    <td class="job-table-td text-dim">${job.model ?? '---'}</td>
                    <td class="job-table-td">${job.throughputRps != null ? job.throughputRps.toFixed(1) + ' req/s' : '---'}</td>
                    <td class="job-table-td">${job.latencyP99Ms != null ? (job.latencyP99Ms > 1000 ? (job.latencyP99Ms / 1000).toFixed(1) + ' s' : job.latencyP99Ms.toFixed(0) + ' ms') : '---'}</td>
                    <td class="job-table-td text-dim">${age}</td>
                  </tr>
                `;
              })}
            </tbody>
          </table>
        </div>
      `}
    </div>
  `;
}
