# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation dashboard generation for synthesized Claude Code datasets."""

from __future__ import annotations

import string
from pathlib import Path

import orjson
from rich.console import Console


def load_sessions(jsonl_path: Path) -> list[dict]:
    """Parse JSONL into grouped sessions with turns."""
    sessions: dict[str, list[dict]] = {}
    with open(jsonl_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            sid = row["session_id"]
            turn = {
                "input_length": row["input_length"],
                "output_length": row["output_length"],
                "delay_ms": row.get("delay", row.get("timestamp", 0.0)),
            }
            sessions.setdefault(sid, []).append(turn)

    return [{"session_id": sid, "turns": turns} for sid, turns in sessions.items()]


def render_simulation(sessions: list[dict], output_path: Path) -> None:
    """Inline session data into HTML template and write file."""
    console = Console()
    sessions_json = orjson.dumps(sessions).decode("utf-8")
    html = _HTML_TEMPLATE.safe_substitute(SESSIONS_JSON=sessions_json)
    output_path.write_text(html, encoding="utf-8")
    console.print(f"[green]Wrote {output_path} ({len(sessions)} sessions)[/green]")


# Using string.Template ($-substitution) to avoid conflicts with JS template literals
_HTML_TEMPLATE = string.Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulation Dashboard</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
  --bg: #1a1a2e;
  --surface: #16213e;
  --border: #0f3460;
  --text: #e0e0e0;
  --muted: #8888aa;
  --green: #76b900;
  --green-dim: rgba(118, 185, 0, 0.3);
  --blue: #4da6ff;
  --blue-dim: rgba(77, 166, 255, 0.3);
  --gray: #555;
  --orange: #e89b00;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  padding: 20px;
}
h1 { font-size: 1.4rem; margin-bottom: 16px; }
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: center;
  background: var(--surface);
  padding: 14px 18px;
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-bottom: 8px;
}
.control-group { display: flex; align-items: center; gap: 6px; }
.control-group label { font-size: 0.85rem; color: var(--muted); white-space: nowrap; }
.control-group input[type="number"] {
  width: 90px;
  padding: 5px 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  font-size: 0.9rem;
  text-align: right;
}
.control-group input[type="range"] {
  width: 120px;
  accent-color: var(--green);
}
.control-group .range-val {
  font-size: 0.85rem;
  color: var(--text);
  min-width: 36px;
  text-align: right;
}
button#run-btn {
  padding: 6px 20px;
  background: var(--green);
  color: #000;
  border: none;
  border-radius: 4px;
  font-weight: 600;
  cursor: pointer;
  font-size: 0.9rem;
}
button#run-btn:hover { filter: brightness(1.15); }
.computed-stats {
  font-size: 0.8rem;
  color: var(--muted);
  background: var(--surface);
  padding: 6px 18px;
  border-radius: 0 0 8px 8px;
  border: 1px solid var(--border);
  border-top: none;
  margin-bottom: 16px;
}
.computed-stats span { color: var(--green); font-weight: 600; }
.stats {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  font-size: 0.85rem;
  color: var(--muted);
  background: var(--surface);
  padding: 10px 18px;
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-bottom: 16px;
}
.stats span { color: var(--text); font-weight: 600; }
.chart-container {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}
.chart-container h2 {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 10px;
  font-weight: 500;
}
svg { display: block; }
.axis text { fill: var(--muted); font-size: 11px; }
.axis path, .axis line { stroke: var(--border); }
.crosshair { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3,3; pointer-events: none; }
.gantt-wrap {
  max-height: 500px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--border) var(--surface);
}
.gantt-legend {
  display: flex;
  gap: 16px;
  margin-bottom: 8px;
  font-size: 0.8rem;
  color: var(--muted);
}
.gantt-legend .swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}
.tooltip {
  position: fixed;
  background: rgba(22, 33, 62, 0.95);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 10px;
  font-size: 0.8rem;
  pointer-events: none;
  display: none;
  z-index: 100;
  line-height: 1.5;
}
</style>
</head>
<body>
<h1>Simulation Dashboard</h1>

<div class="controls">
  <div class="control-group">
    <label>Concurrency:</label>
    <input id="concurrency" type="number" value="50" min="1">
  </div>
  <div class="control-group">
    <label>Cache Hit %:</label>
    <input id="cache-hit" type="range" min="0" max="100" value="95">
    <span class="range-val" id="cache-hit-val">95%</span>
  </div>
  <div class="control-group">
    <label>Prefill Workers:</label>
    <input id="prefill-workers" type="number" value="8" min="1">
  </div>
  <div class="control-group">
    <label>DP Workers:</label>
    <input id="dp-workers" type="number" value="4" min="1">
  </div>
  <div class="control-group">
    <label>Per-Worker TPS:</label>
    <input id="per-worker-tps" type="number" value="4000" min="1">
  </div>
  <div class="control-group">
    <label>Decode TPS:</label>
    <input id="decode-tps" type="number" value="200" min="1">
  </div>
  <button id="run-btn">Run</button>
</div>
<div class="computed-stats" id="computed-stats"></div>

<div class="stats" id="stats"></div>
<div class="chart-container"><h2>Active Requests Over Time</h2><div id="chart-requests"></div></div>
<div class="chart-container"><h2>Input Tokens In-Flight</h2><div id="chart-tokens"></div></div>
<div class="chart-container">
  <h2>Session Gantt (by concurrency slot)</h2>
  <div class="gantt-legend">
    <span><span class="swatch" style="background:#555;opacity:0.5"></span>Wait / Delay</span>
    <span><span class="swatch" style="background:#e89b00"></span>Prefill</span>
    <span><span class="swatch" style="background:#76b900"></span>Decode</span>
    <span><span class="swatch" style="background:#4da6ff"></span>Session boundary</span>
  </div>
  <div class="gantt-wrap" id="gantt-wrap"><div id="chart-gantt"></div></div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const SESSIONS = $SESSIONS_JSON;

function simulate(sessions, concurrency, cacheHitRate, prefillWorkers, dpWorkers, perWorkerTps, decodeTps) {
  const effectivePrefillWorkers = prefillWorkers * dpWorkers;
  const prefillWorkerFreeAt = new Float64Array(effectivePrefillWorkers);

  const sessionStates = sessions.map(() => ({
    startTime: null, endTime: null, turnEvents: [], slot: undefined,
  }));

  const pq = [];
  function pushEvent(time, type, sIdx, tIdx) {
    pq.push({ time, type, sIdx, tIdx });
  }
  function popEvent() {
    if (!pq.length) return null;
    let minIdx = 0;
    for (let i = 1; i < pq.length; i++) {
      if (pq[i].time < pq[minIdx].time) minIdx = i;
    }
    const e = pq[minIdx];
    pq[minIdx] = pq[pq.length - 1];
    pq.pop();
    return e;
  }

  function acquirePrefillWorker(readyTime) {
    let bestIdx = 0;
    for (let i = 1; i < effectivePrefillWorkers; i++) {
      if (prefillWorkerFreeAt[i] < prefillWorkerFreeAt[bestIdx]) bestIdx = i;
    }
    const workerFree = prefillWorkerFreeAt[bestIdx];
    const startTime = Math.max(readyTime, workerFree);
    return { workerIdx: bestIdx, startTime };
  }

  function releasePrefillWorker(workerIdx, endTime) {
    prefillWorkerFreeAt[workerIdx] = endTime;
  }

  // Slot tracking for Gantt
  const slotFreeAt = new Float64Array(concurrency);
  let nextSlot = 0;

  function assignSlot(sIdx, startTime) {
    if (nextSlot < concurrency) {
      const slot = nextSlot++;
      sessionStates[sIdx].slot = slot;
      return;
    }
    let bestSlot = 0;
    for (let i = 1; i < concurrency; i++) {
      if (slotFreeAt[i] < slotFreeAt[bestSlot]) bestSlot = i;
    }
    sessionStates[sIdx].slot = bestSlot;
  }

  function releaseSlot(sIdx, endTime) {
    const slot = sessionStates[sIdx].slot;
    if (slot !== undefined) slotFreeAt[slot] = endTime;
  }

  let activeCount = 0;
  let nextSession = 0;

  function startSession(sIdx, time, inheritSlot) {
    sessionStates[sIdx].startTime = time;
    if (inheritSlot !== undefined) {
      sessionStates[sIdx].slot = inheritSlot;
    } else {
      assignSlot(sIdx, time);
    }
    startTurn(sIdx, 0, time);
  }

  function startTurn(sIdx, tIdx, time) {
    const turn = sessions[sIdx].turns[tIdx];
    const delay = tIdx === 0 ? 0 : turn.delay_ms;
    const turnReadyTime = time + delay;

    const effectiveInputTokens = turn.input_length * (1 - cacheHitRate / 100);
    const prefillDuration = (effectiveInputTokens / perWorkerTps) * 1000;
    const decodeDuration = (turn.output_length / decodeTps) * 1000;

    const { workerIdx, startTime: prefillStart } = acquirePrefillWorker(turnReadyTime);
    const prefillEnd = prefillStart + prefillDuration;
    releasePrefillWorker(workerIdx, prefillEnd);

    const decodeStart = prefillEnd;
    const decodeEnd = decodeStart + decodeDuration;

    pushEvent(decodeStart, 'request_start', sIdx, tIdx);
    pushEvent(decodeEnd, 'request_end', sIdx, tIdx);

    sessionStates[sIdx].turnEvents.push({
      turnIdx: tIdx,
      delayStart: time,
      prefillStart: prefillStart,
      decodeStart: decodeStart,
      decodeEnd: decodeEnd,
      input_length: turn.input_length,
      output_length: turn.output_length,
      effective_input_tokens: Math.round(effectiveInputTokens),
    });
  }

  // Launch initial batch
  while (nextSession < sessions.length && activeCount < concurrency) {
    activeCount++;
    startSession(nextSession, 0);
    nextSession++;
  }

  let activeRequests = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  let maxTime = 0;

  const timeSeriesRaw = [];
  let ev;
  while ((ev = popEvent()) !== null) {
    const { time, type, sIdx, tIdx } = ev;
    maxTime = Math.max(maxTime, time);

    if (type === 'request_start') {
      activeRequests++;
      const turn = sessions[sIdx].turns[tIdx];
      inputTokens += turn.input_length;
      outputTokens += turn.output_length;
    } else if (type === 'request_end') {
      activeRequests--;
      const turn = sessions[sIdx].turns[tIdx];
      inputTokens -= turn.input_length;
      outputTokens -= turn.output_length;

      if (tIdx + 1 < sessions[sIdx].turns.length) {
        startTurn(sIdx, tIdx + 1, time);
      } else {
        sessionStates[sIdx].endTime = time;
        const freedSlot = sessionStates[sIdx].slot;
        releaseSlot(sIdx, time);
        activeCount--;
        if (nextSession < sessions.length) {
          activeCount++;
          startSession(nextSession, time, freedSlot);
          nextSession++;
        }
      }
    }

    timeSeriesRaw.push({
      time_s: time / 1000,
      active_requests: activeRequests,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      queued: sessions.length - nextSession,
      active_sessions: activeCount,
    });
  }

  // Adaptive downsampling
  const maxPoints = Math.min(2000, Math.max(500, timeSeriesRaw.length));
  let timeSeries;
  if (timeSeriesRaw.length <= maxPoints) {
    timeSeries = timeSeriesRaw;
  } else {
    timeSeries = [];
    const step = timeSeriesRaw.length / maxPoints;
    for (let i = 0; i < maxPoints; i++) {
      timeSeries.push(timeSeriesRaw[Math.floor(i * step)]);
    }
    timeSeries.push(timeSeriesRaw[timeSeriesRaw.length - 1]);
  }

  // Compute aggregate stats
  let totalPrefillMs = 0, totalDecodeMs = 0, totalWaitMs = 0, turnCount = 0;
  let ttftSum = 0;
  sessionStates.forEach(s => {
    s.turnEvents.forEach(evt => {
      totalWaitMs += evt.prefillStart - evt.delayStart;
      totalPrefillMs += evt.decodeStart - evt.prefillStart;
      totalDecodeMs += evt.decodeEnd - evt.decodeStart;
      ttftSum += evt.decodeStart - evt.delayStart;
      turnCount++;
    });
  });
  const avgTtft = turnCount > 0 ? ttftSum / turnCount : 0;

  return { timeSeries, sessionStates, maxTime, totalPrefillMs, totalDecodeMs, totalWaitMs, avgTtft, turnCount };
}

function formatDuration(ms) {
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(0);
  return m + 'm ' + rem + 's';
}

function formatNumber(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

const tooltip = d3.select('#tooltip');

function updateComputedStats() {
  const pw = +document.getElementById('prefill-workers').value;
  const dp = +document.getElementById('dp-workers').value;
  const tps = +document.getElementById('per-worker-tps').value;
  const eff = pw * dp;
  const sysTps = eff * tps;
  document.getElementById('computed-stats').innerHTML =
    `Effective prefill workers: <span>${eff}</span> &nbsp;|&nbsp; System prefill TPS: <span>${formatNumber(sysTps)}</span> tok/s`;
}

function runSimulation() {
  const concurrency = +document.getElementById('concurrency').value;
  const cacheHitRate = +document.getElementById('cache-hit').value;
  const prefillWorkers = +document.getElementById('prefill-workers').value;
  const dpWorkers = +document.getElementById('dp-workers').value;
  const perWorkerTps = +document.getElementById('per-worker-tps').value;
  const decodeTps = +document.getElementById('decode-tps').value;

  updateComputedStats();

  const { timeSeries, sessionStates, maxTime, totalPrefillMs, totalDecodeMs, totalWaitMs, avgTtft, turnCount } =
    simulate(SESSIONS, concurrency, cacheHitRate, prefillWorkers, dpWorkers, perWorkerTps, decodeTps);

  const peakRequests = d3.max(timeSeries, d => d.active_requests);
  const peakISL = d3.max(timeSeries, d => d.input_tokens);
  document.getElementById('stats').innerHTML =
    `Duration: <span>${formatDuration(maxTime)}</span>` +
    ` &nbsp;|&nbsp; Sessions: <span>${SESSIONS.length}</span>` +
    ` &nbsp;|&nbsp; Turns: <span>${turnCount}</span>` +
    ` &nbsp;|&nbsp; Peak requests: <span>${peakRequests}</span>` +
    ` &nbsp;|&nbsp; Peak input ISL: <span>${formatNumber(peakISL)}</span>` +
    ` &nbsp;|&nbsp; Avg TTFT: <span>${(avgTtft / 1000).toFixed(3)}s</span>` +
    ` &nbsp;|&nbsp; Wall-clock breakdown: prefill <span>${formatDuration(totalPrefillMs)}</span>` +
    ` / decode <span>${formatDuration(totalDecodeMs)}</span>` +
    ` / wait <span>${formatDuration(totalWaitMs)}</span>`;

  drawAreaChart('#chart-requests', timeSeries, 'active_requests', 'Requests', '--green', concurrency);
  drawAreaChart('#chart-tokens', timeSeries, 'input_tokens', 'Tokens', '--blue', null);
  drawGantt('#chart-gantt', sessionStates, maxTime, concurrency);
}

function drawAreaChart(selector, data, field, label, colorVar, limitLine) {
  const container = d3.select(selector);
  container.selectAll('*').remove();

  const margin = { top: 10, right: 20, bottom: 30, left: 60 };
  const width = container.node().clientWidth - margin.left - margin.right;
  const height = 180;

  const svg = container.append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom);

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear()
    .domain(d3.extent(data, d => d.time_s))
    .range([0, width]);

  const yMax = d3.max(data, d => d[field]);
  const y = d3.scaleLinear()
    .domain([0, limitLine ? Math.max(yMax, limitLine * 1.1) : yMax * 1.1])
    .range([height, 0])
    .nice();

  const color = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
  const colorDim = color + '40';

  const area = d3.area()
    .x(d => x(d.time_s))
    .y0(height)
    .y1(d => y(d[field]))
    .curve(d3.curveStepAfter);

  const line = d3.line()
    .x(d => x(d.time_s))
    .y(d => y(d[field]))
    .curve(d3.curveStepAfter);

  g.append('path').datum(data).attr('d', area).attr('fill', colorDim);
  g.append('path').datum(data).attr('d', line).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5);

  if (limitLine) {
    g.append('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', y(limitLine)).attr('y2', y(limitLine))
      .attr('stroke', color).attr('stroke-dasharray', '6,4').attr('stroke-width', 1).attr('opacity', 0.6);
    g.append('text')
      .attr('x', width - 4).attr('y', y(limitLine) - 4)
      .attr('text-anchor', 'end').attr('fill', color).attr('font-size', '10px').attr('opacity', 0.8)
      .text('concurrency limit');
  }

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(8).tickFormat(d => d + 's'));
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickFormat(d => formatNumber(d)));

  // Crosshair
  const crosshairLine = g.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', height).style('display', 'none');
  const crosshairDot = g.append('circle').attr('r', 4).attr('fill', color).style('display', 'none');

  const overlay = g.append('rect')
    .attr('width', width).attr('height', height).attr('fill', 'none').attr('pointer-events', 'all');

  overlay.on('mousemove', function(event) {
    const [mx] = d3.pointer(event, this);
    const xVal = x.invert(mx);
    const bisect = d3.bisector(d => d.time_s).left;
    const idx = Math.min(bisect(data, xVal), data.length - 1);
    const d = data[idx];
    crosshairLine.attr('x1', x(d.time_s)).attr('x2', x(d.time_s)).style('display', null);
    crosshairDot.attr('cx', x(d.time_s)).attr('cy', y(d[field])).style('display', null);

    tooltip.style('display', 'block')
      .style('left', (event.clientX + 12) + 'px')
      .style('top', (event.clientY - 30) + 'px')
      .html(`Time: ${d.time_s.toFixed(1)}s<br>${label}: ${formatNumber(d[field])}`);
  });
  overlay.on('mouseleave', function() {
    crosshairLine.style('display', 'none');
    crosshairDot.style('display', 'none');
    tooltip.style('display', 'none');
  });
}

function drawGantt(selector, sessionStates, maxTime, concurrency) {
  const container = d3.select(selector);
  container.selectAll('*').remove();

  // Group sessions by slot
  const slots = Array.from({length: concurrency}, () => []);
  sessionStates.forEach((s, i) => {
    if (s.slot !== undefined) slots[s.slot].push({...s, idx: i});
  });

  const margin = { top: 10, right: 20, bottom: 30, left: 70 };
  const barHeight = 14;
  const barGap = 2;
  const width = container.node().clientWidth - margin.left - margin.right;
  const height = concurrency * (barHeight + barGap);

  const svg = container.append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom);

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, maxTime / 1000]).range([0, width]);
  const green = getComputedStyle(document.documentElement).getPropertyValue('--green').trim();
  const gray = getComputedStyle(document.documentElement).getPropertyValue('--gray').trim();
  const orange = getComputedStyle(document.documentElement).getPropertyValue('--orange').trim();

  slots.forEach((slotSessions, slotIdx) => {
    const y = slotIdx * (barHeight + barGap);

    // Slot label
    g.append('text')
      .attr('x', -4).attr('y', y + barHeight / 2 + 3)
      .attr('text-anchor', 'end').attr('fill', '#8888aa').attr('font-size', '9px')
      .text('Slot ' + slotIdx);

    slotSessions.forEach((session, sInSlot) => {
      // Session boundary marker
      if (session.startTime !== null) {
        g.append('rect')
          .attr('x', x(session.startTime / 1000) - 1)
          .attr('y', y)
          .attr('width', 2)
          .attr('height', barHeight)
          .attr('fill', '#4da6ff')
          .attr('opacity', 0.8);
      }

      session.turnEvents.forEach(evt => {
        // Wait / delay segment (gray)
        if (evt.prefillStart > evt.delayStart) {
          const seg = g.append('rect')
            .attr('x', x(evt.delayStart / 1000))
            .attr('y', y)
            .attr('width', Math.max(0.5, x(evt.prefillStart / 1000) - x(evt.delayStart / 1000)))
            .attr('height', barHeight)
            .attr('fill', gray)
            .attr('opacity', 0.5);
          seg.on('mousemove', function(event) {
            const dur = (evt.prefillStart - evt.delayStart) / 1000;
            tooltip.style('display', 'block')
              .style('left', (event.clientX + 12) + 'px')
              .style('top', (event.clientY - 30) + 'px')
              .html(`S${session.idx} Turn ${evt.turnIdx} - <b>Wait</b><br>Duration: ${dur.toFixed(3)}s`);
          }).on('mouseleave', () => tooltip.style('display', 'none'));
        }

        // Prefill segment (orange)
        if (evt.decodeStart > evt.prefillStart) {
          const seg = g.append('rect')
            .attr('x', x(evt.prefillStart / 1000))
            .attr('y', y)
            .attr('width', Math.max(0.5, x(evt.decodeStart / 1000) - x(evt.prefillStart / 1000)))
            .attr('height', barHeight)
            .attr('fill', orange);
          seg.on('mousemove', function(event) {
            const dur = (evt.decodeStart - evt.prefillStart) / 1000;
            tooltip.style('display', 'block')
              .style('left', (event.clientX + 12) + 'px')
              .style('top', (event.clientY - 30) + 'px')
              .html(`S${session.idx} Turn ${evt.turnIdx} - <b>Prefill</b><br>` +
                `Input: ${formatNumber(evt.input_length)} tok (effective: ${formatNumber(evt.effective_input_tokens)})<br>` +
                `Duration: ${dur.toFixed(3)}s`);
          }).on('mouseleave', () => tooltip.style('display', 'none'));
        }

        // Decode segment (green)
        const seg = g.append('rect')
          .attr('x', x(evt.decodeStart / 1000))
          .attr('y', y)
          .attr('width', Math.max(0.5, x(evt.decodeEnd / 1000) - x(evt.decodeStart / 1000)))
          .attr('height', barHeight)
          .attr('fill', green);
        seg.on('mousemove', function(event) {
          const dur = (evt.decodeEnd - evt.decodeStart) / 1000;
          tooltip.style('display', 'block')
            .style('left', (event.clientX + 12) + 'px')
            .style('top', (event.clientY - 30) + 'px')
            .html(`S${session.idx} Turn ${evt.turnIdx} - <b>Decode</b><br>` +
              `Output: ${formatNumber(evt.output_length)} tok<br>` +
              `Duration: ${dur.toFixed(3)}s`);
        }).on('mouseleave', () => tooltip.style('display', 'none'));
      });
    });
  });

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(8).tickFormat(d => d + 's'));
}

// Cache hit slider live update
document.getElementById('cache-hit').addEventListener('input', function() {
  document.getElementById('cache-hit-val').textContent = this.value + '%';
});

// Update computed stats on any input change
['prefill-workers', 'dp-workers', 'per-worker-tps'].forEach(id => {
  document.getElementById(id).addEventListener('input', updateComputedStats);
});

// Run on load and on button click
document.getElementById('run-btn').addEventListener('click', runSimulation);
document.addEventListener('keydown', e => { if (e.key === 'Enter') runSimulation(); });
updateComputedStats();
runSimulation();
</script>
</body>
</html>
""")
