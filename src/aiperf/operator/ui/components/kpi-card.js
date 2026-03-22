import { html } from 'htm/preact';

/**
 * Metric card with colored top border strip.
 * @param {{ label: string, value: string|number, unit?: string, color?: string, sub?: string }} props
 */
export function KpiCard({ label, value, unit, color, sub }) {
  const valueStyle = color ? `color: ${color}` : '';
  const borderStyle = color ? `position:absolute;top:0;left:0;right:0;height:2px;background:${color};border-radius:10px 10px 0 0` : '';

  return html`
    <div class="metric-card">
      ${color && html`<div style=${borderStyle} />`}
      <span class="metric-label">${label}</span>
      <div class="metric-val-row">
        <span class="metric-val" style=${valueStyle}>
          ${value ?? '\u2014'}
        </span>
        ${unit && html`<span class="metric-unit">${unit}</span>`}
      </div>
      ${sub && html`<div class="metric-sub">${sub}</div>`}
    </div>
  `;
}
