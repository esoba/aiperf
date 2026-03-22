import { html } from 'htm/preact';

/**
 * Metric card — simple card, brand-colored value for key metrics.
 * @param {{ label: string, value: string|number, unit?: string, color?: string, sub?: string }} props
 */
export function KpiCard({ label, value, unit, color, sub }) {
  const valueStyle = color ? `color: ${color}` : '';

  return html`
    <div class="metric-card">
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
