import { html } from 'htm/preact';

/**
 * KPI metric card.
 * @param {{ label: string, value: string|number, unit?: string, color?: string }} props
 */
export function KpiCard({ label, value, unit, color }) {
  const valueStyle = color ? `color: ${color}` : '';

  return html`
    <div class="kpi-card">
      <span class="kpi-card-label">${label}</span>
      <div class="kpi-card-value-row">
        <span class="kpi-card-value" style=${valueStyle}>
          ${value ?? '—'}
        </span>
        ${unit && html`<span class="kpi-card-unit">${unit}</span>`}
      </div>
    </div>
  `;
}
