import { html } from 'htm/preact';
import { useRef, useEffect } from 'preact/hooks';

/**
 * Chart.js lifecycle wrapper for Preact.
 * Chart.js is loaded as UMD via <script> in index.html, accessed as window.Chart.
 *
 * @param {{ type: string, data: object, options?: object, height?: number }} props
 */
export function ChartWrapper({ type, data, options = {}, height = 300 }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  // Create chart on mount, destroy on unmount
  useEffect(() => {
    if (!canvasRef.current) return;
    if (!window.Chart) {
      console.warn('ChartWrapper: window.Chart not available - Chart.js not loaded');
      return;
    }

    chartRef.current = new window.Chart(canvasRef.current, {
      type,
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        ...options,
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps - intentionally mount-only

  // Update data when it changes
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.data = data;
    chartRef.current.update();
  }, [data]);

  // Update options when they change
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.options = { responsive: true, maintainAspectRatio: false, ...options };
    chartRef.current.update();
  }, [options]);

  return html`
    <div class="chart-container" style=${'height: ' + height + 'px'}>
      <canvas ref=${canvasRef} />
    </div>
  `;
}
