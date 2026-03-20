import { html, render } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { route, matchRoute } from './lib/router.js';
import { globalError } from './lib/state.js';
import { IconRail } from './components/icon-rail.js';
import { CommandPalette } from './components/command-palette.js';
import { Dashboard } from './pages/dashboard.js';
import { Jobs } from './pages/jobs.js';
import { JobDetail } from './pages/job-detail.js';
import { Leaderboard } from './pages/leaderboard.js';
import { Compare } from './pages/compare.js';
import { History } from './pages/history.js';

function App() {
  const [showPalette, setShowPalette] = useState(false);
  const currentRoute = route.value;
  const error = globalError.value;

  // Ctrl+K to open command palette
  useEffect(() => {
    function handleKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setShowPalette((v) => !v);
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  // Route matching
  let page;
  let title;

  const jobDetailMatch = matchRoute('/jobs/:ns/:name', currentRoute);

  if (currentRoute === '/' || currentRoute === '') {
    page = html`<${Dashboard} />`;
    title = 'Dashboard';
  } else if (currentRoute === '/jobs') {
    page = html`<${Jobs} />`;
    title = 'Jobs';
  } else if (jobDetailMatch) {
    page = html`<${JobDetail} namespace=${jobDetailMatch.ns} name=${jobDetailMatch.name} />`;
    title = jobDetailMatch.name;
  } else if (currentRoute === '/leaderboard') {
    page = html`<${Leaderboard} />`;
    title = 'Leaderboard';
  } else if (currentRoute === '/compare') {
    page = html`<${Compare} />`;
    title = 'Compare';
  } else if (currentRoute === '/history') {
    page = html`<${History} />`;
    title = 'History';
  } else {
    page = html`<div class="page-stub"><h2>Not Found</h2><p class="text-dim">${currentRoute}</p></div>`;
    title = 'Not Found';
  }

  return html`
    <div class="app">
      <${IconRail} />
      <div class="main">
        <div class="top-bar">
          <h1 class="top-bar-title">${title}</h1>
          <div class="top-bar-actions">
            ${error && html`
              <span class="top-bar-error" title=${error}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                Error
              </span>
            `}
            <button
              class="cmd-k-btn"
              onclick=${() => setShowPalette(true)}
              title="Open command palette (Ctrl+K)"
              aria-label="Open command palette"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              <kbd>Ctrl+K</kbd>
            </button>
          </div>
        </div>
        ${error && html`
          <div class="error-banner">
            <strong>Error:</strong> ${error}
          </div>
        `}
        <div class="content">${page}</div>
      </div>
      ${showPalette && html`<${CommandPalette} onClose=${() => setShowPalette(false)} />`}
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));
