// AIPerf dark theme - benchmark-focused color system
export const palette = {
  // Base layers (near-black with slight blue tint)
  bg: '#0a0a10',
  bgCard: '#0f0f18',
  bgRaised: '#141420',

  // Borders
  border: '#1c1c2e',
  borderHover: '#2a2a44',
  borderSubtle: '#14141e',

  // Text
  dim: '#3a3a55',
  muted: '#6a6a88',
  sub: '#9090ab',
  text: '#d8d8e8',
  white: '#f0f0ff',

  // Accent
  accent: '#8b5cf6',
  accentDim: 'rgba(139,92,246,0.10)',

  // Semantic
  blue: '#3b82f6',
  cyan: '#06b6d4',
  green: '#10b981',
  amber: '#f59e0b',
  red: '#ef4444',
  pink: '#ec4899',
  orange: '#f97316',
  teal: '#14b8a6',
  indigo: '#6366f1',
  mauve: '#8b5cf6',

  // Compatibility aliases (used by other pages not being rewritten)
  base: '#0a0a10',
  mantle: '#0f0f18',
  crust: '#070710',
  surface0: '#1c1c2e',
  surface1: '#2a2a44',
  surface2: '#3a3a55',
  overlay0: '#6a6a88',
  overlay1: '#9090ab',
  overlay2: '#9090ab',
  subtext0: '#9090ab',
  subtext1: '#d8d8e8',
  yellow: '#f59e0b',
  peach: '#f97316',
  maroon: '#ef4444',
  sapphire: '#06b6d4',
  sky: '#06b6d4',
  lavender: '#8b5cf6',
  flamingo: '#ec4899',
  rosewater: '#ec4899',
};

// Semantic mappings
export const colors = {
  bg: palette.bg,
  bgAlt: palette.bgCard,
  bgElevated: palette.bgRaised,
  bgRaised: palette.bgRaised,

  border: palette.border,
  borderSubtle: palette.borderSubtle,

  text: palette.text,
  textMuted: palette.sub,
  textDim: palette.muted,

  accent: palette.accent,
  accentAlt: palette.blue,

  success: palette.green,
  warning: palette.amber,
  error: palette.red,
  info: palette.blue,

  // Job phase colors
  phaseRunning: palette.blue,
  phaseCompleted: palette.green,
  phaseFailed: palette.red,
  phasePending: palette.amber,
  phaseUnknown: palette.muted,
};

// Status to color mapping
export function phaseColor(phase) {
  const p = (phase || '').toLowerCase();
  if (p === 'running') return colors.phaseRunning;
  if (p === 'completed' || p === 'succeeded') return colors.phaseCompleted;
  if (p === 'failed' || p === 'error') return colors.phaseFailed;
  if (p === 'pending' || p === 'initializing') return colors.phasePending;
  return colors.phaseUnknown;
}

// Stable model-color assignment via string hash
const MODEL_COLORS = [
  palette.blue, palette.green, palette.amber, palette.pink,
  palette.cyan, palette.teal, palette.orange, palette.indigo,
  palette.red, palette.accent,
];

/**
 * Get a stable color for a model name (hashed).
 * @param {string} model
 * @returns {string}
 */
export function modelColor(model) {
  if (!model) return palette.muted;
  let hash = 0;
  for (let i = 0; i < model.length; i++) {
    hash = ((hash << 5) - hash + model.charCodeAt(i)) | 0;
  }
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length];
}
