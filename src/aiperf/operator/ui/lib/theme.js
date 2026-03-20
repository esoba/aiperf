// Catppuccin Mocha color tokens
export const palette = {
  // Base layers
  base: '#1e1e2e',
  mantle: '#181825',
  crust: '#11111b',

  // Surface layers
  surface0: '#313244',
  surface1: '#45475a',
  surface2: '#585b70',

  // Overlay layers
  overlay0: '#6c7086',
  overlay1: '#7f849c',
  overlay2: '#9399b2',

  // Text
  subtext0: '#a6adc8',
  subtext1: '#bac2de',
  text: '#cdd6f4',

  // Accent colors
  lavender: '#b4befe',
  blue: '#89b4fa',
  sapphire: '#74c7ec',
  sky: '#89dceb',
  teal: '#94e2d5',
  green: '#a6e3a1',
  yellow: '#f9e2af',
  peach: '#fab387',
  maroon: '#eba0ac',
  red: '#f38ba8',
  mauve: '#cba6f7',
  pink: '#f5c2e7',
  flamingo: '#f2cdcd',
  rosewater: '#f5e0dc',
};

// Semantic mappings
export const colors = {
  bg: palette.base,
  bgAlt: palette.mantle,
  bgElevated: palette.surface0,
  bgRaised: palette.surface1,

  border: palette.surface1,
  borderSubtle: palette.surface0,

  text: palette.text,
  textMuted: palette.subtext0,
  textDim: palette.overlay0,

  accent: palette.mauve,
  accentAlt: palette.blue,

  success: palette.green,
  warning: palette.yellow,
  error: palette.red,
  info: palette.blue,

  // Job phase colors
  phaseRunning: palette.blue,
  phaseCompleted: palette.green,
  phaseFailed: palette.red,
  phasePending: palette.yellow,
  phaseUnknown: palette.overlay0,
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
