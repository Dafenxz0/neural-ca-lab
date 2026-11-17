import { stampPattern2D } from './math.js';
import { PLANARIAN_WEIGHTS, LOTUS_WEIGHTS } from './nca_weights.js';

// Classic Bert Chan Orbium glider matrix (20x20)
export const ORBIUM_PATTERN = [
  [0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
  [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
  [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
  [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
  [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
  [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
  [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
  [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
  [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
  [0,0.58,0.49,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
  [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
  [0,0,0,0,0,0,0,0,0.68,0.92,1,1,1,0.88,0.19,0,0,0,0.11,0.12],
  [0,0,0,0,0,0,0,0,0.56,0.87,0.96,1,1,0.97,0.65,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0.21,0.73,0.91,0.98,1,0.99,0.94,0.45,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0.32,0.75,0.91,0.97,0.99,0.96,0.79,0.11,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0.31,0.72,0.88,0.95,0.96,0.88,0.47,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0.28,0.65,0.84,0.91,0.89,0.68,0.04,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0.12,0.48,0.72,0.78,0.65,0.24,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0.02,0.22,0.42,0.38,0.15,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

export const PRESETS = [
  {
    id: 'orbium',
    name: 'Orbium Gliders',
    kind: 'LENIA',
    mode: 'lenia',
    description: 'Smooth ring-kernel solitons (Orbium) that self-propel, collide, and exchange momentum on a toroidal manifold.',
    R: 13,
    beta: [1, 0, 0],
    mu: 0.15,
    sigma: 0.017,
    dt: 0.1,
    palette: 0,
    seedType: 'orbium-fleet',
  },
  {
    id: 'hydrogeminium',
    name: 'Hydrogeminium',
    kind: 'LENIA',
    mode: 'lenia',
    description: 'Dual-shell Lenia colony with concentric kernel peaks producing budding swimmers and dividing membranes.',
    R: 13,
    beta: [1, 0.5, 0],
    mu: 0.26,
    sigma: 0.036,
    dt: 0.08,
    palette: 1,
    seedType: 'rings',
  },
  {
    id: 'synapsia',
    name: 'Synapsia Gyre',
    kind: 'LENIA',
    mode: 'lenia',
    description: 'Three-ring harmonic Lenia field forming pulsating neural-like rings. Slice or perturb waves to trigger fission.',
    R: 14,
    beta: [1, 0.75, 0.25],
    mu: 0.29,
    sigma: 0.045,
    dt: 0.09,
    palette: 2,
    seedType: 'vortex',
  },
  {
    id: 'planarian',
    name: 'Planarian NCA',
    kind: 'NEURAL CA',
    mode: 'nca',
    description: 'Trained 8-channel Morphogenetic Neural CA (Sobel + Laplacian). Grows from 1 pixel and regenerates sliced organs.',
    mu: 0.15,
    sigma: 0.05,
    dt: 0.1,
    palette: 4,
    seedType: 'nca-colony',
    weights: PLANARIAN_WEIGHTS,
  },
  {
    id: 'lotus',
    name: 'Cryo-Lotus NCA',
    kind: 'NEURAL CA',
    mode: 'nca',
    description: 'Self-healing 5-fold morphogenetic tissue trained with damage augmentation. Cut chunks out and watch them regrow.',
    mu: 0.15,
    sigma: 0.05,
    dt: 0.1,
    palette: 4,
    seedType: 'nca-garden',
    weights: LOTUS_WEIGHTS,
  },
];

function stampRing(grid, gridW, gridH, cx, cy, rOuter, rInner = 0, amp = 1.0) {
  for (let dy = -rOuter; dy <= rOuter; dy++) {
    for (let dx = -rOuter; dx <= rOuter; dx++) {
      const d = Math.hypot(dx, dy);
      if (d <= rOuter && d >= rInner) {
        const t = (d - rInner) / Math.max(1, rOuter - rInner);
        const val = Math.sin(t * Math.PI) * amp;
        const gx = ((Math.round(cx + dx) % gridW) + gridW) % gridW;
        const gy = ((Math.round(cy + dy) % gridH) + gridH) % gridH;
        const idx = gy * gridW + gx;
        if (val > grid[idx]) grid[idx] = val;
      }
    }
  }
}

function plantNcaSeed(state0, state1, gridW, gridH, cx, cy) {
  const gx = ((Math.round(cx) % gridW) + gridW) % gridW;
  const gy = ((Math.round(cy) % gridH) + gridH) % gridH;
  const i4 = (gy * gridW + gx) * 4;
  state0[i4 + 0] = 0;
  state0[i4 + 1] = 0;
  state0[i4 + 2] = 0;
  state0[i4 + 3] = 1;
  state1[i4 + 0] = 1;
  state1[i4 + 1] = 1;
  state1[i4 + 2] = 1;
  state1[i4 + 3] = 1;
}

export function createInitialStates(preset, gridW, gridH = gridW) {
  const cells = gridW * gridH;
  const state0 = new Float32Array(cells * 4);
  const state1 = new Float32Array(cells * 4);

  if (preset.mode === 'lenia') {
    const scalar = new Float32Array(cells);
    if (preset.seedType === 'orbium-fleet') {
      const positions = [
        [gridW * 0.25, gridH * 0.3, 0],
        [gridW * 0.7, gridH * 0.32, 1],
        [gridW * 0.35, gridH * 0.72, 2],
        [gridW * 0.72, gridH * 0.74, 3],
        [gridW * 0.5, gridH * 0.5, 0],
      ];
      for (const [cx, cy, rot] of positions) {
        stampPattern2D(scalar, gridW, ORBIUM_PATTERN, cx, cy, rot, gridH);
      }
    } else if (preset.seedType === 'rings') {
      stampRing(scalar, gridW, gridH, gridW * 0.35, gridH * 0.4, 12, 2, 0.95);
      stampRing(scalar, gridW, gridH, gridW * 0.65, gridH * 0.58, 13, 3, 0.92);
      stampRing(scalar, gridW, gridH, gridW * 0.5, gridH * 0.28, 10, 2, 0.9);
    } else {
      stampRing(scalar, gridW, gridH, gridW * 0.5, gridH * 0.5, 16, 4, 1.0);
      stampRing(scalar, gridW, gridH, gridW * 0.28, gridH * 0.3, 11, 2, 0.9);
      stampRing(scalar, gridW, gridH, gridW * 0.72, gridH * 0.7, 11, 2, 0.9);
    }

    for (let i = 0; i < cells; i++) {
      const v = scalar[i];
      state0[i * 4 + 0] = v;
      state0[i * 4 + 1] = v * 0.25;
      state0[i * 4 + 2] = 0.5;
      state0[i * 4 + 3] = v;
    }
  } else {
    if (preset.seedType === 'nca-colony') {
      plantNcaSeed(state0, state1, gridW, gridH, gridW * 0.34, gridH * 0.42);
      plantNcaSeed(state0, state1, gridW, gridH, gridW * 0.66, gridH * 0.42);
      plantNcaSeed(state0, state1, gridW, gridH, gridW * 0.5, gridH * 0.68);
    } else {
      plantNcaSeed(state0, state1, gridW, gridH, gridW * 0.5, gridH * 0.5);
      plantNcaSeed(state0, state1, gridW, gridH, gridW * 0.28, gridH * 0.34);
      plantNcaSeed(state0, state1, gridW, gridH, gridW * 0.72, gridH * 0.66);
    }
  }

  return { state0, state1 };
}
