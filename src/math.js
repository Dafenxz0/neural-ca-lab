/**
 * Pure simulation math, kernel builders, CPU reference steppers, and weight packers
 * shared between the WebGL2 frontend and the Node.js verification suite.
 */

export const NCA_CHANNELS = 8;
export const NCA_FEATURES = NCA_CHANNELS * 4; // 32
export const NCA_HIDDEN = 20;

export function leniaKernelShell(r) {
  if (r <= 0 || r >= 1) return 0;
  const denom = Math.max(1e-5, 4 * r * (1 - r));
  return Math.exp(4 - 4 / denom);
}

export function buildLeniaKernel(radius, beta = [1, 0, 0]) {
  const R = Math.max(1, Math.min(18, Math.round(radius)));
  const size = 2 * R + 1;
  const weights = new Float32Array(size * size);
  const shells = beta[2] > 0 ? 3 : beta[1] > 0 ? 2 : 1;

  let total = 0;
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const r = Math.hypot(dx, dy) / R;
      if (r <= 0 || r >= 1) continue;
      const rs = r * shells;
      const idx = Math.min(2, Math.floor(rs));
      const rf = rs - Math.floor(rs);
      const bw = beta[idx] || 0;
      if (bw <= 0) continue;
      const w = bw * leniaKernelShell(rf);
      weights[(dy + R) * size + (dx + R)] = w;
      total += w;
    }
  }

  if (total > 0) {
    for (let i = 0; i < weights.length; i++) {
      weights[i] /= total;
    }
  }
  return { radius: R, size, weights };
}

export function leniaGrowth(potential, mu, sigma) {
  const s = Math.max(0.001, sigma);
  const z = (potential - mu) / s;
  return 2 * Math.exp(-0.5 * z * z) - 1;
}

export function stepLeniaCPU(grid, gridSize, { R, beta, mu, sigma, dt }) {
  const { radius, size, weights } = buildLeniaKernel(R, beta);
  const next = new Float32Array(grid.length);

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      let potential = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const sy = (y + dy + gridSize) % gridSize;
        const kRow = (dy + radius) * size;
        for (let dx = -radius; dx <= radius; dx++) {
          const w = weights[kRow + (dx + radius)];
          if (w === 0) continue;
          const sx = (x + dx + gridSize) % gridSize;
          potential += w * grid[sy * gridSize + sx];
        }
      }
      const prev = grid[y * gridSize + x];
      const g = leniaGrowth(potential, mu, sigma);
      next[y * gridSize + x] = Math.min(1, Math.max(0, prev + dt * g));
    }
  }
  return next;
}

export function stepNcaCPU(state, gridSize, weights, params = {}, mask = null) {
  const C = NCA_CHANNELS;
  const F = NCA_FEATURES;
  const H = NCA_HIDDEN;
  const mu = params.mu ?? 0.15;
  const sigma = params.sigma ?? 0.05;
  const dt = params.dt ?? 0.1;

  const aliveThresh = Math.min(0.35, Math.max(0.03, 0.1 + (mu - 0.15) * 0.5));
  const gain = Math.min(1.8, Math.max(0.5, 1.0 + (sigma - 0.05) * 4.0));
  const stepScale = Math.min(2.2, Math.max(0.2, dt / 0.1));

  const next = new Float32Array(state.length);
  const feat = new Float32Array(F);
  const hidden = new Float32Array(H);
  const { w1, b1, w2, b2 } = weights;

  for (let y = 0; y < gridSize; y++) {
    const ym = (y - 1 + gridSize) % gridSize;
    const yp = (y + 1) % gridSize;
    for (let x = 0; x < gridSize; x++) {
      const xm = (x - 1 + gridSize) % gridSize;
      const xp = (x + 1) % gridSize;
      const idx = y * gridSize + x;

      const coords = [
        ym * gridSize + xm, ym * gridSize + x, ym * gridSize + xp,
        y * gridSize + xm,  idx,               y * gridSize + xp,
        yp * gridSize + xm, yp * gridSize + x, yp * gridSize + xp,
      ];

      let maxAlpha = 0;
      for (let k = 0; k < 9; k++) {
        const a = state[coords[k] * C + 3];
        if (a > maxAlpha) maxAlpha = a;
      }

      const sBase = idx * C;
      if (maxAlpha <= aliveThresh) {
        continue;
      }
      if (mask && mask[idx] === 0) {
        for (let c = 0; c < C; c++) next[sBase + c] = state[sBase + c];
        continue;
      }

      for (let c = 0; c < C; c++) {
        const v00 = state[coords[0] * C + c], v10 = state[coords[1] * C + c], v20 = state[coords[2] * C + c];
        const v01 = state[coords[3] * C + c], v11 = state[coords[4] * C + c], v21 = state[coords[5] * C + c];
        const v02 = state[coords[6] * C + c], v12 = state[coords[7] * C + c], v22 = state[coords[8] * C + c];

        feat[c * 4 + 0] = v11;
        feat[c * 4 + 1] = (-v00 + v20 - 2 * v01 + 2 * v21 - v02 + v22) * 0.125;
        feat[c * 4 + 2] = (-v00 - 2 * v10 - v20 + v02 + 2 * v12 + v22) * 0.125;
        feat[c * 4 + 3] = 0.25 * (v00 + v20 + v02 + v22) + 0.5 * (v10 + v01 + v21 + v12) - 3.0 * v11;
      }

      for (let h = 0; h < H; h++) {
        let sum = b1[h];
        for (let f = 0; f < F; f++) {
          sum += feat[f] * w1[f * H + h];
        }
        hidden[h] = sum > 0 ? sum * gain : 0;
      }

      for (let c = 0; c < C; c++) {
        let ds = b2[c];
        for (let h = 0; h < H; h++) {
          ds += hidden[h] * w2[h * C + c];
        }
        const v = state[sBase + c] + ds * stepScale;
        next[sBase + c] = Math.min(4, Math.max(-4, v));
      }
    }
  }
  return next;
}

export function packNcaWeights(weights) {
  const { w1, b1, w2, b2 } = weights;
  if (w1.length !== NCA_FEATURES * NCA_HIDDEN) {
    throw new Error(`Invalid w1 length: expected ${NCA_FEATURES * NCA_HIDDEN}, got ${w1.length}`);
  }
  if (b1.length !== NCA_HIDDEN) {
    throw new Error(`Invalid b1 length: expected ${NCA_HIDDEN}, got ${b1.length}`);
  }
  if (w2.length !== NCA_HIDDEN * NCA_CHANNELS) {
    throw new Error(`Invalid w2 length: expected ${NCA_HIDDEN * NCA_CHANNELS}, got ${w2.length}`);
  }
  if (b2.length !== NCA_CHANNELS) {
    throw new Error(`Invalid b2 length: expected ${NCA_CHANNELS}, got ${b2.length}`);
  }

  return {
    w1Vec4: new Float32Array(w1),
    b1Vec4: new Float32Array(b1),
    w2Vec4: new Float32Array(w2),
    b2Vec4: new Float32Array(b2),
  };
}

export function stampPattern2D(grid, gridW, pattern, cx, cy, rotate90 = 0, gridH = gridW) {
  const ph = pattern.length;
  const pw = pattern[0].length;
  const ox = Math.floor(cx - pw / 2);
  const oy = Math.floor(cy - ph / 2);

  for (let py = 0; py < ph; py++) {
    for (let px = 0; px < pw; px++) {
      let rx = px;
      let ry = py;
      if (rotate90 === 1) {
        rx = ph - 1 - py;
        ry = px;
      } else if (rotate90 === 2) {
        rx = pw - 1 - px;
        ry = ph - 1 - py;
      } else if (rotate90 === 3) {
        rx = py;
        ry = pw - 1 - px;
      }
      const gx = ((ox + rx) % gridW + gridW) % gridW;
      const gy = ((oy + ry) % gridH + gridH) % gridH;
      const val = pattern[py][px];
      if (val > grid[gy * gridW + gx]) {
        grid[gy * gridW + gx] = val;
      }
    }
  }
}

export function validatePreset(preset) {
  const errors = [];
  if (!preset || typeof preset !== 'object') errors.push('Preset must be an object');
  if (!preset.id || typeof preset.id !== 'string') errors.push('Missing preset id');
  if (!preset.name || typeof preset.name !== 'string') errors.push('Missing preset name');
  if (preset.mode !== 'lenia' && preset.mode !== 'nca') {
    errors.push(`Invalid mode: ${preset.mode}`);
  }
  if (typeof preset.mu !== 'number' || preset.mu < 0.01 || preset.mu > 0.8) {
    errors.push(`mu out of bounds: ${preset.mu}`);
  }
  if (typeof preset.sigma !== 'number' || preset.sigma < 0.002 || preset.sigma > 0.25) {
    errors.push(`sigma out of bounds: ${preset.sigma}`);
  }
  if (typeof preset.dt !== 'number' || preset.dt < 0.005 || preset.dt > 0.5) {
    errors.push(`dt out of bounds: ${preset.dt}`);
  }
  if (!Number.isInteger(preset.palette) || preset.palette < 0 || preset.palette > 4) {
    errors.push(`palette out of bounds: ${preset.palette}`);
  }
  if (preset.mode === 'lenia') {
    if (!Number.isInteger(preset.R) || preset.R < 3 || preset.R > 18) {
      errors.push(`Lenia R out of bounds: ${preset.R}`);
    }
    if (!Array.isArray(preset.beta) || preset.beta.length !== 3) {
      errors.push('Lenia beta must be a 3-element array');
    }
  } else if (preset.mode === 'nca') {
    if (!preset.weights) {
      errors.push('NCA preset missing weights');
    } else {
      try {
        packNcaWeights(preset.weights);
      } catch (err) {
        errors.push(err.message);
      }
    }
  }
  return errors;
}
