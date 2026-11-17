import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  NCA_CHANNELS,
  leniaKernelShell,
  buildLeniaKernel,
  leniaGrowth,
  stepLeniaCPU,
  stepNcaCPU,
  packNcaWeights,
  stampPattern2D,
  validatePreset,
} from '../src/math.js';
import { PRESETS, ORBIUM_PATTERN, createInitialStates } from '../src/presets.js';
import * as shaders from '../src/shaders.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');

test('all presets pass parameter bounds and schema validation', () => {
  assert.ok(PRESETS.length >= 4 && PRESETS.length <= 6, 'Expected 4-6 curated presets');
  const ids = new Set();
  let leniaCount = 0;
  let ncaCount = 0;

  for (const preset of PRESETS) {
    const errors = validatePreset(preset);
    assert.deepEqual(errors, [], `Preset ${preset.id} validation errors: ${errors.join('; ')}`);
    assert.ok(!ids.has(preset.id), `Duplicate preset id: ${preset.id}`);
    ids.add(preset.id);
    if (preset.mode === 'lenia') leniaCount++;
    if (preset.mode === 'nca') ncaCount++;
  }
  assert.ok(leniaCount >= 2, 'Expected at least 2 Lenia presets');
  assert.ok(ncaCount >= 2, 'Expected at least 2 Self-Healing NCA presets');
});

test('Lenia kernel shell and 2D kernel normalization', () => {
  assert.equal(leniaKernelShell(0), 0);
  assert.equal(leniaKernelShell(1), 0);
  assert.equal(leniaKernelShell(-0.2), 0);
  assert.ok(Math.abs(leniaKernelShell(0.5) - 1.0) < 1e-12, 'Kernel shell peaks at 1.0 at r=0.5');

  for (const [R, beta] of [
    [13, [1, 0, 0]],
    [13, [1, 0.5, 0]],
    [14, [1, 0.75, 0.25]],
  ]) {
    const { size, weights } = buildLeniaKernel(R, beta);
    assert.equal(size, 2 * R + 1);
    assert.equal(weights.length, size * size);
    let sum = 0;
    for (let i = 0; i < weights.length; i++) {
      assert.ok(weights[i] >= 0 && Number.isFinite(weights[i]));
      sum += weights[i];
    }
    assert.ok(Math.abs(sum - 1.0) < 1e-5, `Kernel weights must sum to 1, got ${sum}`);
  }
});

test('Lenia growth mapping peaks at +1 at mu and decays toward -1', () => {
  assert.ok(Math.abs(leniaGrowth(0.15, 0.15, 0.017) - 1.0) < 1e-12);
  assert.ok(leniaGrowth(0.9, 0.15, 0.017) < -0.99);
  assert.ok(leniaGrowth(0.0, 0.15, 0.017) < -0.99);
});

test('Orbium Lenia preset preserves mass and self-propels on toroidal grid', () => {
  const orbium = PRESETS.find((p) => p.id === 'orbium');
  assert.ok(orbium);
  const N = 64;
  let grid = new Float32Array(N * N);
  stampPattern2D(grid, N, ORBIUM_PATTERN, 28, 28, 0, N);

  const centroid = (arr) => {
    let m = 0;
    let cx = 0;
    let cy = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const v = arr[y * N + x];
        m += v;
        cx += x * v;
        cy += y * v;
      }
    }
    return { m, cx: cx / m, cy: cy / m };
  };

  const t0 = centroid(grid);
  for (let i = 0; i < 30; i++) {
    grid = stepLeniaCPU(grid, N, orbium);
  }
  const t30 = centroid(grid);

  assert.ok(t30.m > 50 && t30.m < 110, `Orbium mass should stay bounded, got ${t30.m}`);
  const displacement = Math.hypot(t30.cx - t0.cx, t30.cy - t0.cy);
  assert.ok(displacement > 8, `Orbium should glide at least 8px in 30 steps, moved ${displacement.toFixed(2)}`);
});

test('Self-Healing Neural CA grows from a single seed and regenerates after slicing', () => {
  const ncaPresets = PRESETS.filter((p) => p.mode === 'nca');
  assert.ok(ncaPresets.length >= 2);

  for (const preset of ncaPresets) {
    const N = 28;
    let state = new Float32Array(N * N * NCA_CHANNELS);
    const mid = (Math.floor(N / 2) * N + Math.floor(N / 2)) * NCA_CHANNELS;
    for (let c = 3; c < NCA_CHANNELS; c++) state[mid + c] = 1.0;

    let rng = 424242;
    const nextMask = () => {
      const m = new Uint8Array(N * N);
      for (let i = 0; i < m.length; i++) {
        rng = (rng * 1664525 + 1013904223) >>> 0;
        m[i] = rng / 4294967296 < 0.5 ? 1 : 0;
      }
      return m;
    };

    const alphaMass = (st) => {
      let sum = 0;
      for (let i = 0; i < N * N; i++) {
        sum += Math.min(1, Math.max(0, st[i * NCA_CHANNELS + 3]));
      }
      return sum;
    };

    for (let t = 0; t < 75; t++) {
      state = stepNcaCPU(state, N, preset.weights, preset, nextMask());
    }
    const grownMass = alphaMass(state);
    assert.ok(grownMass > 80, `${preset.id} should grow from 1 pixel, mass=${grownMass.toFixed(1)}`);

    // Slice half the organism away
    for (let y = 0; y < N; y++) {
      for (let x = Math.floor(N / 2); x < N; x++) {
        const base = (y * N + x) * NCA_CHANNELS;
        for (let c = 0; c < NCA_CHANNELS; c++) state[base + c] = 0;
      }
    }
    const slicedMass = alphaMass(state);
    assert.ok(slicedMass < grownMass * 0.75, 'Slicing must remove significant mass');

    for (let t = 0; t < 75; t++) {
      state = stepNcaCPU(state, N, preset.weights, preset, nextMask());
    }
    const healedMass = alphaMass(state);
    assert.ok(
      healedMass > slicedMass * 1.25,
      `${preset.id} must regenerate after slice (sliced=${slicedMass.toFixed(1)}, healed=${healedMass.toFixed(1)})`
    );
  }
});

test('NCA weight packer preserves exact flat array lengths', () => {
  const nca = PRESETS.find((p) => p.mode === 'nca');
  const packed = packNcaWeights(nca.weights);
  assert.equal(packed.w1Vec4.length, 640);
  assert.equal(packed.b1Vec4.length, 20);
  assert.equal(packed.w2Vec4.length, 160);
  assert.equal(packed.b2Vec4.length, 8);
});

test('Initial state generator produces valid non-empty Float32Arrays for all presets', () => {
  for (const preset of PRESETS) {
    const { state0, state1 } = createInitialStates(preset, 96, 64);
    assert.equal(state0.length, 96 * 64 * 4);
    assert.equal(state1.length, 96 * 64 * 4);
    let maxVal = 0;
    for (let i = 0; i < state0.length; i++) {
      assert.ok(Number.isFinite(state0[i]));
      assert.ok(Number.isFinite(state1[i]));
      if (state0[i] > maxVal) maxVal = state0[i];
    }
    assert.ok(maxVal > 0, `Preset ${preset.id} initial state should be non-empty`);
  }
});

test('GLSL shaders declare WebGL2 version 300 es, highp precision, and valid entrypoints', () => {
  const entries = [
    ['COMMON_VS', shaders.COMMON_VS, ['gl_Position']],
    ['LENIA_FS', shaders.LENIA_FS, ['out vec4 fragColor', 'kernelShell', 'uMu', 'uSigma', 'uDt']],
    ['NCA_FS', shaders.NCA_FS, ['layout(location = 0) out vec4 out0', 'layout(location = 1) out vec4 out1', 'uW1[160]']],
    ['BRUSH_FS', shaders.BRUSH_FS, ['uBrushUv', 'uBrushMode', 'uBrushRadiusPx']],
    ['DISPLAY_FS', shaders.DISPLAY_FS, ['uPalette', 'uMouseUv', 'paletteBiolum']],
    ['MASS_FS', shaders.MASS_FS, ['uState0', 'uSimMode']],
  ];

  for (const [name, src, tokens] of entries) {
    assert.ok(src.startsWith('#version 300 es'), `${name} must start with #version 300 es`);
    assert.ok(src.includes('precision highp float;'), `${name} must declare highp float`);
    assert.ok(src.includes('void main()'), `${name} must define main()`);
    for (const tok of tokens) {
      assert.ok(src.includes(tok), `${name} missing expected token: ${tok}`);
    }
  }
});

test('index.html references valid local stylesheet and ES module entrypoint', () => {
  const html = fs.readFileSync(path.join(rootDir, 'index.html'), 'utf8');
  assert.ok(html.includes('href="./styles.css"'));
  assert.ok(html.includes('type="module" src="./src/main.js"'));
  assert.ok(fs.existsSync(path.join(rootDir, 'styles.css')));
  assert.ok(fs.existsSync(path.join(rootDir, 'src/main.js')));
});
