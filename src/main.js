import {
  createWebGL2Context,
  createProgram,
  createFullscreenQuad,
  createPingPongTarget,
  createReadbackTarget,
} from './gl.js';
import {
  COMMON_VS,
  LENIA_FS,
  NCA_FS,
  BRUSH_FS,
  DISPLAY_FS,
  MASS_FS,
} from './shaders.js';
import { packNcaWeights } from './math.js';
import { PRESETS, createInitialStates } from './presets.js';

function $(id) {
  return document.getElementById(id);
}

const ui = {
  canvas: $('sim-canvas'),
  modeBadge: $('mode-badge'),
  statFps: $('stat-fps'),
  statMass: $('stat-mass'),
  statStep: $('stat-step'),
  statGrid: $('stat-grid'),
  presetList: $('preset-list'),
  presetTag: $('preset-tag'),
  presetDesc: $('preset-desc'),
  toolSeed: $('tool-seed'),
  toolSlice: $('tool-slice'),
  sliderRadius: $('slider-radius'),
  valRadius: $('val-radius'),
  brushReadout: $('brush-readout'),
  sliderMu: $('slider-mu'),
  valMu: $('val-mu'),
  sliderSigma: $('slider-sigma'),
  valSigma: $('val-sigma'),
  sliderDt: $('slider-dt'),
  valDt: $('val-dt'),
  selectPalette: $('select-palette'),
  btnRestoreDna: $('btn-restore-dna'),
  btnPlay: $('btn-play'),
  btnStep: $('btn-step'),
  btnReset: $('btn-reset'),
  btnClear: $('btn-clear'),
  btnTogglePanel: $('btn-toggle-panel'),
  controlPanel: $('control-panel'),
  errorBanner: $('error-banner'),
};

function showError(msg) {
  ui.errorBanner.textContent = msg;
  ui.errorBanner.classList.remove('hidden');
}

function init() {
  let gl;
  let texConfig;
  try {
    ({ gl, texConfig } = createWebGL2Context(ui.canvas));
  } catch (err) {
    showError(err.message);
    return;
  }

  const quad = createFullscreenQuad(gl);
  const progLenia = createProgram(gl, COMMON_VS, LENIA_FS);
  const progNca = createProgram(gl, COMMON_VS, NCA_FS);
  const progBrush = createProgram(gl, COMMON_VS, BRUSH_FS);
  const progDisplay = createProgram(gl, COMMON_VS, DISPLAY_FS);
  const progMass = createProgram(gl, COMMON_VS, MASS_FS);
  const massTarget = createReadbackTarget(gl, 32);

  const state = {
    presetIndex: 0,
    preset: PRESETS[0],
    mu: PRESETS[0].mu,
    sigma: PRESETS[0].sigma,
    dt: PRESETS[0].dt,
    palette: PRESETS[0].palette,
    brushMode: 0, // 0 = seed, 1 = slice
    brushRadius: 16,
    running: true,
    stepCount: 0,
    gridW: 256,
    gridH: 256,
    pingPong: null,
    packedWeights: null,
    mouseUv: [0.5, 0.5],
    mouseActive: false,
    pointerDown: false,
    massPct: 0,
    fpsSmooth: 60,
    lastFrameMs: performance.now(),
  };

  function computeGridDimensions(mode) {
    const w = Math.max(320, ui.canvas.clientWidth || window.innerWidth);
    const h = Math.max(240, ui.canvas.clientHeight || window.innerHeight);
    const aspect = Math.min(2.4, Math.max(0.5, w / h));
    const baseH = mode === 'lenia' ? 224 : 144;
    const gridH = baseH;
    const gridW = Math.max(96, Math.min(480, Math.round(baseH * aspect)));
    return { gridW, gridH };
  }

  function rebuildSimulationTarget() {
    const { gridW, gridH } = computeGridDimensions(state.preset.mode);
    state.gridW = gridW;
    state.gridH = gridH;
    if (state.pingPong) state.pingPong.destroy();
    state.pingPong = createPingPongTarget(gl, gridW, gridH, 2, texConfig);
    ui.statGrid.textContent = `${gridW}×${gridH}`;
  }

  function uploadSeedState() {
    const { state0, state1 } = createInitialStates(state.preset, state.gridW, state.gridH);
    state.pingPong.uploadFloat32(0, state0);
    state.pingPong.uploadFloat32(1, state1);
    state.stepCount = 0;
    updateMassTelemetry();
  }

  function clearGridState() {
    const zeros = new Float32Array(state.gridW * state.gridH * 4);
    state.pingPong.uploadFloat32(0, zeros);
    state.pingPong.uploadFloat32(1, zeros);
    state.stepCount = 0;
    state.massPct = 0;
    ui.statMass.textContent = '0.0%';
    ui.statStep.textContent = '0';
  }

  function selectPreset(index) {
    const preset = PRESETS[index];
    if (!preset) return;
    state.presetIndex = index;
    state.preset = preset;
    state.mu = preset.mu;
    state.sigma = preset.sigma;
    state.dt = preset.dt;
    state.palette = preset.palette;
    state.packedWeights = preset.mode === 'nca' ? packNcaWeights(preset.weights) : null;

    rebuildSimulationTarget();
    uploadSeedState();
    syncUi();
  }

  function syncUi() {
    const p = state.preset;
    ui.modeBadge.textContent = p.kind;
    ui.modeBadge.classList.toggle('nca', p.mode === 'nca');
    ui.presetTag.textContent = `${state.presetIndex + 1} / ${PRESETS.length}`;
    ui.presetDesc.textContent = p.description;

    for (const btn of ui.presetList.querySelectorAll('.preset-btn')) {
      btn.classList.toggle('active', Number(btn.dataset.index) === state.presetIndex);
    }

    ui.sliderMu.value = String(state.mu);
    ui.valMu.textContent = state.mu.toFixed(3);
    ui.sliderSigma.value = String(state.sigma);
    ui.valSigma.textContent = state.sigma.toFixed(3);
    ui.sliderDt.value = String(state.dt);
    ui.valDt.textContent = state.dt.toFixed(3);
    ui.selectPalette.value = String(state.palette);

    ui.sliderRadius.value = String(state.brushRadius);
    ui.valRadius.textContent = String(state.brushRadius);
    ui.brushReadout.textContent = `R=${state.brushRadius}px`;

    ui.toolSeed.classList.toggle('active', state.brushMode === 0);
    ui.toolSlice.classList.toggle('active', state.brushMode === 1);
    ui.btnPlay.textContent = state.running ? 'Pause' : 'Play';
    ui.statStep.textContent = String(state.stepCount);
  }

  function buildPresetButtons() {
    ui.presetList.innerHTML = '';
    PRESETS.forEach((p, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-btn';
      btn.dataset.index = String(idx);
      btn.innerHTML = `<span>${idx + 1}. ${p.name}</span><span class="preset-kind">${p.kind}</span>`;
      btn.addEventListener('click', () => selectPreset(idx));
      ui.presetList.appendChild(btn);
    });
  }

  function stepLenia() {
    const p = state.preset;
    const { pingPong, gridW, gridH } = state;
    gl.bindFramebuffer(gl.FRAMEBUFFER, pingPong.write.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, gridW, gridH);
    gl.useProgram(progLenia);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pingPong.read.textures[0]);
    gl.uniform1i(gl.getUniformLocation(progLenia, 'uState'), 0);
    gl.uniform2i(gl.getUniformLocation(progLenia, 'uGridSize'), gridW, gridH);
    gl.uniform1i(gl.getUniformLocation(progLenia, 'uRadius'), p.R);
    gl.uniform3f(gl.getUniformLocation(progLenia, 'uBeta'), p.beta[0], p.beta[1], p.beta[2]);
    gl.uniform1f(gl.getUniformLocation(progLenia, 'uMu'), state.mu);
    gl.uniform1f(gl.getUniformLocation(progLenia, 'uSigma'), state.sigma);
    gl.uniform1f(gl.getUniformLocation(progLenia, 'uDt'), state.dt);

    quad.draw();
    pingPong.swap();
    state.stepCount += 1;
  }

  function stepNca() {
    const { pingPong, gridW, gridH, packedWeights } = state;
    gl.bindFramebuffer(gl.FRAMEBUFFER, pingPong.write.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, gridW, gridH);
    gl.useProgram(progNca);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pingPong.read.textures[0]);
    gl.uniform1i(gl.getUniformLocation(progNca, 'uState0'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, pingPong.read.textures[1]);
    gl.uniform1i(gl.getUniformLocation(progNca, 'uState1'), 1);

    gl.uniform2i(gl.getUniformLocation(progNca, 'uGridSize'), gridW, gridH);
    gl.uniform1f(gl.getUniformLocation(progNca, 'uSeed'), (state.stepCount % 10000) * 0.137 + 1.23);
    gl.uniform1f(gl.getUniformLocation(progNca, 'uMu'), state.mu);
    gl.uniform1f(gl.getUniformLocation(progNca, 'uSigma'), state.sigma);
    gl.uniform1f(gl.getUniformLocation(progNca, 'uDt'), state.dt);

    gl.uniform4fv(gl.getUniformLocation(progNca, 'uW1'), packedWeights.w1Vec4);
    gl.uniform4fv(gl.getUniformLocation(progNca, 'uB1'), packedWeights.b1Vec4);
    gl.uniform4fv(gl.getUniformLocation(progNca, 'uW2'), packedWeights.w2Vec4);
    gl.uniform4fv(gl.getUniformLocation(progNca, 'uB2'), packedWeights.b2Vec4);

    quad.draw();
    pingPong.swap();
    state.stepCount += 1;
  }

  function stepOnce() {
    if (state.preset.mode === 'lenia') {
      stepLenia();
    } else {
      stepNca();
    }
  }

  function applyBrush(u, v) {
    const { pingPong, gridW, gridH } = state;
    gl.bindFramebuffer(gl.FRAMEBUFFER, pingPong.write.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, gridW, gridH);
    gl.useProgram(progBrush);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pingPong.read.textures[0]);
    gl.uniform1i(gl.getUniformLocation(progBrush, 'uState0'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, pingPong.read.textures[1]);
    gl.uniform1i(gl.getUniformLocation(progBrush, 'uState1'), 1);

    gl.uniform2i(gl.getUniformLocation(progBrush, 'uGridSize'), gridW, gridH);
    gl.uniform2f(gl.getUniformLocation(progBrush, 'uBrushUv'), u, v);
    gl.uniform1f(gl.getUniformLocation(progBrush, 'uBrushRadiusPx'), state.brushRadius);
    gl.uniform1i(gl.getUniformLocation(progBrush, 'uBrushMode'), state.brushMode);
    gl.uniform1i(gl.getUniformLocation(progBrush, 'uSimMode'), state.preset.mode === 'lenia' ? 0 : 1);
    gl.uniform1f(gl.getUniformLocation(progBrush, 'uSeed'), performance.now() * 0.001);

    quad.draw();
    pingPong.swap();
  }

  function updateMassTelemetry() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, massTarget.fbo);
    gl.viewport(0, 0, massTarget.size, massTarget.size);
    gl.useProgram(progMass);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, state.pingPong.read.textures[0]);
    gl.uniform1i(gl.getUniformLocation(progMass, 'uState0'), 0);
    gl.uniform1i(gl.getUniformLocation(progMass, 'uSimMode'), state.preset.mode === 'lenia' ? 0 : 1);
    quad.draw();

    const avg = massTarget.sampleMass();
    state.massPct = avg * 100;
    ui.statMass.textContent = `${state.massPct.toFixed(1)}%`;
  }

  function resizeCanvasToDisplay() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(ui.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(ui.canvas.clientHeight * dpr));
    if (ui.canvas.width !== width || ui.canvas.height !== height) {
      ui.canvas.width = width;
      ui.canvas.height = height;
    }
  }

  function renderDisplay() {
    resizeCanvasToDisplay();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, ui.canvas.width, ui.canvas.height);
    gl.useProgram(progDisplay);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, state.pingPong.read.textures[0]);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'uState0'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, state.pingPong.read.textures[1]);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'uState1'), 1);

    gl.uniform2i(gl.getUniformLocation(progDisplay, 'uGridSize'), state.gridW, state.gridH);
    gl.uniform2f(gl.getUniformLocation(progDisplay, 'uResolution'), ui.canvas.width, ui.canvas.height);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'uSimMode'), state.preset.mode === 'lenia' ? 0 : 1);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'uPalette'), state.palette);
    gl.uniform2f(gl.getUniformLocation(progDisplay, 'uMouseUv'), state.mouseUv[0], state.mouseUv[1]);
    gl.uniform1f(gl.getUniformLocation(progDisplay, 'uBrushRadiusPx'), state.brushRadius);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'uBrushMode'), state.brushMode);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'uMouseActive'), state.mouseActive ? 1 : 0);

    quad.draw();
  }

  let frameCounter = 0;
  function tick(now) {
    const dtMs = Math.max(1, now - state.lastFrameMs);
    state.lastFrameMs = now;
    const instantFps = 1000 / dtMs;
    state.fpsSmooth = state.fpsSmooth * 0.9 + instantFps * 0.1;

    if (state.pointerDown) {
      applyBrush(state.mouseUv[0], state.mouseUv[1]);
    }

    if (state.running) {
      const subSteps = state.preset.mode === 'lenia' ? 2 : 1;
      for (let i = 0; i < subSteps; i++) {
        stepOnce();
      }
    }

    renderDisplay();

    frameCounter++;
    if (frameCounter % 10 === 0) {
      ui.statFps.textContent = state.fpsSmooth.toFixed(0);
      ui.statStep.textContent = String(state.stepCount);
      updateMassTelemetry();
    }

    requestAnimationFrame(tick);
  }

  function pointerToUv(ev) {
    const rect = ui.canvas.getBoundingClientRect();
    const u = Math.min(1, Math.max(0, (ev.clientX - rect.left) / Math.max(1, rect.width)));
    const v = Math.min(1, Math.max(0, 1 - (ev.clientY - rect.top) / Math.max(1, rect.height)));
    return [u, v];
  }

  ui.canvas.addEventListener('pointerdown', (ev) => {
    ui.canvas.setPointerCapture(ev.pointerId);
    state.pointerDown = true;
    state.mouseActive = true;
    state.mouseUv = pointerToUv(ev);
    applyBrush(state.mouseUv[0], state.mouseUv[1]);
  });

  ui.canvas.addEventListener('pointermove', (ev) => {
    state.mouseActive = true;
    state.mouseUv = pointerToUv(ev);
  });

  ui.canvas.addEventListener('pointerup', () => {
    state.pointerDown = false;
  });

  ui.canvas.addEventListener('pointerleave', () => {
    state.pointerDown = false;
    state.mouseActive = false;
  });

  ui.toolSeed.addEventListener('click', () => {
    state.brushMode = 0;
    syncUi();
  });

  ui.toolSlice.addEventListener('click', () => {
    state.brushMode = 1;
    syncUi();
  });

  ui.sliderRadius.addEventListener('input', (ev) => {
    state.brushRadius = Number(ev.target.value);
    syncUi();
  });

  ui.sliderMu.addEventListener('input', (ev) => {
    state.mu = Number(ev.target.value);
    ui.valMu.textContent = state.mu.toFixed(3);
  });

  ui.sliderSigma.addEventListener('input', (ev) => {
    state.sigma = Number(ev.target.value);
    ui.valSigma.textContent = state.sigma.toFixed(3);
  });

  ui.sliderDt.addEventListener('input', (ev) => {
    state.dt = Number(ev.target.value);
    ui.valDt.textContent = state.dt.toFixed(3);
  });

  ui.selectPalette.addEventListener('change', (ev) => {
    state.palette = Number(ev.target.value);
  });

  ui.btnRestoreDna.addEventListener('click', () => {
    state.mu = state.preset.mu;
    state.sigma = state.preset.sigma;
    state.dt = state.preset.dt;
    state.palette = state.preset.palette;
    syncUi();
  });

  ui.btnPlay.addEventListener('click', () => {
    state.running = !state.running;
    syncUi();
  });

  ui.btnStep.addEventListener('click', () => {
    state.running = false;
    stepOnce();
    updateMassTelemetry();
    syncUi();
  });

  ui.btnReset.addEventListener('click', () => {
    uploadSeedState();
    syncUi();
  });

  ui.btnClear.addEventListener('click', () => {
    clearGridState();
  });

  ui.btnTogglePanel.addEventListener('click', () => {
    ui.controlPanel.classList.toggle('collapsed');
  });

  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
    if (ev.code === 'Space') {
      ev.preventDefault();
      state.running = !state.running;
      syncUi();
    } else if (ev.key === 's' || ev.key === 'S') {
      state.running = false;
      stepOnce();
      updateMassTelemetry();
      syncUi();
    } else if (ev.key === 'r' || ev.key === 'R') {
      uploadSeedState();
      syncUi();
    } else if (ev.key === 'h' || ev.key === 'H') {
      ui.controlPanel.classList.toggle('collapsed');
    } else if (ev.key === '[') {
      state.brushRadius = Math.max(4, state.brushRadius - 2);
      syncUi();
    } else if (ev.key === ']') {
      state.brushRadius = Math.min(44, state.brushRadius + 2);
      syncUi();
    } else if (ev.key >= '1' && ev.key <= String(PRESETS.length)) {
      selectPreset(Number(ev.key) - 1);
    }
  });

  buildPresetButtons();
  selectPreset(0);
  requestAnimationFrame(tick);
}

init();
