export const COMMON_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aPos;
out vec2 vUv;

void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

export const LENIA_FS = `#version 300 es
precision highp float;

uniform sampler2D uState;
uniform ivec2 uGridSize;
uniform int uRadius;
uniform vec3 uBeta;
uniform float uMu;
uniform float uSigma;
uniform float uDt;

in vec2 vUv;
out vec4 fragColor;

float kernelShell(float r) {
  float denom = max(1e-5, 4.0 * r * (1.0 - r));
  return exp(4.0 - 4.0 / denom);
}

void main() {
  ivec2 coord = ivec2(floor(vUv * vec2(uGridSize)));
  vec4 center = texelFetch(uState, coord, 0);

  float shells = uBeta.z > 0.0 ? 3.0 : (uBeta.y > 0.0 ? 2.0 : 1.0);
  float rMax = float(uRadius);

  float weightedSum = 0.0;
  float weightTotal = 0.0;

  for (int dy = -18; dy <= 18; dy++) {
    if (abs(dy) > uRadius) continue;
    for (int dx = -18; dx <= 18; dx++) {
      if (abs(dx) > uRadius) continue;
      float dist = length(vec2(float(dx), float(dy)));
      float r = dist / rMax;
      if (r <= 0.0 || r >= 1.0) continue;

      float rs = r * shells;
      int idx = int(floor(rs));
      float rf = fract(rs);
      float bw = idx == 0 ? uBeta.x : (idx == 1 ? uBeta.y : uBeta.z);
      if (bw <= 0.0) continue;

      float w = bw * kernelShell(rf);
      ivec2 sampleCoord = (coord + ivec2(dx, dy) + uGridSize) % uGridSize;
      float val = texelFetch(uState, sampleCoord, 0).r;
      weightedSum += w * val;
      weightTotal += w;
    }
  }

  float potential = weightTotal > 0.0 ? weightedSum / weightTotal : 0.0;
  float z = (potential - uMu) / max(0.001, uSigma);
  float growth = 2.0 * exp(-0.5 * z * z) - 1.0;
  float nextA = clamp(center.r + uDt * growth, 0.0, 1.0);
  float trace = clamp(center.a * 0.94 + nextA * 0.06, 0.0, 1.0);

  fragColor = vec4(nextA, potential, growth * 0.5 + 0.5, trace);
}
`;

export const NCA_FS = `#version 300 es
precision highp float;

uniform sampler2D uState0;
uniform sampler2D uState1;
uniform ivec2 uGridSize;
uniform float uSeed;
uniform float uMu;
uniform float uSigma;
uniform float uDt;

uniform vec4 uW1[160];
uniform vec4 uB1[5];
uniform vec4 uW2[40];
uniform vec4 uB2[2];

in vec2 vUv;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;

float hash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  ivec2 coord = ivec2(floor(vUv * vec2(uGridSize)));

  // Alive mask threshold modulated gently by mu mutator
  float aliveThresh = clamp(0.10 + (uMu - 0.15) * 0.5, 0.03, 0.35);

  vec4 c0[9];
  vec4 c1[9];
  float maxAlpha = 0.0;
  int k = 0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 sc = (coord + ivec2(dx, dy) + uGridSize) % uGridSize;
      vec4 s0 = texelFetch(uState0, sc, 0);
      vec4 s1 = texelFetch(uState1, sc, 0);
      c0[k] = s0;
      c1[k] = s1;
      maxAlpha = max(maxAlpha, s0.a);
      k++;
    }
  }

  if (maxAlpha <= aliveThresh) {
    out0 = vec4(0.0);
    out1 = vec4(0.0);
    return;
  }

  // Stochastic update gate (p = 0.5)
  if (hash13(vec3(vec2(coord), uSeed)) >= 0.5) {
    out0 = c0[4];
    out1 = c1[4];
    return;
  }

  // Perception: Identity, Sobel-X, Sobel-Y, 9-point Laplacian
  vec4 id0 = c0[4];
  vec4 id1 = c1[4];

  vec4 sx0 = (-c0[0] + c0[2] - 2.0 * c0[3] + 2.0 * c0[5] - c0[6] + c0[8]) * 0.125;
  vec4 sx1 = (-c1[0] + c1[2] - 2.0 * c1[3] + 2.0 * c1[5] - c1[6] + c1[8]) * 0.125;

  vec4 sy0 = (-c0[0] - 2.0 * c0[1] - c0[2] + c0[6] + 2.0 * c0[7] + c0[8]) * 0.125;
  vec4 sy1 = (-c1[0] - 2.0 * c1[1] - c1[2] + c1[6] + 2.0 * c1[7] + c1[8]) * 0.125;

  vec4 lap0 = 0.25 * (c0[0] + c0[2] + c0[6] + c0[8]) + 0.5 * (c0[1] + c0[3] + c0[5] + c0[7]) - 3.0 * c0[4];
  vec4 lap1 = 0.25 * (c1[0] + c1[2] + c1[6] + c1[8]) + 0.5 * (c1[1] + c1[3] + c1[5] + c1[7]) - 3.0 * c1[4];

  float feat[32];
  for (int c = 0; c < 4; c++) {
    feat[c * 4 + 0] = id0[c];
    feat[c * 4 + 1] = sx0[c];
    feat[c * 4 + 2] = sy0[c];
    feat[c * 4 + 3] = lap0[c];

    feat[(c + 4) * 4 + 0] = id1[c];
    feat[(c + 4) * 4 + 1] = sx1[c];
    feat[(c + 4) * 4 + 2] = sy1[c];
    feat[(c + 4) * 4 + 3] = lap1[c];
  }

  vec4 hidden[5];
  for (int hg = 0; hg < 5; hg++) {
    hidden[hg] = uB1[hg];
  }
  for (int f = 0; f < 32; f++) {
    float fv = feat[f];
    int base = f * 5;
    hidden[0] += uW1[base + 0] * fv;
    hidden[1] += uW1[base + 1] * fv;
    hidden[2] += uW1[base + 2] * fv;
    hidden[3] += uW1[base + 3] * fv;
    hidden[4] += uW1[base + 4] * fv;
  }

  float gain = clamp(1.0 + (uSigma - 0.05) * 4.0, 0.5, 1.8);
  vec4 ds0 = uB2[0];
  vec4 ds1 = uB2[1];

  for (int hg = 0; hg < 5; hg++) {
    vec4 act = max(vec4(0.0), hidden[hg]) * gain;
    int hBase = hg * 4;
    ds0 += uW2[(hBase + 0) * 2 + 0] * act.x;
    ds1 += uW2[(hBase + 0) * 2 + 1] * act.x;

    ds0 += uW2[(hBase + 1) * 2 + 0] * act.y;
    ds1 += uW2[(hBase + 1) * 2 + 1] * act.y;

    ds0 += uW2[(hBase + 2) * 2 + 0] * act.z;
    ds1 += uW2[(hBase + 2) * 2 + 1] * act.z;

    ds0 += uW2[(hBase + 3) * 2 + 0] * act.w;
    ds1 += uW2[(hBase + 3) * 2 + 1] * act.w;
  }

  float stepScale = clamp(uDt / 0.10, 0.2, 2.2);
  out0 = clamp(c0[4] + ds0 * stepScale, vec4(-4.0), vec4(4.0));
  out1 = clamp(c1[4] + ds1 * stepScale, vec4(-4.0), vec4(4.0));
}
`;

export const BRUSH_FS = `#version 300 es
precision highp float;

uniform sampler2D uState0;
uniform sampler2D uState1;
uniform ivec2 uGridSize;
uniform vec2 uBrushUv;
uniform float uBrushRadiusPx;
uniform int uBrushMode; // 0 = seed, 1 = slice
uniform int uSimMode;   // 0 = lenia, 1 = nca
uniform float uSeed;

in vec2 vUv;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  ivec2 coord = ivec2(floor(vUv * vec2(uGridSize)));
  vec4 s0 = texelFetch(uState0, coord, 0);
  vec4 s1 = texelFetch(uState1, coord, 0);

  vec2 delta = (vUv - uBrushUv) * vec2(uGridSize);
  // Toroidal shortest distance on periodic canvas
  delta = mod(delta + vec2(uGridSize) * 0.5, vec2(uGridSize)) - vec2(uGridSize) * 0.5;
  float dist = length(delta);

  if (dist <= uBrushRadiusPx) {
    if (uBrushMode == 1) {
      // Slice / Damage: clean excision
      out0 = vec4(0.0);
      out1 = vec4(0.0);
      return;
    }

    if (uSimMode == 0) {
      // Lenia soft ring/soliton blob seed
      float rn = dist / max(1.0, uBrushRadiusPx);
      float ring = exp(-pow((rn - 0.35) / 0.35, 2.0));
      float noise = 0.72 + 0.28 * hash12(vec2(coord) + uSeed * 17.0);
      float val = clamp(max(s0.r, ring * noise), 0.0, 1.0);
      out0 = vec4(val, s0.g, s0.b, max(s0.a, val));
      out1 = s1;
      return;
    } else {
      // NCA seed: plant a compact seed core
      if (dist <= clamp(uBrushRadiusPx * 0.12, 0.9, 2.2)) {
        out0 = vec4(0.0, 0.0, 0.0, 1.0);
        out1 = vec4(1.0, 1.0, 1.0, 1.0);
        return;
      }
    }
  }

  out0 = s0;
  out1 = s1;
}
`;

export const DISPLAY_FS = `#version 300 es
precision highp float;

uniform sampler2D uState0;
uniform sampler2D uState1;
uniform ivec2 uGridSize;
uniform vec2 uResolution;
uniform int uSimMode;   // 0 = lenia, 1 = nca
uniform int uPalette;
uniform vec2 uMouseUv;
uniform float uBrushRadiusPx;
uniform int uBrushMode;
uniform int uMouseActive;

in vec2 vUv;
out vec4 fragColor;

vec3 paletteBiolum(float v, float pot, float trace) {
  vec3 bg = vec3(0.020, 0.032, 0.055);
  vec3 c1 = vec3(0.03, 0.35, 0.55);
  vec3 c2 = vec3(0.18, 0.90, 0.78);
  vec3 c3 = vec3(0.96, 0.98, 0.72);
  vec3 col = mix(bg, c1, smoothstep(0.0, 0.28, v));
  col = mix(col, c2, smoothstep(0.22, 0.68, v));
  col = mix(col, c3, smoothstep(0.65, 1.0, v));
  col += vec3(0.04, 0.14, 0.22) * smoothstep(0.05, 0.6, trace) * (1.0 - v);
  col += vec3(0.0, 0.08, 0.12) * pot * 0.35;
  return col;
}

vec3 paletteViridis(float t) {
  const vec3 c0 = vec3(0.04, 0.03, 0.08);
  const vec3 c1 = vec3(0.27, 0.01, 0.33);
  const vec3 c2 = vec3(0.13, 0.57, 0.55);
  const vec3 c3 = vec3(0.99, 0.91, 0.14);
  vec3 col = mix(c0, c1, smoothstep(0.0, 0.25, t));
  col = mix(col, c2, smoothstep(0.25, 0.65, t));
  return mix(col, c3, smoothstep(0.65, 1.0, t));
}

vec3 paletteThermal(float t) {
  const vec3 c0 = vec3(0.02, 0.02, 0.04);
  const vec3 c1 = vec3(0.55, 0.06, 0.24);
  const vec3 c2 = vec3(0.95, 0.45, 0.08);
  const vec3 c3 = vec3(1.00, 0.97, 0.75);
  vec3 col = mix(c0, c1, smoothstep(0.0, 0.32, t));
  col = mix(col, c2, smoothstep(0.32, 0.70, t));
  return mix(col, c3, smoothstep(0.70, 1.0, t));
}

vec3 palettePhase(float v, float pot, float growth) {
  float edge = clamp(abs(growth - 0.5) * 3.0, 0.0, 1.0);
  vec3 base = vec3(0.03, 0.05, 0.08) + vec3(0.12, 0.22, 0.32) * pot;
  vec3 pos = vec3(0.22, 0.92, 0.80) * v;
  vec3 neg = vec3(0.96, 0.55, 0.18) * edge * 0.55;
  return clamp(base + pos + neg, 0.0, 1.0);
}

void main() {
  // Subtle microscope slide background with faint coordinate grid
  vec2 gridLines = abs(fract(vUv * 16.0 - 0.5) - 0.5) / fwidth(vUv * 16.0);
  float line = 1.0 - min(min(gridLines.x, gridLines.y), 1.0);
  vec3 slideBg = vec3(0.022, 0.033, 0.052) + vec3(0.014, 0.024, 0.038) * line;

  // Smooth bilinear lookup manually using texelFetch so it works cleanly regardless of texture filtering
  vec2 texCoord = vUv * vec2(uGridSize) - 0.5;
  ivec2 p0 = ivec2(floor(texCoord));
  vec2 f = fract(texCoord);
  ivec2 p00 = (p0 + uGridSize) % uGridSize;
  ivec2 p10 = (p0 + ivec2(1, 0) + uGridSize) % uGridSize;
  ivec2 p01 = (p0 + ivec2(0, 1) + uGridSize) % uGridSize;
  ivec2 p11 = (p0 + ivec2(1, 1) + uGridSize) % uGridSize;

  vec4 s00 = texelFetch(uState0, p00, 0);
  vec4 s10 = texelFetch(uState0, p10, 0);
  vec4 s01 = texelFetch(uState0, p01, 0);
  vec4 s11 = texelFetch(uState0, p11, 0);
  vec4 s0 = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);

  vec4 h00 = texelFetch(uState1, p00, 0);
  vec4 h10 = texelFetch(uState1, p10, 0);
  vec4 h01 = texelFetch(uState1, p01, 0);
  vec4 h11 = texelFetch(uState1, p11, 0);
  vec4 s1 = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);

  vec3 col = slideBg;

  if (uSimMode == 0) {
    float v = clamp(s0.r, 0.0, 1.0);
    float pot = clamp(s0.g, 0.0, 1.0);
    float growth = clamp(s0.b, 0.0, 1.0);
    float trace = clamp(s0.a, 0.0, 1.0);

    if (uPalette == 0) col = paletteBiolum(v, pot, trace);
    else if (uPalette == 1) col = paletteViridis(v);
    else if (uPalette == 2) col = paletteThermal(v);
    else if (uPalette == 3) col = palettePhase(v, pot, growth);
    else col = mix(slideBg, vec3(v, pot, growth), max(v, pot * 0.4));
  } else {
    float alpha = clamp(s0.a, 0.0, 1.0);
    vec3 orgRgb = clamp(s0.rgb, 0.0, 1.0);
    float morphSignal = clamp(length(s1.rgb) * 0.45 + alpha * 0.65, 0.0, 1.0);

    if (uPalette == 4 || uPalette == 0) {
      // Cellular RGB / Bioluminescent overlay
      vec3 glow = uPalette == 0 ? vec3(0.05, 0.22, 0.28) * clamp(length(s1) * 0.35, 0.0, 1.0) : vec3(0.0);
      col = mix(slideBg + glow * 0.4, orgRgb, smoothstep(0.05, 0.75, alpha));
    } else if (uPalette == 1) {
      col = mix(slideBg, paletteViridis(morphSignal), smoothstep(0.04, 0.6, alpha));
    } else if (uPalette == 2) {
      col = mix(slideBg, paletteThermal(morphSignal), smoothstep(0.04, 0.6, alpha));
    } else {
      col = mix(slideBg, vec3(0.25, 0.9, 0.75) * alpha + vec3(0.95, 0.45, 0.25) * abs(s1.x) * 0.35, smoothstep(0.03, 0.5, alpha));
    }
  }

  // Brush reticle in screen pixels
  if (uMouseActive == 1) {
    vec2 cellDelta = (vUv - uMouseUv) * vec2(uGridSize);
    cellDelta = mod(cellDelta + vec2(uGridSize) * 0.5, vec2(uGridSize)) - vec2(uGridSize) * 0.5;
    float d = length(cellDelta);
    float ringDist = abs(d - uBrushRadiusPx);
    float aa = max(0.75, float(uGridSize.x) / max(1.0, uResolution.x) * 1.5);
    float ring = 1.0 - smoothstep(0.0, aa, ringDist);
    vec3 ringColor = uBrushMode == 1 ? vec3(0.96, 0.62, 0.04) : vec3(0.22, 0.91, 0.78);
    col = mix(col, ringColor, ring * 0.85);
  }

  fragColor = vec4(col, 1.0);
}
`;

export const MASS_FS = `#version 300 es
precision highp float;

uniform sampler2D uState0;
uniform int uSimMode;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec4 s = texture(uState0, vUv);
  float val = uSimMode == 0 ? clamp(s.r, 0.0, 1.0) : clamp(s.a, 0.0, 1.0);
  fragColor = vec4(val, val, val, 1.0);
}
`;
