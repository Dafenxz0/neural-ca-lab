# Neural-CA Lab

[![CI & Deploy GitHub Pages](https://github.com/Dafenxz0/neural-ca-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Dafenxz0/neural-ca-lab/actions/workflows/ci.yml)

**Live Demo:** [https://dafenxz0.github.io/neural-ca-lab/](https://dafenxz0.github.io/neural-ca-lab/)

An interactive WebGL2 GPU **Lenia Continuous Cellular Automata** & **Self-Healing Morphogenetic Neural Cellular Automata** laboratory running at 60 FPS in the browser with zero build step or runtime dependencies.

## Features

- **Dual GPU Simulation Engines (WebGL2 Ping-Pong `RGBA16F` + MRT)**
  - **Lenia Continuous CA**: Multi-shell concentric ring-kernel convolution ($R \le 18$, $\beta \in \mathbb{R}^3$) paired with Gaussian growth mapping producing self-propelling solitons (**Orbium Gliders**), budding membranes (**Hydrogeminium**), and oscillating wave colonies (**Synapsia Gyre**).
  - **Self-Healing Morphogenetic Neural CA**: 8-channel Multiple Render Target (`COLOR_ATTACHMENT0` + `COLOR_ATTACHMENT1`) state grid with Identity, Sobel-X, Sobel-Y, and 9-point Laplacian perception ($32 \to 20 \text{ ReLU} \to 8$ residual MLP) trained with damage augmentation (**Planarian NCA**, **Cryo-Lotus NCA**).
- **Interactive Micromanipulator**
  - **Seed / Spawn**: Plant new Lenia solitons or single-pixel Neural CA seeds anywhere on the toroidal manifold.
  - **Slice / Damage**: Carve chunks out of living organisms and watch Neural CA specimens regenerate missing organs or Lenia colonies split and re-equilibrate.
- **DNA / Kernel Mutator & Optics**
  - Real-time sliders for growth center $\mu$, width $\sigma$, integration step $dt$, brush radius, 5 scientific false-color/RGB palettes, pause/step/reset, and keyboard shortcuts (`Space`, `S`, `R`, `H`, `1–5`, `[`, `]`).

## Math & Shader Overview

### 1. Lenia (`LENIA_FS`)
For state field $A(x,y) \in [0,1]$, concentric ring kernel $K(r) = \sum_i \beta_i \exp\!\left(4 - \frac{4}{4 r_i (1 - r_i)}\right)$ normalized to $\sum K = 1$:

$$U = K * A, \qquad G(U; \mu, \sigma) = 2 \exp\!\left(-\frac{(U - \mu)^2}{2\sigma^2}\right) - 1, \qquad A^{t+dt} = \mathrm{clamp}\!\left(A^t + dt \cdot G(U),\, 0,\, 1\right)$$

### 2. Self-Healing Morphogenetic Neural CA (`NCA_FS`)
Each cell stores 8 channels across two `RGBA16F` textures (`[R, G, B, A]` and 4 hidden morphogen channels `[h4..h7]`). At each step, active cells ($\max_{3\times 3} \alpha > \text{threshold}$) perceive 32 local features via discrete `Identity`, `Sobel-X`, `Sobel-Y`, and `9-point Laplacian` stencils, pass them through a trained 2-layer MLP (`32 -> 20 ReLU -> 8`), and apply stochastic residual updates ($p = 0.5$).

## Project Structure

```text
index.html                  Full-bleed WebGL2 microscope viewport & HUD
styles.css                  Dark laboratory monospace UI styling
src/
  gl.js                     WebGL2 context, RGBA16F ping-pong FBOs, 32x32 mass readback
  math.js                   Kernel builders, CPU reference steppers, weight packer, validator
  shaders.js                GLSL ES 3.00 shaders (Lenia, MRT NCA, Brush, Display, Mass)
  nca_weights.js            Trained 32->20->8 weights for Planarian & Cryo-Lotus
  presets.js                5 curated specimens & toroidal seed generators
  main.js                   Simulation loop, pointer micromanipulator, keyboard & HUD bindings
tests/
  neural_ca.test.js         Node.js test suite (bounds, Lenia glider motion, NCA regeneration, GLSL sanity)
```

## Local Development & Tests

```bash
# Run ES module syntax checks and unit/simulation tests (Node.js >= 20)
npm run check
npm test

# Serve locally (any static HTTP server works)
npx serve .
```
