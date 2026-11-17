/**
 * Minimal WebGL2 utilities for ping-pong float FBO simulation and MRT rendering.
 */

export function createWebGL2Context(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: 'high-performance',
  });
  if (!gl) {
    throw new Error('WebGL2 is not supported in this browser.');
  }

  const extFloat = gl.getExtension('EXT_color_buffer_float');
  const extHalf = gl.getExtension('EXT_color_buffer_half_float');
  if (!extFloat && !extHalf) {
    throw new Error('Required extension EXT_color_buffer_float or EXT_color_buffer_half_float is unavailable.');
  }

  // RGBA16F is universally renderable with either extension and supports LINEAR filtering in WebGL2.
  const texConfig = {
    internalFormat: gl.RGBA16F,
    format: gl.RGBA,
    type: gl.HALF_FLOAT,
    hasFloat32: Boolean(extFloat),
  };

  return { gl, texConfig };
}

export function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(`Shader compilation failed:\n${info}`);
  }
  return shader;
}

export function createProgram(gl, vsSource, fsSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) || 'Unknown program link error';
    gl.deleteProgram(program);
    throw new Error(`Program link failed:\n${info}`);
  }
  return program;
}

export function createFullscreenQuad(gl) {
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW
  );
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return {
    vao,
    draw() {
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
    destroy() {
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(vbo);
    },
  };
}

function createFloatTexture(gl, width, height, texConfig, filter = gl.NEAREST) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    texConfig.internalFormat,
    width,
    height,
    0,
    texConfig.format,
    texConfig.type,
    null
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

export function createPingPongTarget(gl, width, height, attachmentCount, texConfig) {
  function makeTarget() {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const textures = [];
    const drawBuffers = [];
    for (let i = 0; i < attachmentCount; i++) {
      const tex = createFloatTexture(gl, width, height, texConfig, gl.NEAREST);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0 + i,
        gl.TEXTURE_2D,
        tex,
        0
      );
      textures.push(tex);
      drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(drawBuffers);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Incomplete framebuffer (status 0x${status.toString(16)})`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, textures, drawBuffers };
  }

  let read = makeTarget();
  let write = makeTarget();

  return {
    width,
    height,
    attachmentCount,
    get read() {
      return read;
    },
    get write() {
      return write;
    },
    swap() {
      const tmp = read;
      read = write;
      write = tmp;
    },
    uploadFloat32(attachmentIndex, floatData) {
      const halfOrFloat =
        texConfig.type === gl.HALF_FLOAT ? float32ToFloat16Array(floatData) : floatData;
      for (const target of [read, write]) {
        gl.bindTexture(gl.TEXTURE_2D, target.textures[attachmentIndex]);
        gl.texSubImage2D(
          gl.TEXTURE_2D,
          0,
          0,
          0,
          width,
          height,
          texConfig.format,
          texConfig.type,
          halfOrFloat
        );
      }
      gl.bindTexture(gl.TEXTURE_2D, null);
    },
    destroy() {
      for (const target of [read, write]) {
        gl.deleteFramebuffer(target.fbo);
        for (const t of target.textures) gl.deleteTexture(t);
      }
    },
  };
}

export function createReadbackTarget(gl, size = 32) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const pixels = new Uint8Array(size * size * 4);

  return {
    size,
    fbo,
    pixels,
    sampleMass() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      let sum = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        sum += pixels[i] / 255;
      }
      return sum / (size * size);
    },
    destroy() {
      gl.deleteFramebuffer(fbo);
      gl.deleteTexture(tex);
    },
  };
}

// IEEE-754 float32 -> float16 converter for uploading RGBA16F textures in WebGL2
export function float32ToFloat16Array(src) {
  const out = new Uint16Array(src.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < src.length; i++) {
    f32[0] = src[i];
    const x = u32[0];
    const sign = (x >> 16) & 0x8000;
    let exp = ((x >> 23) & 0xff) - 112;
    let mant = x & 0x7fffff;
    if (exp <= 0) {
      if (exp < -10) {
        out[i] = sign;
      } else {
        mant = (mant | 0x800000) >> (1 - exp);
        out[i] = sign | (mant >> 13);
      }
    } else if (exp === 143) {
      out[i] = sign | 0x7c00 | (mant ? 1 : 0);
    } else if (exp > 30) {
      out[i] = sign | 0x7c00;
    } else {
      out[i] = sign | (exp << 10) | (mant >> 13);
    }
  }
  return out;
}
