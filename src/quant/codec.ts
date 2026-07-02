// int8 scalar quantization codec ( TurboQuant step 2).
//
// After rotation, each coordinate is quantized to a signed 8-bit value using the
// exact semantics of WGSL `pack4x8snorm` / `unpack4x8snorm`, so the GPU can
// dequantize 4 coordinates per instruction with a single hardware unpack:
//
//   byte b in [-127,127]  ->  unpack4x8snorm gives  max(b/127, -1)
//   dequantized value     ≈  (b / 127) * scale
//
// We use a per-row `scale` = max |coord| so nothing clips (lossless dynamic
// range), and store 4 bytes packed little-endian per u32 (component 0 = low byte),
// matching the WGSL layout. This yields ~4× less memory than fp32 and feeds the
// asymmetric kernel (fp32 query × int8 corpus).

export interface QuantizedRow {
  /** paddedDim/4 packed words (4 snorm bytes each). */
  words: Uint32Array;
  /** Per-row dequantization scale (max |coord|). */
  scale: number;
}

function snormByte(v: number): number {
  // clamp to [-1,1] then map to [-127,127] (−128 unused to mirror unpack4x8snorm)
  const c = v < -1 ? -1 : v > 1 ? 1 : v;
  let b = Math.round(c * 127);
  if (b < -127) b = -127;
  if (b > 127) b = 127;
  return b & 0xff; // two's complement low byte
}

/** Quantize one rotated vector (length paddedDim, multiple of 4). */
export function quantizeRow(rotated: Float32Array): QuantizedRow {
  const n = rotated.length;
  if (n % 4 !== 0) throw new Error('paddedDim must be a multiple of 4');

  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(rotated[i]!);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs > 0 ? maxAbs : 1;
  const inv = 1 / scale;

  const words = new Uint32Array(n / 4);
  for (let g = 0; g < words.length; g++) {
    const o = g * 4;
    const b0 = snormByte(rotated[o]! * inv);
    const b1 = snormByte(rotated[o + 1]! * inv);
    const b2 = snormByte(rotated[o + 2]! * inv);
    const b3 = snormByte(rotated[o + 3]! * inv);
    words[g] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) >>> 0;
  }
  return { words, scale };
}

/** Reference dequantization (CPU) — used for validation, mirrors unpack4x8snorm. */
export function dequantRow(words: Uint32Array, scale: number): Float32Array {
  const out = new Float32Array(words.length * 4);
  for (let g = 0; g < words.length; g++) {
    const w = words[g]!;
    for (let j = 0; j < 4; j++) {
      let b = (w >>> (j * 8)) & 0xff;
      if (b > 127) b -= 256; // sign-extend
      out[g * 4 + j] = Math.max(b / 127, -1) * scale;
    }
  }
  return out;
}

// ---- 4-bit (sub-byte) variant -------------------------------------------------
// Same idea as int8 but each coordinate is a signed 4-bit value in [-7,7], so 8
// coords pack into one u32 (half the bytes of int8, ~6× less than fp32 after the
// 768→1024 pad). There is no hardware unpack for 4-bit, so the kernel extracts
// nibbles by hand; raw recall drops vs int8, which is exactly where the exact
// fp32 re-rank earns its place.

function snormNibble(v: number): number {
  const c = v < -1 ? -1 : v > 1 ? 1 : v;
  let b = Math.round(c * 7);
  if (b < -7) b = -7;
  if (b > 7) b = 7;
  return b & 0xf; // two's-complement low nibble
}

/** Quantize one rotated vector to 4-bit codes (length paddedDim, multiple of 8). */
export function quantizeRow4(rotated: Float32Array): QuantizedRow {
  const n = rotated.length;
  if (n % 8 !== 0) throw new Error('paddedDim must be a multiple of 8 for 4-bit');

  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(rotated[i]!);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs > 0 ? maxAbs : 1;
  const inv = 1 / scale;

  const words = new Uint32Array(n / 8);
  for (let g = 0; g < words.length; g++) {
    const o = g * 8;
    let w = 0;
    for (let j = 0; j < 8; j++) {
      w |= snormNibble(rotated[o + j]! * inv) << (4 * j);
    }
    words[g] = w >>> 0;
  }
  return { words, scale };
}

/** Reference 4-bit dequantization (CPU), mirrors the in-shader nibble unpack. */
export function dequantRow4(words: Uint32Array, scale: number): Float32Array {
  const out = new Float32Array(words.length * 8);
  for (let g = 0; g < words.length; g++) {
    const w = words[g]!;
    for (let j = 0; j < 8; j++) {
      let b = (w >>> (4 * j)) & 0xf;
      if (b > 7) b -= 16; // sign-extend nibble
      out[g * 8 + j] = Math.max(b / 7, -1) * scale;
    }
  }
  return out;
}

// ---- 1-bit (binary) variant ---------------------------------------------------
// The extreme rung: keep only each rotated coordinate's SIGN, so 32 coords pack
// into one u32 (~32× less than fp32 after the pad). A row reconstructs as
// `sign_i * scale`, and the standard result is that the reconstruction magnitude
// minimizing error is `scale = mean|coord|` (not maxAbs) — so we use the mean.
// Scoring is asymmetric (fp32 query × ±1 corpus): the kernel accumulates
// Σ sign_i·q_i and multiplies by the row scale. Rotation spreads energy so the
// sign bits carry signal; the exact fp32 re-rank then recovers precision, which
// is why binary is viable at all. Bit j of word g holds coord g*32+j (LSB first),
// set when the coord is ≥ 0 — matching the in-shader `(w >> j) & 1` unpack.

/** Quantize one rotated vector to 1-bit sign codes (length paddedDim, multiple of 32). */
export function quantizeRow1(rotated: Float32Array): QuantizedRow {
  const n = rotated.length;
  if (n % 32 !== 0) throw new Error('paddedDim must be a multiple of 32 for 1-bit');

  let absSum = 0;
  for (let i = 0; i < n; i++) absSum += Math.abs(rotated[i]!);
  const scale = absSum > 0 ? absSum / n : 1; // mean|coord|

  const words = new Uint32Array(n / 32);
  for (let g = 0; g < words.length; g++) {
    const o = g * 32;
    let w = 0;
    for (let j = 0; j < 32; j++) {
      if (rotated[o + j]! >= 0) w |= 1 << j;
    }
    words[g] = w >>> 0;
  }
  return { words, scale };
}

/** Reference 1-bit dequantization (CPU), mirrors the in-shader sign unpack. */
export function dequantRow1(words: Uint32Array, scale: number): Float32Array {
  const out = new Float32Array(words.length * 32);
  for (let g = 0; g < words.length; g++) {
    const w = words[g]!;
    for (let j = 0; j < 32; j++) {
      const bit = (w >>> j) & 1;
      out[g * 32 + j] = (bit ? 1 : -1) * scale;
    }
  }
  return out;
}
