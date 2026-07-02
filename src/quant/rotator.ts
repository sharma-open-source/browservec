// Randomized Hadamard rotation — the data-oblivious mixing step of TurboQuant
// ( arXiv:2504.19874 step 1).
//
// Each round applies a random ±1 diagonal sign flip then a fast Walsh-Hadamard
// transform (FHT), scaled to stay orthonormal. This spreads each vector's energy
// across all coordinates so a simple per-coordinate scalar quantizer is near
// optimal, and it is *data-oblivious*: the rotation depends only on a seed, never
// on the data — so we can reproduce it anywhere from the stored seed.
//
// Orthonormality matters: <Rx, Ry> = <x, y>, so dot/cosine are preserved exactly
// and only the quantizer introduces error. The query is rotated the same way, so
// search happens in the rotated space.

import { randomSigns } from './prng.js';

/** Smallest power of two >= n (and >= 4, since the kernel works in groups of 4). */
export function padToPow2(n: number): number {
  let p = 4;
  while (p < n) p <<= 1;
  return p;
}

/** In-place fast Walsh-Hadamard transform; length must be a power of two. */
export function fwht(a: Float32Array): void {
  const n = a.length;
  for (let len = 1; len < n; len <<= 1) {
    for (let i = 0; i < n; i += len << 1) {
      for (let j = i; j < i + len; j++) {
        const x = a[j]!;
        const y = a[j + len]!;
        a[j] = x + y;
        a[j + len] = x - y;
      }
    }
  }
}

export class Rotator {
  readonly paddedDim: number;
  // One ±1/sqrt(P) vector per round: the FWHT is linear, so the orthonormal
  // 1/sqrt(P) scale is pre-folded into the signs — one pass instead of two.
  private readonly signs: Float32Array[];

  constructor(
    readonly dim: number,
    readonly seed: number,
    readonly rounds = 2,
  ) {
    this.paddedDim = padToPow2(dim);
    const invScale = 1 / Math.sqrt(this.paddedDim);
    this.signs = [];
    for (let r = 0; r < rounds; r++) {
      // Distinct seed per round so the rounds aren't identical.
      const s = randomSigns(this.paddedDim, (seed ^ (0x9e3779b9 * (r + 1))) >>> 0);
      for (let i = 0; i < s.length; i++) s[i]! *= invScale;
      this.signs.push(s);
    }
  }

  /** Rotate `src` (length dim) into `dst` (length paddedDim). dst is overwritten. */
  rotateInto(src: Float32Array, dst: Float32Array): void {
    if (dst.length !== this.paddedDim) throw new Error('dst must be paddedDim long');
    dst.set(src);
    dst.fill(0, src.length); // zero the padding tail
    for (let r = 0; r < this.rounds; r++) {
      const s = this.signs[r]!;
      for (let i = 0; i < dst.length; i++) dst[i]! *= s[i]!; // sign flip + 1/sqrt(P) in one pass
      fwht(dst);
    }
  }

  /** Convenience: allocate and return a rotated copy. */
  rotate(src: Float32Array): Float32Array {
    const dst = new Float32Array(this.paddedDim);
    this.rotateInto(src, dst);
    return dst;
  }
}
