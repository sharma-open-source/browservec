// Tiny seeded PRNG (mulberry32) — deterministic so a given seed reproduces the
// exact rotation, which lets us re-quantize from the fp32 source of truth on
// reload ("store the rotation seed").

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random ±1 vector of length n from a seed. */
export function randomSigns(n: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rng() < 0.5 ? -1 : 1;
  return out;
}
