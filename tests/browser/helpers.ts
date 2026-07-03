import { BrowserVec, type Metric, type VectorRecord } from '../../src/index';
import { mulberry32 } from '../../src/quant/prng';
import { normalizeInPlace } from '../../src/store/store';

/** True when a WebGPU adapter is actually reachable (flag-gated in headless CI). */
export async function webgpuAvailable(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

export function randomVectors(rows: number, dim: number, seed: number, normalize = false): Float32Array[] {
  const rng = mulberry32(seed);
  const out: Float32Array[] = [];
  for (let r = 0; r < rows; r++) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = rng() * 2 - 1;
    if (normalize) normalizeInPlace(v);
    out.push(v);
  }
  return out;
}

export function records(vectors: Float32Array[], meta?: (i: number) => Record<string, string | number | boolean | null>): VectorRecord[] {
  return vectors.map((vector, i) => ({
    id: `v${i}`,
    vector,
    ...(meta ? { metadata: meta(i) } : {}),
  }));
}

/** Exact brute-force reference over the ORIGINAL vectors, library score conventions. */
export function bruteForce(
  vectors: Float32Array[],
  q: Float32Array,
  k: number,
  metric: Metric,
  normalize: boolean,
): Array<{ id: string; score: number }> {
  const qq = q.slice();
  if (normalize) normalizeInPlace(qq);
  const scored = vectors.map((v, i) => {
    const vv = v.slice();
    if (normalize) normalizeInPlace(vv);
    let s = 0;
    if (metric === 'l2') {
      for (let j = 0; j < vv.length; j++) {
        const d = vv[j]! - qq[j]!;
        s -= d * d;
      }
    } else {
      for (let j = 0; j < vv.length; j++) s += vv[j]! * qq[j]!;
    }
    return { id: `v${i}`, score: s };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

export function recall(got: Array<{ id: string }>, want: Array<{ id: string }>): number {
  const truth = new Set(want.map((h) => h.id));
  let hit = 0;
  for (const h of got) if (truth.has(h.id)) hit++;
  return want.length === 0 ? 1 : hit / want.length;
}

export { BrowserVec };
