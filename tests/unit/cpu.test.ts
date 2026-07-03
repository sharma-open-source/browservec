import { describe, it, expect } from 'vitest';
import { CpuIndex } from '../../src/fallback/cpu';
import { simdAvailable, createSimd } from '../../src/fallback/simd';
import { topK } from '../../src/index/flat';
import { mulberry32 } from '../../src/quant/prng';

function randomCorpus(rows: number, dim: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const data = new Float32Array(rows * dim);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  return data;
}

/** Scalar reference scores matching the library convention (higher = closer). */
function referenceScores(data: Float32Array, rows: number, dim: number, q: Float32Array, metric: 'dot' | 'l2') {
  const scores = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let i = 0; i < dim; i++) {
      if (metric === 'l2') {
        const d = data[r * dim + i]! - q[i]!;
        s -= d * d;
      } else {
        s += data[r * dim + i]! * q[i]!;
      }
    }
    scores[r] = s;
  }
  return scores;
}

describe('WASM-SIMD kernel', () => {
  it('is available in this Node runtime', () => {
    expect(simdAvailable()).toBe(true);
    expect(createSimd()).not.toBeNull();
  });

  it('each instance gets its own linear memory', () => {
    const a = createSimd()!;
    const b = createSimd()!;
    expect(a.mem).not.toBe(b.mem);
  });
});

// Run the same behavioral suite on both engines: the real SIMD path and the
// scalar JS fallback (forced by nulling the instance's simd handle).
describe.each([
  ['simd', false],
  ['scalar', true],
])('CpuIndex (%s path)', (_name, forceScalar) => {
  function makeIndex(dim: number, metric: 'dot' | 'l2' | 'cosine'): CpuIndex {
    const idx = new CpuIndex(dim, metric);
    if (forceScalar) (idx as unknown as { simd: null }).simd = null;
    return idx;
  }

  it.each(['dot', 'l2'] as const)('matches the exact brute-force reference (%s)', async (metric) => {
    const dim = 33; // odd dim exercises the scalar tail after the ×4 unroll
    const rows = 500;
    const data = randomCorpus(rows, dim, 1);
    const idx = makeIndex(dim, metric);
    idx.append(data, rows);
    expect(idx.size).toBe(rows);

    const rng = mulberry32(2);
    for (let t = 0; t < 5; t++) {
      const q = new Float32Array(dim);
      for (let i = 0; i < dim; i++) q[i] = rng() * 2 - 1;
      const hits = await idx.query(q, 10);
      const want = topK(referenceScores(data, rows, dim, q, metric), rows, 10);
      expect(hits.map((h) => h.row)).toEqual(want.map((h) => h.row));
      for (let i = 0; i < hits.length; i++) {
        expect(hits[i]!.score).toBeCloseTo(want[i]!.score, 3);
      }
    }
    idx.destroy();
  });

  it('returns [] before any append and validates append length', async () => {
    const idx = makeIndex(8, 'dot');
    expect(await idx.query(new Float32Array(8), 5)).toEqual([]);
    expect(() => idx.append(new Float32Array(9), 1)).toThrow(/expected 8 floats/);
    idx.destroy();
  });

  it('supports incremental appends across memory growth', async () => {
    const dim = 64;
    const rows = 3000; // forces the SIMD linear memory (1 page) to grow
    const data = randomCorpus(rows, dim, 3);
    const idx = makeIndex(dim, 'dot');
    for (let r = 0; r < rows; r += 500) {
      idx.append(data.subarray(r * dim, (r + 500) * dim), 500);
    }
    expect(idx.size).toBe(rows);

    // Query for an exact stored row: it must come back first with score ≈ |v|².
    const target = data.slice(2222 * dim, 2223 * dim);
    const hits = await idx.query(target, 1);
    expect(hits[0]!.row).toBe(2222);
    idx.destroy();
  });

  it('writeRow overwrites a row in place', async () => {
    const dim = 16;
    const idx = makeIndex(dim, 'l2');
    idx.append(randomCorpus(4, dim, 4), 4);
    const v = new Float32Array(dim).fill(0.5);
    idx.writeRow(2, v);
    const hits = await idx.query(v, 1);
    expect(hits[0]!.row).toBe(2);
    expect(hits[0]!.score).toBeCloseTo(0, 4);
    idx.destroy();
  });
});

describe('CpuIndex engine parity', () => {
  it('simd and scalar paths agree on scores', async () => {
    const dim = 20;
    const rows = 200;
    const data = randomCorpus(rows, dim, 5);
    const simd = new CpuIndex(dim, 'l2');
    const scalar = new CpuIndex(dim, 'l2');
    (scalar as unknown as { simd: null }).simd = null;
    simd.append(data, rows);
    scalar.append(data, rows);

    const q = randomCorpus(1, dim, 6);
    const a = await simd.query(q, 10);
    const b = await scalar.query(q, 10);
    expect(a.map((h) => h.row)).toEqual(b.map((h) => h.row));
    for (let i = 0; i < a.length; i++) expect(a[i]!.score).toBeCloseTo(b[i]!.score, 3);
    simd.destroy();
    scalar.destroy();
  });
});
