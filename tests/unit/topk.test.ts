import { describe, it, expect } from 'vitest';
import { topK } from '../../src/index/flat';
import { mulberry32 } from '../../src/quant/prng';

/** Exact reference: full sort descending by score. */
function reference(scores: Float32Array, n: number, k: number) {
  return Array.from(scores.subarray(0, n))
    .map((score, row) => ({ row, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

describe('topK', () => {
  it('matches a full sort on random scores', () => {
    const rng = mulberry32(1);
    for (const [n, k] of [
      [100, 10],
      [1000, 1],
      [1000, 50],
      [17, 17],
    ] as const) {
      const scores = new Float32Array(n);
      for (let i = 0; i < n; i++) scores[i] = rng() * 200 - 100;
      const got = topK(scores, n, k);
      const want = reference(scores, n, k);
      expect(got.map((h) => h.score)).toEqual(want.map((h) => h.score));
      expect(got.map((h) => h.row)).toEqual(want.map((h) => h.row));
    }
  });

  it('returns all rows in descending order when k > n', () => {
    const scores = new Float32Array([3, 1, 2]);
    const got = topK(scores, 3, 10);
    expect(got.map((h) => h.row)).toEqual([0, 2, 1]);
  });

  it('handles k = 0 and n = 0', () => {
    expect(topK(new Float32Array([1, 2]), 2, 0)).toEqual([]);
    expect(topK(new Float32Array(0), 0, 5)).toEqual([]);
  });

  it('handles negative scores (l2 convention: higher = closer)', () => {
    const scores = new Float32Array([-5, -1, -3]);
    const got = topK(scores, 3, 2);
    expect(got[0]).toEqual({ row: 1, score: -1 });
    expect(got[1]).toEqual({ row: 2, score: -3 });
  });

  it('keeps every tied score', () => {
    const scores = new Float32Array([1, 1, 1, 0]);
    const got = topK(scores, 4, 3);
    expect(got.map((h) => h.score)).toEqual([1, 1, 1]);
    expect(new Set(got.map((h) => h.row))).toEqual(new Set([0, 1, 2]));
  });
});
