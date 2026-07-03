// The nprobe auto-tuner's pure core: given the probe rank of each ground-truth
// hit's cluster, pick the smallest nprobe whose cumulative recall meets the
// target (see src/index/ivfTune.ts for why one scan covers every nprobe).

import { describe, expect, it } from 'vitest';
import { chooseNprobe, pickTuneQueries } from '../../src/index/ivfTune.js';

describe('chooseNprobe', () => {
  it('picks the smallest nprobe whose cumulative recall meets the target', () => {
    // 10 hits: 6 in the nearest cluster (rank 0), 3 at rank 1, 1 at rank 7.
    const ranks = [0, 0, 0, 0, 0, 0, 1, 1, 1, 7];
    expect(chooseNprobe(ranks, 16, 0.5)).toEqual({ nprobe: 1, recall: 0.6 });
    expect(chooseNprobe(ranks, 16, 0.9)).toEqual({ nprobe: 2, recall: 0.9 });
    expect(chooseNprobe(ranks, 16, 0.95)).toEqual({ nprobe: 8, recall: 1 });
  });

  it('recall is monotone in nprobe: a higher target never yields a smaller nprobe', () => {
    const ranks = [0, 2, 2, 3, 5, 5, 5, 9];
    let prev = 0;
    for (const target of [0.1, 0.3, 0.5, 0.7, 0.9, 1]) {
      const { nprobe, recall } = chooseNprobe(ranks, 10, target);
      expect(recall).toBeGreaterThanOrEqual(target);
      expect(nprobe).toBeGreaterThanOrEqual(prev);
      prev = nprobe;
    }
  });

  it('target 1 requires probing up to the worst hit, never past nlist', () => {
    expect(chooseNprobe([0, 4], 5, 1)).toEqual({ nprobe: 5, recall: 1 });
    expect(chooseNprobe([0, 0], 5, 1)).toEqual({ nprobe: 1, recall: 1 });
  });

  it('degenerate inputs fall back safely', () => {
    expect(chooseNprobe([], 16, 0.95)).toEqual({ nprobe: 16, recall: 1 });
    expect(chooseNprobe([0], 1, 0.95)).toEqual({ nprobe: 1, recall: 1 });
    expect(chooseNprobe([], 0, 0.95)).toEqual({ nprobe: 1, recall: 1 });
  });

  it('accepts typed arrays', () => {
    expect(chooseNprobe(Uint32Array.from([0, 1, 1, 3]), 8, 0.75)).toEqual({ nprobe: 2, recall: 0.75 });
  });
});

describe('pickTuneQueries', () => {
  it('returns evenly spaced distinct indices within range', () => {
    const picks = pickTuneQueries(1000, 32);
    expect(picks).toHaveLength(32);
    expect(new Set(picks).size).toBe(32);
    for (const p of picks) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThan(1000);
    }
    expect(picks).toEqual([...picks].sort((a, b) => a - b));
  });

  it('clamps to the filled count when the reservoir is small', () => {
    expect(pickTuneQueries(5, 32)).toEqual([0, 1, 2, 3, 4]);
    expect(pickTuneQueries(0, 32)).toEqual([]);
  });

  it('is deterministic', () => {
    expect(pickTuneQueries(777, 32)).toEqual(pickTuneQueries(777, 32));
  });
});
