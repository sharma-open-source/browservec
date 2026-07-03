import { describe, it, expect } from 'vitest';
import { BatchEncoder } from '../../src/quant/encode';
import { Rotator } from '../../src/quant/rotator';
import { quantizeRow, quantizeRow4, quantizeRow1 } from '../../src/quant/codec';
import { mulberry32 } from '../../src/quant/prng';

function randomBatch(rows: number, dim: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const data = new Float32Array(rows * dim);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  return data;
}

describe('BatchEncoder', () => {
  it('matches manual rotate + quantize per row (int8)', () => {
    const dim = 48;
    const rows = 5;
    const cfg = { dim, seed: 11, rounds: 2, bits: 8 as const };
    const enc = new BatchEncoder(cfg);
    const data = randomBatch(rows, dim, 1);
    const batch = enc.encode(data, rows, false);

    const rot = new Rotator(dim, cfg.seed, cfg.rounds);
    for (let r = 0; r < rows; r++) {
      const rotated = rot.rotate(data.subarray(r * dim, (r + 1) * dim));
      const q = quantizeRow(rotated);
      expect(batch.words.subarray(r * enc.wordsPerRow, (r + 1) * enc.wordsPerRow)).toEqual(q.words);
      expect(batch.scales[r]!).toBeCloseTo(q.scale, 6);
    }
    expect(batch.rotated).toBeUndefined();
  });

  it('matches the 4-bit and 1-bit codecs too', () => {
    const dim = 40;
    const data = randomBatch(3, dim, 2);
    for (const [bits, quantize] of [
      [4, quantizeRow4],
      [1, quantizeRow1],
    ] as const) {
      const enc = new BatchEncoder({ dim, seed: 5, rounds: 2, bits });
      const batch = enc.encode(data, 3, false);
      const rot = new Rotator(dim, 5, 2);
      const q0 = quantize(rot.rotate(data.subarray(0, dim)));
      expect(batch.words.subarray(0, enc.wordsPerRow)).toEqual(q0.words);
    }
  });

  it('computes wordsPerRow from the padded dim and bit width', () => {
    // dim 384 pads to 512.
    expect(new BatchEncoder({ dim: 384, seed: 1, rounds: 2, bits: 8 }).wordsPerRow).toBe(128);
    expect(new BatchEncoder({ dim: 384, seed: 1, rounds: 2, bits: 4 }).wordsPerRow).toBe(64);
    expect(new BatchEncoder({ dim: 384, seed: 1, rounds: 2, bits: 1 }).wordsPerRow).toBe(16);
  });

  it('returns rotated rows when requested, matching the rotator', () => {
    const dim = 20;
    const enc = new BatchEncoder({ dim, seed: 3, rounds: 2, bits: 8 });
    const data = randomBatch(2, dim, 3);
    const batch = enc.encode(data, 2, true);
    expect(batch.rotated).toBeDefined();
    expect(batch.rotated!.length).toBe(2 * enc.paddedDim);

    const rot = new Rotator(dim, 3, 2);
    const r1 = rot.rotate(data.subarray(dim, 2 * dim));
    expect(batch.rotated!.subarray(enc.paddedDim, 2 * enc.paddedDim)).toEqual(r1);
  });

  it('validates the data length', () => {
    const enc = new BatchEncoder({ dim: 8, seed: 1, rounds: 2, bits: 8 });
    expect(() => enc.encode(new Float32Array(15), 2, false)).toThrow(/expected 16 floats/);
  });
});
