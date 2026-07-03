import { describe, it, expect } from 'vitest';
import {
  quantizeRow,
  dequantRow,
  quantizeRow4,
  dequantRow4,
  quantizeRow1,
  dequantRow1,
} from '../../src/quant/codec';
import { mulberry32 } from '../../src/quant/prng';

function randomVec(n: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = (rng() * 2 - 1) * 3;
  return v;
}

describe('int8 codec', () => {
  it('round-trips within the quantization step (scale/127 per coord)', () => {
    const v = randomVec(64, 1);
    const q = quantizeRow(v);
    const back = dequantRow(q.words, q.scale);
    const step = q.scale / 127;
    for (let i = 0; i < v.length; i++) {
      expect(Math.abs(back[i]! - v[i]!)).toBeLessThanOrEqual(step / 2 + 1e-6);
    }
  });

  it('uses max |coord| as the scale so nothing clips', () => {
    const v = new Float32Array([0.5, -2.5, 1, 0]);
    const q = quantizeRow(v);
    expect(q.scale).toBe(2.5);
    const back = dequantRow(q.words, q.scale);
    expect(back[1]!).toBeCloseTo(-2.5, 4);
  });

  it('packs 4 coords per word', () => {
    const q = quantizeRow(randomVec(32, 2));
    expect(q.words.length).toBe(8);
  });

  it('handles the all-zero row (scale defaults to 1)', () => {
    const q = quantizeRow(new Float32Array(16));
    expect(q.scale).toBe(1);
    expect(dequantRow(q.words, q.scale)).toEqual(new Float32Array(16));
  });

  it('rejects lengths not divisible by 4', () => {
    expect(() => quantizeRow(new Float32Array(7))).toThrow(/multiple of 4/);
  });
});

describe('int4 codec', () => {
  it('round-trips within the coarser step (scale/7 per coord)', () => {
    const v = randomVec(64, 3);
    const q = quantizeRow4(v);
    const back = dequantRow4(q.words, q.scale);
    const step = q.scale / 7;
    for (let i = 0; i < v.length; i++) {
      expect(Math.abs(back[i]! - v[i]!)).toBeLessThanOrEqual(step / 2 + 1e-6);
    }
  });

  it('packs 8 coords per word', () => {
    const q = quantizeRow4(randomVec(64, 4));
    expect(q.words.length).toBe(8);
  });

  it('rejects lengths not divisible by 8', () => {
    expect(() => quantizeRow4(new Float32Array(12))).toThrow(/multiple of 8/);
  });
});

describe('1-bit codec', () => {
  it('keeps signs exactly and scales by mean |coord|', () => {
    const v = randomVec(64, 5);
    const q = quantizeRow1(v);
    let absSum = 0;
    for (const x of v) absSum += Math.abs(x);
    expect(q.scale).toBeCloseTo(absSum / v.length, 5);

    const back = dequantRow1(q.words, q.scale);
    for (let i = 0; i < v.length; i++) {
      const sign = v[i]! >= 0 ? 1 : -1;
      expect(back[i]!).toBeCloseTo(sign * q.scale, 5);
    }
  });

  it('packs 32 coords per word', () => {
    const q = quantizeRow1(randomVec(64, 6));
    expect(q.words.length).toBe(2);
  });

  it('rejects lengths not divisible by 32', () => {
    expect(() => quantizeRow1(new Float32Array(48))).toThrow(/multiple of 32/);
  });

  it('treats zero as non-negative (bit set)', () => {
    const v = new Float32Array(32); // all zeros
    const q = quantizeRow1(v);
    expect(q.words[0]).toBe(0xffffffff);
  });
});
