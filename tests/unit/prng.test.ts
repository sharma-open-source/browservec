import { describe, it, expect } from 'vitest';
import { mulberry32, randomSigns } from '../../src/quant/prng';

describe('mulberry32', () => {
  it('is deterministic for a given seed', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 100; i++) expect(a()).toBe(b());
  });

  it('produces different streams for different seeds', () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    const same = Array.from({ length: 20 }, () => a() === b());
    expect(same.every(Boolean)).toBe(false);
  });

  it('stays in [0, 1)', () => {
    const rng = mulberry32(0xdeadbeef);
    for (let i = 0; i < 10_000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('has a roughly uniform mean', () => {
    const rng = mulberry32(7);
    let sum = 0;
    const n = 50_000;
    for (let i = 0; i < n; i++) sum += rng();
    expect(sum / n).toBeGreaterThan(0.48);
    expect(sum / n).toBeLessThan(0.52);
  });
});

describe('randomSigns', () => {
  it('contains only +1 and -1', () => {
    const s = randomSigns(1024, 123);
    for (const v of s) expect(Math.abs(v)).toBe(1);
  });

  it('is deterministic per seed and mixes both signs', () => {
    const a = randomSigns(256, 9);
    const b = randomSigns(256, 9);
    expect(a).toEqual(b);
    expect(a.includes(1)).toBe(true);
    expect(a.includes(-1)).toBe(true);
  });
});
