import { describe, it, expect } from 'vitest';
import { Rotator, fwht, padToPow2 } from '../../src/quant/rotator';
import { mulberry32 } from '../../src/quant/prng';

function randomVec(n: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = rng() * 2 - 1;
  return v;
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

describe('padToPow2', () => {
  it('rounds up to the next power of two with a floor of 4', () => {
    expect(padToPow2(1)).toBe(4);
    expect(padToPow2(4)).toBe(4);
    expect(padToPow2(5)).toBe(8);
    expect(padToPow2(384)).toBe(512);
    expect(padToPow2(768)).toBe(1024);
    expect(padToPow2(1024)).toBe(1024);
    expect(padToPow2(1536)).toBe(2048);
  });
});

describe('fwht', () => {
  it('is an involution up to the factor n', () => {
    const n = 64;
    const v = randomVec(n, 1);
    const copy = v.slice();
    fwht(copy);
    fwht(copy);
    for (let i = 0; i < n; i++) expect(copy[i]!).toBeCloseTo(v[i]! * n, 3);
  });

  it('computes the known transform of a unit impulse', () => {
    const v = new Float32Array(8);
    v[0] = 1;
    fwht(v);
    // H·e0 is the all-ones row.
    for (const x of v) expect(x).toBe(1);
  });
});

describe('Rotator', () => {
  it('is orthonormal: preserves dot products and norms', () => {
    const dim = 100; // non-power-of-two to exercise padding
    const rot = new Rotator(dim, 12345);
    const a = randomVec(dim, 2);
    const b = randomVec(dim, 3);
    const ra = rot.rotate(a);
    const rb = rot.rotate(b);
    expect(ra.length).toBe(rot.paddedDim);
    expect(dot(ra, rb)).toBeCloseTo(dot(a, b), 3);
    expect(dot(ra, ra)).toBeCloseTo(dot(a, a), 3);
  });

  it('reproduces the exact rotation from the same seed', () => {
    const a = new Rotator(384, 99, 2);
    const b = new Rotator(384, 99, 2);
    const v = randomVec(384, 4);
    expect(a.rotate(v)).toEqual(b.rotate(v));
  });

  it('differs across seeds and across round counts', () => {
    const v = randomVec(64, 5);
    const r1 = new Rotator(64, 1).rotate(v);
    const r2 = new Rotator(64, 2).rotate(v);
    const r3 = new Rotator(64, 1, 3).rotate(v);
    expect(r1).not.toEqual(r2);
    expect(r1).not.toEqual(r3);
  });

  it('rotateInto validates the destination length', () => {
    const rot = new Rotator(10, 1);
    expect(() => rot.rotateInto(randomVec(10, 6), new Float32Array(8))).toThrow(/paddedDim/);
  });

  it('rotateInto overwrites any stale destination content', () => {
    const rot = new Rotator(6, 1);
    const dst = new Float32Array(rot.paddedDim).fill(123);
    const v = randomVec(6, 7);
    rot.rotateInto(v, dst);
    expect(dst).toEqual(rot.rotate(v));
  });
});
