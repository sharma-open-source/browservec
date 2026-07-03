import { describe, it, expect } from 'vitest';
import { hashingEmbedder } from '../../src/embed/hashing';

function norm(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

describe('hashingEmbedder', () => {
  it('defaults to dimension 384 and honors an override', async () => {
    expect(hashingEmbedder().dimension).toBe(384);
    const e = hashingEmbedder({ dimension: 64 });
    expect(e.dimension).toBe(64);
    const [v] = await e.embed(['hello']);
    expect(v!.length).toBe(64);
  });

  it('produces unit-norm vectors', async () => {
    const e = hashingEmbedder();
    const vs = await e.embed(['the quick brown fox', 'lazy dog']);
    for (const v of vs) expect(norm(v)).toBeCloseTo(1, 4);
  });

  it('is deterministic and order-preserving in batches', async () => {
    const e = hashingEmbedder();
    const [a1, b1] = await e.embed(['alpha', 'beta']);
    const [b2, a2] = await e.embed(['beta', 'alpha']);
    expect(a1).toEqual(a2);
    expect(b1).toEqual(b2);
  });

  it('is case- and punctuation-insensitive via tokenization', async () => {
    const e = hashingEmbedder();
    const [a, b] = await e.embed(['Hello, World!', 'hello world']);
    expect(a).toEqual(b);
  });

  it('scores overlapping text higher than unrelated text', async () => {
    const e = hashingEmbedder();
    const [q, close, far] = await e.embed([
      'webgpu vector search in the browser',
      'fast vector search with webgpu',
      'banana bread recipe with walnuts',
    ]);
    expect(dot(q!, close!)).toBeGreaterThan(dot(q!, far!));
  });

  it('returns a zero vector for empty text', async () => {
    const e = hashingEmbedder();
    const [v] = await e.embed(['   ']);
    expect(norm(v!)).toBe(0);
  });

  it('bigrams distinguish word order', async () => {
    const on = hashingEmbedder();
    const [a, b] = await on.embed(['new york city', 'city york new']);
    expect(a).not.toEqual(b);

    const off = hashingEmbedder({ bigrams: false });
    const [c, d] = await off.embed(['new york city', 'city york new']);
    expect(c).toEqual(d);
  });
});
