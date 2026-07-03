// CPU fallback (§NFR-7) end-to-end: force the no-GPU path via the library's own
// test seam and check the store behaves identically to the exact reference.
// These tests run on any browser, GPU or not — they anchor CI even when the
// runner has no WebGPU adapter.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BrowserVec, bruteForce, randomVectors, records, recall } from './helpers';

beforeAll(() => {
  (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__ = true;
});

afterAll(() => {
  delete (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__;
});

describe('CPU fallback (fallback: "wasm")', () => {
  it('reports the wasm device and supports exact flat search (cosine)', async () => {
    const dim = 32;
    const vectors = randomVectors(300, dim, 1);
    const db = await BrowserVec.create({ dimension: dim, fallback: 'wasm' });
    try {
      expect(db.stats().device).toBe('wasm');
      await db.addBatch(records(vectors));
      expect(db.stats().count).toBe(300);

      const [q] = randomVectors(1, dim, 2);
      const got = await db.query(q!, { k: 10 });
      const want = bruteForce(vectors, q!, 10, 'cosine', true);
      expect(got.map((h) => h.id)).toEqual(want.map((h) => h.id));
      for (let i = 0; i < got.length; i++) {
        expect(got[i]!.score).toBeCloseTo(want[i]!.score, 3);
      }
    } finally {
      db.destroy();
    }
  });

  it('matches the exact reference on l2 too', async () => {
    const dim = 16;
    const vectors = randomVectors(200, dim, 3);
    const db = await BrowserVec.create({ dimension: dim, metric: 'l2', fallback: 'wasm' });
    try {
      await db.addBatch(records(vectors));
      const [q] = randomVectors(1, dim, 4);
      const got = await db.query(q!, { k: 5 });
      const want = bruteForce(vectors, q!, 5, 'l2', false);
      expect(got.map((h) => h.id)).toEqual(want.map((h) => h.id));
    } finally {
      db.destroy();
    }
  });

  it('supports metadata filters, delete, update, and get', async () => {
    const dim = 8;
    const vectors = randomVectors(100, dim, 5);
    const db = await BrowserVec.create({ dimension: dim, fallback: 'wasm' });
    try {
      await db.addBatch(records(vectors, (i) => ({ group: i % 2 === 0 ? 'even' : 'odd', n: i })));

      const [q] = randomVectors(1, dim, 6);
      const filtered = await db.query(q!, { k: 10, filter: { group: 'even' } });
      expect(filtered.length).toBe(10);
      for (const h of filtered) expect(h.metadata!.group).toBe('even');

      const ranged = await db.query(q!, { k: 100, filter: { n: { $gte: 90 } } });
      expect(ranged.length).toBe(10);

      expect(db.delete('v0')).toBe(true);
      expect(db.get('v0')).toBeNull();
      expect(db.stats().count).toBe(99);
      const afterDelete = await db.query(q!, { k: 100 });
      expect(afterDelete.some((h) => h.id === 'v0')).toBe(false);

      const g = db.get('v1')!;
      expect(g.metadata).toEqual({ group: 'odd', n: 1 });

      await db.update({ id: 'v2', vector: vectors[3]!, metadata: { group: 'updated', n: -1 } });
      expect(db.get('v2')!.metadata!.group).toBe('updated');
    } finally {
      db.destroy();
    }
  });

  it('HNSW works on the CPU fallback with good recall', async () => {
    const dim = 16;
    const vectors = randomVectors(500, dim, 7);
    const db = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      ann: { type: 'hnsw', seed: 1, efSearch: 128 },
    });
    try {
      await db.addBatch(records(vectors));
      const stats = db.stats();
      expect(stats.device).toBe('wasm');
      expect(stats.maxLevel).toBeGreaterThanOrEqual(0);

      let total = 0;
      const queries = randomVectors(10, dim, 8);
      for (const q of queries) {
        const got = await db.query(q, { k: 10 });
        total += recall(got, bruteForce(vectors, q, 10, 'cosine', true));
      }
      expect(total / 10).toBeGreaterThanOrEqual(0.9);
    } finally {
      db.destroy();
    }
  });

  it('rejects quantization and IVF under the fallback', async () => {
    await expect(
      BrowserVec.create({ dimension: 16, fallback: 'wasm', quantBits: 8 }),
    ).rejects.toThrow();
    await expect(
      BrowserVec.create({ dimension: 16, fallback: 'wasm', ann: { type: 'ivf' } }),
    ).rejects.toThrow();
  });

  it('validates dimension and duplicate ids', async () => {
    await expect(BrowserVec.create({ dimension: 0, fallback: 'wasm' })).rejects.toThrow(/positive integer/);
    const db = await BrowserVec.create({ dimension: 4, fallback: 'wasm' });
    try {
      await db.add({ id: 'a', vector: [1, 2, 3, 4] });
      await expect(db.add({ id: 'a', vector: [1, 2, 3, 4] })).rejects.toThrow(/duplicate/);
      await expect(db.add({ id: 'b', vector: [1, 2] })).rejects.toThrow(/dim/);
    } finally {
      db.destroy();
    }
  });

  it('supports text ingestion via the hashing embedder', async () => {
    const { hashingEmbedder } = await import('../../src/index');
    const embedder = hashingEmbedder({ dimension: 64 });
    const db = await BrowserVec.create({ dimension: 64, fallback: 'wasm', embedder });
    try {
      await db.addTexts([
        { id: 'gpu', text: 'webgpu compute shaders for vector search' },
        { id: 'cake', text: 'chocolate cake baking recipe' },
        { id: 'db', text: 'vector database search in the browser' },
      ]);
      const hits = await db.queryText('vector search', { k: 2 });
      expect(hits.map((h) => h.id)).toContain('db');
      expect(hits.map((h) => h.id)).not.toContain('cake');
    } finally {
      db.destroy();
    }
  });
});
