// WebGPU end-to-end suite: exact flat parity against the CPU reference, filtered
// queries, the quantization ladder, IVF, HNSW (cpu + gpu search), and chunking.
// Skipped as a block when the environment exposes no WebGPU adapter (the
// fallback + persistence suites still anchor CI there).

import { describe, expect, it } from 'vitest';
import { BrowserVec, bruteForce, randomVectors, records, recall, webgpuAvailable } from './helpers';

const hasGpu = await webgpuAvailable();
// eslint-disable-next-line no-console
console.log(`[browservec tests] WebGPU adapter available: ${hasGpu}`);

describe.skipIf(!hasGpu)('WebGPU flat index', () => {
  it('matches the exact CPU reference (cosine)', async () => {
    const dim = 64;
    const vectors = randomVectors(1000, dim, 1);
    const db = await BrowserVec.create({ dimension: dim });
    try {
      expect(db.stats().device).toBe('webgpu');
      await db.addBatch(records(vectors));

      for (const q of randomVectors(5, dim, 2)) {
        const got = await db.query(q, { k: 10 });
        const want = bruteForce(vectors, q, 10, 'cosine', true);
        expect(got.map((h) => h.id)).toEqual(want.map((h) => h.id));
        for (let i = 0; i < got.length; i++) {
          expect(got[i]!.score).toBeCloseTo(want[i]!.score, 2);
        }
      }
    } finally {
      db.destroy();
    }
  });

  it('matches the exact CPU reference (l2 and dot)', async () => {
    const dim = 32;
    for (const metric of ['l2', 'dot'] as const) {
      const vectors = randomVectors(400, dim, 3);
      const db = await BrowserVec.create({ dimension: dim, metric });
      try {
        await db.addBatch(records(vectors));
        const [q] = randomVectors(1, dim, 4);
        const got = await db.query(q!, { k: 10 });
        const want = bruteForce(vectors, q!, 10, metric, false);
        expect(got.map((h) => h.id)).toEqual(want.map((h) => h.id));
      } finally {
        db.destroy();
      }
    }
  });

  it('agrees with the forced-CPU fallback on the same data (NFR-7 parity)', async () => {
    const dim = 24;
    const vectors = randomVectors(500, dim, 5);
    const gpu = await BrowserVec.create({ dimension: dim });
    (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__ = true;
    const cpu = await BrowserVec.create({ dimension: dim, fallback: 'wasm' });
    delete (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__;
    try {
      await gpu.addBatch(records(vectors));
      await cpu.addBatch(records(vectors));
      for (const q of randomVectors(3, dim, 6)) {
        const a = await gpu.query(q, { k: 10 });
        const b = await cpu.query(q, { k: 10 });
        expect(a.map((h) => h.id)).toEqual(b.map((h) => h.id));
      }
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  });

  it('filtered queries are exact at both selectivities (CPU-scan and GPU-mask paths)', async () => {
    const dim = 16;
    const n = 6000; // > FILTER_CPU_SCAN_MAX so broad filters take the GPU mask path
    const vectors = randomVectors(n, dim, 7);
    const db = await BrowserVec.create({ dimension: dim });
    try {
      await db.addBatch(records(vectors, (i) => ({ n: i, rare: i % 1000 === 0 })));
      const [q] = randomVectors(1, dim, 8);

      // Selective filter (6 matches): exact CPU scan path.
      const rare = await db.query(q!, { k: 10, filter: { rare: true } });
      expect(rare.length).toBe(6);
      for (const h of rare) expect(h.metadata!.rare).toBe(true);

      // Broad filter (5000 matches): in-index GPU mask path, still exact.
      const broad = await db.query(q!, { k: 10, filter: { n: { $lt: 5000 } } });
      const want = bruteForce(vectors.slice(0, 5000), q!, 10, 'cosine', true);
      expect(broad.map((h) => h.id)).toEqual(want.map((h) => h.id));
    } finally {
      db.destroy();
    }
  });

  it('queryBatch agrees with individual queries', async () => {
    const dim = 32;
    const vectors = randomVectors(300, dim, 9);
    const db = await BrowserVec.create({ dimension: dim });
    try {
      await db.addBatch(records(vectors));
      const queries = randomVectors(4, dim, 10);
      const batch = await db.queryBatch(queries, { k: 5 });
      expect(batch.length).toBe(4);
      for (let i = 0; i < queries.length; i++) {
        const single = await db.query(queries[i]!, { k: 5 });
        expect(batch[i]!.map((h) => h.id)).toEqual(single.map((h) => h.id));
      }
    } finally {
      db.destroy();
    }
  });

  it('chunked corpora (chunkRows) return the same results as unchunked', async () => {
    const dim = 16;
    const vectors = randomVectors(350, dim, 11);
    const plain = await BrowserVec.create({ dimension: dim });
    const chunked = await BrowserVec.create({ dimension: dim, chunkRows: 100 });
    try {
      await plain.addBatch(records(vectors));
      await chunked.addBatch(records(vectors));
      expect(chunked.stats().chunks).toBeGreaterThanOrEqual(4);

      for (const q of randomVectors(3, dim, 12)) {
        const a = await plain.query(q, { k: 10 });
        const b = await chunked.query(q, { k: 10 });
        expect(b.map((h) => h.id)).toEqual(a.map((h) => h.id));
      }
    } finally {
      plain.destroy();
      chunked.destroy();
    }
  });

  it('delete + compact rebuilds the index correctly', async () => {
    const dim = 16;
    const vectors = randomVectors(100, dim, 13);
    const db = await BrowserVec.create({ dimension: dim });
    try {
      await db.addBatch(records(vectors));
      db.delete('v10');
      db.delete('v20');
      const removed = await db.compact();
      expect(removed).toBe(2);
      expect(db.stats().count).toBe(98);
      expect(db.stats().deleted).toBeUndefined();

      const got = await db.query(vectors[30]!, { k: 1 });
      expect(got[0]!.id).toBe('v30');
      const gone = await db.query(vectors[10]!, { k: 100 });
      expect(gone.some((h) => h.id === 'v10')).toBe(false);
    } finally {
      db.destroy();
    }
  });
});

describe.skipIf(!hasGpu)('quantization ladder (TurboQuant)', () => {
  // Re-rank makes the top-k nearly exact; recall floors are set per bit width.
  const cases = [
    { bits: 8 as const, floor: 0.95 },
    { bits: 4 as const, floor: 0.85 },
    { bits: 1 as const, floor: 0.7 },
  ];

  it.each(cases)('int$bits achieves recall ≥ $floor with exact re-rank', async ({ bits, floor }) => {
    const dim = 64;
    const vectors = randomVectors(1000, dim, 20 + bits);
    const db = await BrowserVec.create({ dimension: dim, quantBits: bits });
    try {
      expect(db.stats().quantBits).toBe(bits);
      await db.addBatch(records(vectors));

      let total = 0;
      const queries = randomVectors(10, dim, 30 + bits);
      for (const q of queries) {
        const got = await db.query(q, { k: 10 });
        total += recall(got, bruteForce(vectors, q, 10, 'cosine', true));
      }
      expect(total / queries.length).toBeGreaterThanOrEqual(floor);
    } finally {
      db.destroy();
    }
  });

  it('rejects quantization on l2', async () => {
    await expect(BrowserVec.create({ dimension: 16, metric: 'l2', quantBits: 8 })).rejects.toThrow();
  });
});

describe.skipIf(!hasGpu)('IVF index', () => {
  it('builds clusters and reaches good recall at a generous nprobe', async () => {
    const dim = 32;
    const vectors = randomVectors(2000, dim, 40);
    const db = await BrowserVec.create({
      dimension: dim,
      ann: { type: 'ivf', nlist: 32, nprobe: 16, seed: 1 },
    });
    try {
      await db.addBatch(records(vectors));

      let total = 0;
      const queries = randomVectors(10, dim, 41);
      for (const q of queries) {
        const got = await db.query(q, { k: 10 });
        total += recall(got, bruteForce(vectors, q, 10, 'cosine', true));
      }
      expect(total / queries.length).toBeGreaterThanOrEqual(0.8);
      // The index builds on demand — nlist is reported once the first query has run.
      expect(db.stats().nlist).toBe(32);
    } finally {
      db.destroy();
    }
  });

  it('a per-query nprobe override trades recall for speed', async () => {
    const dim = 32;
    const vectors = randomVectors(2000, dim, 42);
    const db = await BrowserVec.create({
      dimension: dim,
      ann: { type: 'ivf', nlist: 32, nprobe: 1, seed: 2 },
    });
    try {
      await db.addBatch(records(vectors));
      const queries = randomVectors(10, dim, 43);
      let low = 0;
      let high = 0;
      for (const q of queries) {
        const truth = bruteForce(vectors, q, 10, 'cosine', true);
        low += recall(await db.query(q, { k: 10 }), truth);
        high += recall(await db.query(q, { k: 10, nprobe: 32 }), truth);
      }
      // Scanning every cluster must be exact; nprobe=1 strictly can't beat it.
      expect(high / queries.length).toBeGreaterThanOrEqual(0.99);
      expect(low / queries.length).toBeLessThanOrEqual(high / queries.length);
    } finally {
      db.destroy();
    }
  });
});

describe.skipIf(!hasGpu)('HNSW index (GPU host)', () => {
  it('reaches good recall with the CPU graph walk', async () => {
    const dim = 32;
    const vectors = randomVectors(800, dim, 50);
    const db = await BrowserVec.create({
      dimension: dim,
      ann: { type: 'hnsw', seed: 1, efSearch: 128 },
    });
    try {
      await db.addBatch(records(vectors));
      expect(db.stats().maxLevel).toBeGreaterThanOrEqual(0);

      let total = 0;
      const queries = randomVectors(10, dim, 51);
      for (const q of queries) {
        total += recall(await db.query(q, { k: 10 }), bruteForce(vectors, q, 10, 'cosine', true));
      }
      expect(total / queries.length).toBeGreaterThanOrEqual(0.9);
    } finally {
      db.destroy();
    }
  });

  it("search: 'gpu' answers batches with good recall", async () => {
    const dim = 32;
    const vectors = randomVectors(800, dim, 52);
    const db = await BrowserVec.create({
      dimension: dim,
      ann: { type: 'hnsw', seed: 2, search: 'gpu', M: 16, efSearch: 128 },
    });
    try {
      await db.addBatch(records(vectors));
      // May report 'cpu' if the kernel's constraints aren't met; either way the
      // results must be good.
      expect(['gpu', 'cpu']).toContain(db.stats().graphSearch);

      const queries = randomVectors(8, dim, 53);
      const results = await db.queryBatch(queries, { k: 10 });
      let total = 0;
      for (let i = 0; i < queries.length; i++) {
        total += recall(results[i]!, bruteForce(vectors, queries[i]!, 10, 'cosine', true));
      }
      expect(total / queries.length).toBeGreaterThanOrEqual(0.85);
    } finally {
      db.destroy();
    }
  });
});
