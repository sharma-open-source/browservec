// Persistence e2e (M2/M7c): save/auto-load, export/import, encryption at rest,
// and tombstone compaction. Runs on the forced-CPU path so it works with or
// without a WebGPU adapter — the persistence layer is identical either way.

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BrowserVec, randomVectors, records } from './helpers';

beforeAll(() => {
  (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__ = true;
});

afterAll(() => {
  delete (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__;
});

let nameCounter = 0;
function uniqueName(): string {
  return `bvec-test-${Date.now()}-${nameCounter++}`;
}

describe('persistence (IndexedDB)', () => {
  it('save() then create() auto-loads the snapshot', async () => {
    const dim = 8;
    const name = uniqueName();
    const vectors = randomVectors(50, dim, 1);

    const a = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb' },
    });
    await a.addBatch(records(vectors, (i) => ({ n: i })));
    const beforeSave = await a.query(vectors[7]!, { k: 3 });
    await a.save();
    a.destroy();

    const b = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb' },
    });
    try {
      expect(b.stats().count).toBe(50);
      expect(b.stats().persist).toBe('indexeddb');
      expect(b.get('v7')!.metadata).toEqual({ n: 7 });
      const afterLoad = await b.query(vectors[7]!, { k: 3 });
      expect(afterLoad.map((h) => h.id)).toEqual(beforeSave.map((h) => h.id));
      expect(afterLoad[0]!.id).toBe('v7');
    } finally {
      b.destroy();
    }
  });

  it('a dimension mismatch on auto-load throws instead of mis-loading', async () => {
    const name = uniqueName();
    const a = await BrowserVec.create({
      dimension: 8,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb' },
    });
    await a.addBatch(records(randomVectors(5, 8, 2)));
    await a.save();
    a.destroy();

    await expect(
      BrowserVec.create({
        dimension: 16,
        fallback: 'wasm',
        persist: { name, backend: 'indexeddb' },
      }),
    ).rejects.toThrow(/dim/);
  });

  it('save() compacts tombstones — deleted rows are gone after reload', async () => {
    const dim = 8;
    const name = uniqueName();
    const a = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb' },
    });
    await a.addBatch(records(randomVectors(20, dim, 3)));
    a.delete('v0');
    a.delete('v11');
    expect(a.stats().deleted).toBe(2);
    await a.save();
    a.destroy();

    const b = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb' },
    });
    try {
      expect(b.stats().count).toBe(18);
      expect(b.stats().deleted).toBeUndefined();
      expect(b.get('v0')).toBeNull();
      expect(b.get('v1')).not.toBeNull();
    } finally {
      b.destroy();
    }
  });

  it('save() without a persist config throws', async () => {
    const db = await BrowserVec.create({ dimension: 4, fallback: 'wasm' });
    try {
      await expect(db.save()).rejects.toThrow(/persist/);
    } finally {
      db.destroy();
    }
  });
});

describe('export / import', () => {
  it('round-trips a store through a Blob', async () => {
    const dim = 8;
    const vectors = randomVectors(30, dim, 4);
    const a = await BrowserVec.create({ dimension: dim, fallback: 'wasm' });
    await a.addBatch(records(vectors, (i) => ({ tag: `t${i}` })));
    const blob = await a.export();
    const expected = await a.query(vectors[3]!, { k: 5 });
    a.destroy();

    const b = await BrowserVec.import(blob, { fallback: 'wasm' });
    try {
      expect(b.stats().count).toBe(30);
      expect(b.get('v3')!.metadata).toEqual({ tag: 't3' });
      const got = await b.query(vectors[3]!, { k: 5 });
      expect(got.map((h) => h.id)).toEqual(expected.map((h) => h.id));
    } finally {
      b.destroy();
    }
  });

  it('encrypted export requires the right passphrase', async () => {
    const dim = 8;
    const a = await BrowserVec.create({ dimension: dim, fallback: 'wasm' });
    await a.addBatch(records(randomVectors(10, dim, 5)));
    const blob = await a.export({ encryption: { passphrase: 'sekret' } });
    a.destroy();

    // Right passphrase works.
    const ok = await BrowserVec.import(blob, {
      fallback: 'wasm',
      encryption: { passphrase: 'sekret' },
    });
    expect(ok.stats().count).toBe(10);
    ok.destroy();

    // Wrong passphrase and missing passphrase both fail loudly.
    await expect(
      BrowserVec.import(blob, { fallback: 'wasm', encryption: { passphrase: 'nope' } }),
    ).rejects.toThrow(/wrong passphrase or corrupted/);
    await expect(BrowserVec.import(blob, { fallback: 'wasm' })).rejects.toThrow(/encrypted/);
  });

  it('a passphrase against a plaintext blob is rejected', async () => {
    const a = await BrowserVec.create({ dimension: 4, fallback: 'wasm' });
    await a.add({ id: 'x', vector: [1, 0, 0, 0] });
    const blob = await a.export();
    a.destroy();
    await expect(
      BrowserVec.import(blob, { fallback: 'wasm', encryption: { passphrase: 'p' } }),
    ).rejects.toThrow(/not encrypted/);
  });
});

describe('encrypted persistence at rest', () => {
  it('auto-loads through the encryption envelope', async () => {
    const dim = 8;
    const name = uniqueName();
    const a = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb', encryption: { passphrase: 'hunter2' } },
    });
    await a.addBatch(records(randomVectors(12, dim, 6)));
    await a.save();
    a.destroy();

    const b = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'indexeddb', encryption: { passphrase: 'hunter2' } },
    });
    expect(b.stats().count).toBe(12);
    b.destroy();

    await expect(
      BrowserVec.create({
        dimension: dim,
        fallback: 'wasm',
        persist: { name, backend: 'indexeddb', encryption: { passphrase: 'wrong' } },
      }),
    ).rejects.toThrow(/wrong passphrase or corrupted/);
  });
});

describe('HNSW graph persistence (M7c)', () => {
  it('reloads an HNSW store with identical results and no drift after appends', async () => {
    const dim = 16;
    const name = uniqueName();
    const vectors = randomVectors(300, dim, 7);
    const a = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      ann: { type: 'hnsw', seed: 42 },
      persist: { name, backend: 'indexeddb' },
    });
    await a.addBatch(records(vectors));
    const [q] = randomVectors(1, dim, 8);
    const beforeSave = await a.query(q!, { k: 10 });
    await a.save();
    a.destroy();

    const b = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      ann: { type: 'hnsw', seed: 42 },
      persist: { name, backend: 'indexeddb' },
    });
    try {
      expect(b.stats().count).toBe(300);
      expect(b.stats().maxLevel).toBeGreaterThanOrEqual(0);
      const afterLoad = await b.query(q!, { k: 10 });
      expect(afterLoad.map((h) => h.id)).toEqual(beforeSave.map((h) => h.id));

      // Appends after the reload keep working.
      await b.addBatch(records(randomVectors(50, dim, 9)).map((r, i) => ({ ...r, id: `w${i}` })));
      expect(b.stats().count).toBe(350);
    } finally {
      b.destroy();
    }
  });
});

describe('OPFS backend', () => {
  it('saves and loads via OPFS when available', async (ctx) => {
    if (!navigator.storage?.getDirectory) return ctx.skip();
    const dim = 8;
    const name = uniqueName();
    const a = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'opfs' },
    });
    await a.addBatch(records(randomVectors(10, dim, 10)));
    await a.save();
    a.destroy();

    const b = await BrowserVec.create({
      dimension: dim,
      fallback: 'wasm',
      persist: { name, backend: 'opfs' },
    });
    try {
      expect(b.stats().count).toBe(10);
      expect(b.stats().persist).toBe('opfs');
    } finally {
      b.destroy();
    }
  });
});
