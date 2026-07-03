import { describe, it, expect } from 'vitest';
import { Store, normalizeInPlace } from '../../src/store/store';

describe('normalizeInPlace', () => {
  it('scales to unit norm and leaves the zero vector untouched', () => {
    const v = new Float32Array([3, 4]);
    normalizeInPlace(v);
    expect(v[0]).toBeCloseTo(0.6, 5);
    expect(v[1]).toBeCloseTo(0.8, 5);

    const z = new Float32Array(4);
    normalizeInPlace(z);
    expect(z).toEqual(new Float32Array(4));
  });
});

describe('Store', () => {
  it('prepare validates dimension and normalizes when configured', () => {
    const s = new Store(3, 'cosine', true);
    expect(() => s.prepare([1, 2])).toThrow(/dim 2 != store dim 3/);
    const v = s.prepare([3, 0, 4]);
    expect(v[0]).toBeCloseTo(0.6, 5);
    expect(v[2]).toBeCloseTo(0.8, 5);

    const raw = new Store(3, 'dot', false).prepare([3, 0, 4]);
    expect(Array.from(raw)).toEqual([3, 0, 4]);
  });

  it('prepare copies its input (no aliasing)', () => {
    const s = new Store(2, 'dot', false);
    const src = new Float32Array([1, 2]);
    const out = s.prepare(src);
    out[0] = 99;
    expect(src[0]).toBe(1);
  });

  it('insert assigns sequential rows and rejects duplicate ids', () => {
    const s = new Store(2, 'dot', false);
    expect(s.insert('a', s.prepare([1, 0]))).toBe(0);
    expect(s.insert('b', s.prepare([0, 1]), { tag: 'x' })).toBe(1);
    expect(() => s.insert('a', s.prepare([1, 1]))).toThrow(/duplicate id: a/);
    expect(s.count).toBe(2);
    expect(s.rowCount).toBe(2);
    expect(s.entryById('b')!.metadata).toEqual({ tag: 'x' });
    expect(s.entryByRow(0)!.id).toBe('a');
  });

  it('delete tombstones without dropping the row', () => {
    const s = new Store(2, 'dot', false);
    s.insert('a', s.prepare([1, 0]));
    s.insert('b', s.prepare([0, 1]));
    expect(s.delete('a')).toBe(true);
    expect(s.delete('a')).toBe(false); // already gone
    expect(s.delete('nope')).toBe(false);

    expect(s.count).toBe(1);
    expect(s.rowCount).toBe(2); // tombstone still occupies the row
    expect(s.deletedCount).toBe(1);
    expect(s.isDeleted(0)).toBe(true);
    expect(s.has('a')).toBe(false);
    expect(s.has('b')).toBe(true);
  });

  it('liveEntries/liveVectors stay aligned and skip tombstones', () => {
    const s = new Store(2, 'dot', false);
    s.insert('a', s.prepare([1, 1]));
    s.insert('b', s.prepare([2, 2]));
    s.insert('c', s.prepare([3, 3]));
    s.delete('b');

    const entries = s.liveEntries();
    const vectors = s.liveVectors();
    expect(entries.map((e) => e.id)).toEqual(['a', 'c']);
    expect(Array.from(vectors)).toEqual([1, 1, 3, 3]);
  });

  it('vectorAt returns a copy and bounds-checks', () => {
    const s = new Store(2, 'dot', false);
    s.insert('a', s.prepare([5, 6]));
    const v = s.vectorAt(0)!;
    expect(Array.from(v)).toEqual([5, 6]);
    v[0] = 0;
    expect(s.vectorAt(0)![0]).toBe(5);
    expect(s.vectorAt(1)).toBeUndefined();
    expect(s.vectorAt(-1)).toBeUndefined();
  });

  it('dotRow and l2Row follow the higher-is-closer convention', () => {
    const dim = 5; // odd: exercises the unroll tail
    const s = new Store(dim, 'dot', false);
    const v = new Float32Array([1, -2, 3, -4, 5]);
    s.insert('a', s.prepare(v));
    const q = new Float32Array([2, 1, 0, -1, 3]);

    let dot = 0;
    let l2 = 0;
    for (let i = 0; i < dim; i++) {
      dot += v[i]! * q[i]!;
      const d = v[i]! - q[i]!;
      l2 += d * d;
    }
    expect(s.dotRow(0, q)).toBeCloseTo(dot, 4);
    expect(s.l2Row(0, q)).toBeCloseTo(-l2, 4);
  });

  it('survives raw-buffer growth past the initial capacity', () => {
    const dim = 8;
    const s = new Store(dim, 'dot', false);
    const rows = 3000;
    for (let r = 0; r < rows; r++) {
      const v = new Float32Array(dim).fill(r);
      s.insert(`id-${r}`, s.prepare(v));
    }
    expect(s.count).toBe(rows);
    expect(s.rawView().length).toBe(rows * dim);
    expect(s.vectorAt(2999)![0]).toBe(2999);
  });

  it('clear drops everything', () => {
    const s = new Store(2, 'dot', false);
    s.insert('a', s.prepare([1, 0]));
    s.delete('a');
    s.clear();
    expect(s.count).toBe(0);
    expect(s.rowCount).toBe(0);
    expect(s.deletedCount).toBe(0);
    expect(s.rawView().length).toBe(0);
  });
});
