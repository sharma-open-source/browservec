import { describe, it, expect } from 'vitest';
import {
  kmeans,
  normalizeRows,
  randomInitCentroids,
  updateCentroids,
} from '../../src/index/kmeans';
import { mulberry32 } from '../../src/quant/prng';

/** Sample with `k` well-separated Gaussian-ish blobs on axis-aligned centers. */
function blobs(k: number, perCluster: number, dim: number, seed: number) {
  const rng = mulberry32(seed);
  const n = k * perCluster;
  const data = new Float32Array(n * dim);
  const labels = new Int32Array(n);
  for (let c = 0; c < k; c++) {
    for (let j = 0; j < perCluster; j++) {
      const r = c * perCluster + j;
      labels[r] = c;
      for (let i = 0; i < dim; i++) {
        // Center: 10 on coordinate c (mod dim), noise ±0.5 elsewhere.
        const center = i === c % dim ? 10 : 0;
        data[r * dim + i] = center + (rng() - 0.5);
      }
    }
  }
  return { data, labels, n };
}

function nearest(data: Float32Array, row: number, centroids: Float32Array, nlist: number, dim: number) {
  let best = 0;
  let bestD = Infinity;
  for (let c = 0; c < nlist; c++) {
    let d = 0;
    for (let i = 0; i < dim; i++) {
      const t = data[row * dim + i]! - centroids[c * dim + i]!;
      d += t * t;
    }
    if (d < bestD) {
      bestD = d;
      best = c;
    }
  }
  return best;
}

describe('kmeans', () => {
  it('separates well-separated clusters (non-spherical / l2)', () => {
    const dim = 8;
    const k = 4;
    const { data, labels, n } = blobs(k, 50, dim, 1);
    const { centroids, nlist } = kmeans(data, n, { nlist: k, dim, seed: 7, spherical: false });
    expect(nlist).toBe(k);

    // Every point in a blob should map to the same centroid, and distinct blobs
    // to distinct centroids.
    const blobToCentroid = new Map<number, number>();
    for (let r = 0; r < n; r++) {
      const c = nearest(data, r, centroids, nlist, dim);
      const prev = blobToCentroid.get(labels[r]!);
      if (prev === undefined) blobToCentroid.set(labels[r]!, c);
      else expect(c).toBe(prev);
    }
    expect(new Set(blobToCentroid.values()).size).toBe(k);
  });

  it('is deterministic for a fixed seed', () => {
    const dim = 6;
    const { data, n } = blobs(3, 30, dim, 2);
    const a = kmeans(data, n, { nlist: 3, dim, seed: 42 });
    const b = kmeans(data, n, { nlist: 3, dim, seed: 42 });
    expect(a.centroids).toEqual(b.centroids);
  });

  it('normalizes centroids in spherical mode', () => {
    const dim = 5;
    const { data, n } = blobs(3, 30, dim, 3);
    const { centroids, nlist } = kmeans(data, n, { nlist: 3, dim, seed: 1, spherical: true });
    for (let c = 0; c < nlist; c++) {
      let s = 0;
      for (let i = 0; i < dim; i++) s += centroids[c * dim + i]! ** 2;
      expect(Math.sqrt(s)).toBeCloseTo(1, 4);
    }
  });

  it('clamps nlist to the sample count', () => {
    const dim = 4;
    const data = new Float32Array(3 * dim).fill(1);
    const { nlist } = kmeans(data, 3, { nlist: 16, dim });
    expect(nlist).toBe(3);
  });
});

describe('normalizeRows', () => {
  it('gives every row unit norm and leaves zero rows finite', () => {
    const m = new Float32Array([3, 4, 0, 0]);
    normalizeRows(m, 2, 2);
    expect(m[0]).toBeCloseTo(0.6, 5);
    expect(m[1]).toBeCloseTo(0.8, 5);
    expect(m[2]).toBe(0);
    expect(m[3]).toBe(0);
  });
});

describe('GPU-assisted Lloyd helpers', () => {
  it('randomInitCentroids picks sample rows deterministically', () => {
    const dim = 4;
    const { data, n } = blobs(2, 10, dim, 4);
    const a = randomInitCentroids(data, n, 3, dim, 9, false);
    const b = randomInitCentroids(data, n, 3, dim, 9, false);
    expect(a).toEqual(b);
    expect(a.length).toBe(3 * dim);
  });

  it('updateCentroids computes per-cluster means', () => {
    const dim = 2;
    // Two clusters: rows 0,1 -> cluster 0; row 2 -> cluster 1.
    const sample = new Float32Array([0, 0, 2, 2, 10, 10]);
    const assign = new Uint32Array([0, 0, 1]);
    const centroids = new Float32Array(2 * dim);
    updateCentroids(sample, 3, assign, centroids, 2, dim, 1, false);
    expect(Array.from(centroids)).toEqual([1, 1, 10, 10]);
  });

  it('updateCentroids re-seeds empty clusters from the sample', () => {
    const dim = 2;
    const sample = new Float32Array([1, 2, 3, 4]);
    const assign = new Uint32Array([0, 0]); // cluster 1 empty
    const centroids = new Float32Array(2 * dim);
    updateCentroids(sample, 2, assign, centroids, 2, dim, 5, false);
    // Cluster 1 must equal one of the sample rows.
    const c1 = [centroids[2], centroids[3]];
    expect([JSON.stringify([1, 2]), JSON.stringify([3, 4])]).toContain(JSON.stringify(c1));
  });
});
