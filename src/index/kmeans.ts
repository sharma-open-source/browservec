// CPU k-means for IVF coarse quantization.
//
// Trains `nlist` centroids on a sample of the corpus. We only ever run this on a
// bounded reservoir sample (not the whole corpus), so a simple single-threaded
// Lloyd's algorithm with k-means++ seeding is fast enough and keeps the build
// allocation-light. For cosine/dot we use *spherical* k-means: centroids are
// re-normalized each iteration so that "nearest by dot" partitions the unit
// sphere the same way the search metric does.

import { mulberry32 } from '../quant/prng.js';

export interface KMeansResult {
  /** nlist * dim, row-major. Normalized when `spherical`. */
  centroids: Float32Array;
  nlist: number;
}

export interface KMeansOptions {
  nlist: number;
  dim: number;
  iters?: number; // Lloyd iterations; default 12
  seed?: number;
  /** Normalize centroids each step (cosine/dot). Default true. */
  spherical?: boolean;
}

/**
 * Train centroids on `sample` (sampleCount * dim, row-major). `sample` may hold
 * more rows than centroids; if it holds fewer we clamp nlist to the row count.
 */
export function kmeans(sample: Float32Array, sampleCount: number, opts: KMeansOptions): KMeansResult {
  const { dim } = opts;
  const iters = opts.iters ?? 12;
  const spherical = opts.spherical ?? true;
  const rng = mulberry32((opts.seed ?? 0x51ed270b) >>> 0);
  const nlist = Math.max(1, Math.min(opts.nlist, sampleCount));

  const centroids = kmeansppInit(sample, sampleCount, dim, nlist, rng);
  if (spherical) normalizeRows(centroids, nlist, dim);

  const sums = new Float32Array(nlist * dim);
  const counts = new Int32Array(nlist);
  const assign = new Int32Array(sampleCount);

  for (let it = 0; it < iters; it++) {
    sums.fill(0);
    counts.fill(0);
    let moved = 0;

    for (let r = 0; r < sampleCount; r++) {
      const c = nearestCentroid(sample, r * dim, centroids, nlist, dim);
      if (c !== assign[r]) {
        moved++;
        assign[r] = c;
      }
      const base = r * dim;
      const cb = c * dim;
      for (let i = 0; i < dim; i++) sums[cb + i]! += sample[base + i]!;
      counts[c]!++;
    }

    for (let c = 0; c < nlist; c++) {
      const cb = c * dim;
      if (counts[c] === 0) {
        // Empty cluster: re-seed it from a random sample row to keep nlist live.
        const r = Math.floor(rng() * sampleCount);
        for (let i = 0; i < dim; i++) centroids[cb + i] = sample[r * dim + i]!;
      } else {
        const inv = 1 / counts[c]!;
        for (let i = 0; i < dim; i++) centroids[cb + i] = sums[cb + i]! * inv;
      }
    }
    if (spherical) normalizeRows(centroids, nlist, dim);

    // Converged once assignments stop changing.
    if (it > 0 && moved === 0) break;
  }

  return { centroids, nlist };
}

/** k-means++ seeding: spread initial centroids by D² sampling. */
function kmeansppInit(
  sample: Float32Array,
  n: number,
  dim: number,
  k: number,
  rng: () => number,
): Float32Array {
  const centroids = new Float32Array(k * dim);
  // First centroid: a uniformly random row.
  const first = Math.floor(rng() * n);
  copyRow(sample, first, centroids, 0, dim);

  const d2 = new Float32Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    // Update nearest-centroid squared distance using the centroid we just added.
    let total = 0;
    const prev = (c - 1) * dim;
    for (let r = 0; r < n; r++) {
      const dist = sqDist(sample, r * dim, centroids, prev, dim);
      if (dist < d2[r]!) d2[r] = dist;
      total += d2[r]!;
    }
    // Sample the next centroid with probability proportional to D².
    let target = rng() * total;
    let chosen = n - 1;
    for (let r = 0; r < n; r++) {
      target -= d2[r]!;
      if (target <= 0) {
        chosen = r;
        break;
      }
    }
    copyRow(sample, chosen, centroids, c, dim);
  }
  return centroids;
}

function nearestCentroid(
  vecs: Float32Array,
  base: number,
  centroids: Float32Array,
  nlist: number,
  dim: number,
): number {
  let best = 0;
  let bestDist = Infinity;
  for (let c = 0; c < nlist; c++) {
    const d = sqDist(vecs, base, centroids, c * dim, dim);
    if (d < bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return best;
}

function sqDist(a: Float32Array, ai: number, b: Float32Array, bi: number, dim: number): number {
  let s = 0;
  for (let i = 0; i < dim; i++) {
    const d = a[ai + i]! - b[bi + i]!;
    s += d * d;
  }
  return s;
}

// ---- Pieces used by the GPU-assisted Lloyd loop in IVFIndex --------------------
// At dim≈768 a pure-CPU k-means is far too slow in a browser, so IVFIndex runs the
// assignment step (the O(sample·nlist·dim) cost) on the GPU and calls these cheap
// CPU helpers for init and the centroid-mean update.

/** Random init: pick `nlist` distinct-ish sample rows as starting centroids. */
export function randomInitCentroids(
  sample: Float32Array,
  sampleCount: number,
  nlist: number,
  dim: number,
  seed: number,
  spherical = true,
): Float32Array {
  const rng = mulberry32(seed >>> 0);
  const centroids = new Float32Array(nlist * dim);
  for (let c = 0; c < nlist; c++) {
    const r = Math.floor(rng() * sampleCount);
    copyRow(sample, r, centroids, c, dim);
  }
  if (spherical) normalizeRows(centroids, nlist, dim);
  return centroids;
}

/**
 * Recompute centroids as the (normalized) mean of the sample rows assigned to each
 * cluster, given GPU-produced assignments. Empty clusters are re-seeded from a
 * random sample row so nlist stays live. Mutates and returns `centroids`.
 */
export function updateCentroids(
  sample: Float32Array,
  sampleCount: number,
  assign: Uint32Array,
  centroids: Float32Array,
  nlist: number,
  dim: number,
  seed: number,
  spherical = true,
): Float32Array {
  const sums = new Float32Array(nlist * dim);
  const counts = new Int32Array(nlist);
  for (let r = 0; r < sampleCount; r++) {
    const c = assign[r]!;
    const base = r * dim;
    const cb = c * dim;
    for (let i = 0; i < dim; i++) sums[cb + i]! += sample[base + i]!;
    counts[c]!++;
  }
  const rng = mulberry32(seed >>> 0);
  for (let c = 0; c < nlist; c++) {
    const cb = c * dim;
    if (counts[c] === 0) {
      const r = Math.floor(rng() * sampleCount);
      for (let i = 0; i < dim; i++) centroids[cb + i] = sample[r * dim + i]!;
    } else {
      const inv = 1 / counts[c]!;
      for (let i = 0; i < dim; i++) centroids[cb + i] = sums[cb + i]! * inv;
    }
  }
  if (spherical) normalizeRows(centroids, nlist, dim);
  return centroids;
}

export function normalizeRows(m: Float32Array, rows: number, dim: number): void {
  for (let r = 0; r < rows; r++) {
    const base = r * dim;
    let s = 0;
    for (let i = 0; i < dim; i++) s += m[base + i]! * m[base + i]!;
    const inv = 1 / (Math.sqrt(s) || 1);
    for (let i = 0; i < dim; i++) m[base + i]! *= inv;
  }
}

function copyRow(src: Float32Array, srcRow: number, dst: Float32Array, dstRow: number, dim: number): void {
  dst.set(src.subarray(srcRow * dim, srcRow * dim + dim), dstRow * dim);
}
