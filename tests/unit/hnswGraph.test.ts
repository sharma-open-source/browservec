import { describe, it, expect } from 'vitest';
import { HNSWGraph } from '../../src/index/hnswGraph';
import { mulberry32 } from '../../src/quant/prng';
import { normalizeInPlace } from '../../src/store/store';

function randomCorpus(rows: number, dim: number, seed: number, normalize = false): Float32Array {
  const rng = mulberry32(seed);
  const data = new Float32Array(rows * dim);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  if (normalize) {
    for (let r = 0; r < rows; r++) {
      normalizeInPlace(data.subarray(r * dim, (r + 1) * dim));
    }
  }
  return data;
}

/** Brute-force k-NN reference using the library's higher-is-closer scores. */
function bruteForce(
  data: Float32Array,
  rows: number,
  dim: number,
  q: Float32Array,
  k: number,
  metric: 'dot' | 'l2',
): number[] {
  const scored: Array<{ row: number; score: number }> = [];
  for (let r = 0; r < rows; r++) {
    let s = 0;
    if (metric === 'l2') {
      for (let i = 0; i < dim; i++) {
        const d = data[r * dim + i]! - q[i]!;
        s -= d * d;
      }
    } else {
      for (let i = 0; i < dim; i++) s += data[r * dim + i]! * q[i]!;
    }
    scored.push({ row: r, score: s });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map((h) => h.row);
}

function recallAt(graph: HNSWGraph, data: Float32Array, rows: number, dim: number, metric: 'dot' | 'l2', queries: number, k: number, ef: number): number {
  const rng = mulberry32(999);
  let hit = 0;
  for (let qi = 0; qi < queries; qi++) {
    const q = new Float32Array(dim);
    for (let i = 0; i < dim; i++) q[i] = rng() * 2 - 1;
    if (metric === 'dot') normalizeInPlace(q);
    const truth = new Set(bruteForce(data, rows, dim, q, k, metric));
    for (const h of graph.search(q, k, ef)) if (truth.has(h.row)) hit++;
  }
  return hit / (queries * k);
}

describe('HNSWGraph', () => {
  it('returns [] on an empty graph', () => {
    const g = new HNSWGraph(8, 'l2');
    expect(g.search(new Float32Array(8), 5, 64)).toEqual([]);
    expect(g.size).toBe(0);
    expect(g.maxLevel).toBe(-1);
  });

  it('finds an exact match and scores follow higher-is-closer', () => {
    const dim = 16;
    const rows = 200;
    const data = randomCorpus(rows, dim, 1);
    const g = new HNSWGraph(dim, 'l2', { seed: 1 });
    g.append(data, rows);
    expect(g.size).toBe(rows);

    const target = data.slice(37 * dim, 38 * dim);
    const hits = g.search(target, 3, 64);
    expect(hits[0]!.row).toBe(37);
    expect(hits[0]!.score).toBeCloseTo(0, 4); // -squared-L2 of itself
    expect(hits[0]!.score).toBeGreaterThanOrEqual(hits[1]!.score);
    expect(hits[1]!.score).toBeGreaterThanOrEqual(hits[2]!.score);
  });

  it('achieves high recall vs brute force (l2)', () => {
    const dim = 16;
    const rows = 600;
    const data = randomCorpus(rows, dim, 2);
    const g = new HNSWGraph(dim, 'l2', { seed: 2, M: 16, efConstruction: 200 });
    g.append(data, rows);
    expect(recallAt(g, data, rows, dim, 'l2', 20, 10, 128)).toBeGreaterThanOrEqual(0.9);
  });

  it('achieves high recall vs brute force (dot / normalized cosine)', () => {
    const dim = 16;
    const rows = 600;
    const data = randomCorpus(rows, dim, 3, true);
    const g = new HNSWGraph(dim, 'dot', { seed: 3 });
    g.append(data, rows);
    expect(recallAt(g, data, rows, dim, 'dot', 20, 10, 128)).toBeGreaterThanOrEqual(0.9);
  });

  it('supports incremental appends without a rebuild', () => {
    const dim = 8;
    const data = randomCorpus(300, dim, 4);
    const g = new HNSWGraph(dim, 'l2', { seed: 4 });
    for (let r = 0; r < 300; r += 50) g.append(data.subarray(r * dim, (r + 50) * dim), 50);
    expect(g.size).toBe(300);
    expect(recallAt(g, data, 300, dim, 'l2', 10, 5, 128)).toBeGreaterThanOrEqual(0.9);
  });

  it('builds reproducibly from the same seed', () => {
    const dim = 8;
    const data = randomCorpus(200, dim, 5);
    const a = new HNSWGraph(dim, 'l2', { seed: 42 });
    const b = new HNSWGraph(dim, 'l2', { seed: 42 });
    a.append(data, 200);
    b.append(data, 200);
    const sa = a.serializeGraph();
    const sb = b.serializeGraph();
    expect(sa.levels).toEqual(sb.levels);
    expect(sa.links0).toEqual(sb.links0);
    expect(sa.upper).toEqual(sb.upper);
    expect(sa.entry).toBe(sb.entry);
  });

  it('serialize + loadGraph round-trips to identical search results', () => {
    const dim = 12;
    const rows = 400;
    const data = randomCorpus(rows, dim, 6);
    const g = new HNSWGraph(dim, 'l2', { seed: 6, M: 8 });
    g.append(data, rows);
    const state = g.serializeGraph();

    const g2 = new HNSWGraph(dim, 'l2', { seed: 6, M: 8 });
    g2.loadGraph(data, rows, state);
    expect(g2.size).toBe(rows);
    expect(g2.maxLevel).toBe(g.maxLevel);
    expect(g2.entryNode).toBe(g.entryNode);

    const rng = mulberry32(7);
    for (let qi = 0; qi < 10; qi++) {
      const q = new Float32Array(dim);
      for (let i = 0; i < dim; i++) q[i] = rng() * 2 - 1;
      expect(g2.search(q, 10, 64)).toEqual(g.search(q, 10, 64));
    }
  });

  it('after loadGraph, further appends match the never-saved store (RNG fast-forward)', () => {
    const dim = 8;
    const data = randomCorpus(300, dim, 8);
    const first = data.subarray(0, 200 * dim);
    const rest = data.subarray(200 * dim);

    const never = new HNSWGraph(dim, 'l2', { seed: 9 });
    never.append(first, 200);
    never.append(rest, 100);

    const saved = new HNSWGraph(dim, 'l2', { seed: 9 });
    saved.append(first, 200);
    const reloaded = new HNSWGraph(dim, 'l2', { seed: 9 });
    reloaded.loadGraph(data.slice(0, 200 * dim), 200, saved.serializeGraph());
    reloaded.append(rest, 100);

    const sa = never.serializeGraph();
    const sb = reloaded.serializeGraph();
    expect(sb.levels).toEqual(sa.levels);
    expect(sb.links0).toEqual(sa.links0);
  });

  it('loadGraph validates its inputs', () => {
    const dim = 4;
    const data = randomCorpus(10, dim, 10);
    const g = new HNSWGraph(dim, 'l2', { seed: 1, M: 8 });
    g.append(data, 10);
    const state = g.serializeGraph();

    const nonEmpty = new HNSWGraph(dim, 'l2', { M: 8 });
    nonEmpty.append(data, 10);
    expect(() => nonEmpty.loadGraph(data, 10, state)).toThrow(/empty graph/);

    const wrongM = new HNSWGraph(dim, 'l2', { M: 4 });
    expect(() => wrongM.loadGraph(data, 10, state)).toThrow(/M 8 != configured M 4/);

    const fresh = new HNSWGraph(dim, 'l2', { M: 8 });
    expect(() => fresh.loadGraph(data.subarray(0, 9 * dim), 9, state)).toThrow(/does not match/);
  });

  it('exportLinks produces a dense 2M-wide table padded with 0xFFFFFFFF', () => {
    const dim = 4;
    const M = 4;
    const rows = 20;
    const g = new HNSWGraph(dim, 'l2', { seed: 1, M });
    g.append(randomCorpus(rows, dim, 11), rows);
    const links = g.exportLinks();
    expect(g.degree).toBe(2 * M);
    expect(links.length).toBe(rows * 2 * M);
    const state = g.serializeGraph();
    for (let r = 0; r < rows; r++) {
      const cnt = state.links0[r * (2 * M + 1)]!;
      for (let j = 0; j < 2 * M; j++) {
        const v = links[r * 2 * M + j]!;
        if (j < cnt) {
          expect(v).toBeLessThan(rows);
          expect(v).toBe(state.links0[r * (2 * M + 1) + 1 + j]! >>> 0);
        } else {
          expect(v).toBe(0xffffffff);
        }
      }
    }
  });

  it('validates append length', () => {
    const g = new HNSWGraph(8, 'l2');
    expect(() => g.append(new Float32Array(15), 2)).toThrow(/expected 16 floats/);
  });
});
