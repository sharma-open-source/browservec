// HNSW graph core — the CPU engine behind the graph-based ANN index (M7).
//
// Hierarchical Navigable Small World (Malkov & Yashunin, 2016): every vector is a
// node in a layered proximity graph. Layer 0 holds all nodes with out-degree ≤ 2M;
// each higher layer keeps an exponentially thinner subset (out-degree ≤ M) that
// acts as an express lane. A query greedily descends from the sparse top layer to
// layer 0, then runs a best-first beam search of width `ef` — visiting O(log N)
// nodes instead of scanning all N. Unlike IVF there is no build phase to redo on
// append: inserts are incremental by construction.
//
// This is a deliberately pointer-chasing, sequential algorithm — the opposite of
// the batch-parallel GPU kernels — so it runs entirely on the CPU. The index seam
// (hnsw.ts) hosts this class in a Web Worker when one is available, keeping both
// build and search off the main thread; the fallback runs it in-thread. Both paths
// execute this exact code on the same insert order with a seeded RNG, so a build
// is reproducible regardless of where it ran (same story as the k-means trainer).
//
// Everything lives in flat typed arrays (no per-node objects): adjacency is
// count-prefixed Int32Array blocks, the visited set is a generation-stamped
// Int32Array (no clearing between searches), and the two beam heaps are reused
// scratch. Distances use the house unrolled-×4 loops.

import type { Metric } from '../types.js';
import type { FlatHit } from './flat.js';
import { mulberry32 } from '../quant/prng.js';

export interface HNSWGraphParams {
  /** Out-degree per upper layer; layer 0 keeps 2·M. Default 16. */
  M?: number;
  /** Beam width while inserting. Default 200. */
  efConstruction?: number;
  /** Seed for the level RNG (reproducible builds). */
  seed?: number;
}

/**
 * The graph's persistable structure (M7c) — everything except the vectors,
 * which the snapshot already stores. `upper` is the per-node upper-layer blocks
 * concatenated in row order; node n owns levels[n]·(M+1) of them, so the split
 * points reconstruct from `levels` alone.
 */
export interface HNSWGraphState {
  M: number;
  entry: number;
  top: number;
  levels: Int32Array; // rows
  links0: Int32Array; // rows * (2M+1)
  upper: Int32Array; // Σ levels[n]*(M+1)
}

// Levels are geometric with ratio 1/M — P(level ≥ 30) is astronomically small even
// at M=2; the cap just bounds the per-node upper-links allocation.
const MAX_LEVEL = 30;

/**
 * Binary min-heap over parallel (dist, id) arrays — no per-entry allocation. The
 * beam search needs a max-heap too (evict the worst kept result); callers get one
 * by pushing negated distances into a second MinHeap.
 */
class MinHeap {
  private dists: Float32Array;
  private ids: Int32Array;
  size = 0;

  constructor(cap = 256) {
    this.dists = new Float32Array(cap);
    this.ids = new Int32Array(cap);
  }

  clear(): void {
    this.size = 0;
  }

  push(d: number, id: number): void {
    if (this.size === this.dists.length) this.grow();
    const { dists, ids } = this;
    let i = this.size++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (dists[p]! <= d) break;
      dists[i] = dists[p]!;
      ids[i] = ids[p]!;
      i = p;
    }
    dists[i] = d;
    ids[i] = id;
  }

  peekDist(): number {
    return this.dists[0]!;
  }

  peekId(): number {
    return this.ids[0]!;
  }

  pop(): void {
    const { dists, ids } = this;
    const n = --this.size;
    const d = dists[n]!;
    const id = ids[n]!;
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && dists[c + 1]! < dists[c]!) c++;
      if (dists[c]! >= d) break;
      dists[i] = dists[c]!;
      ids[i] = ids[c]!;
      i = c;
    }
    dists[i] = d;
    ids[i] = id;
  }

  private grow(): void {
    const nd = new Float32Array(this.dists.length * 2);
    const ni = new Int32Array(this.ids.length * 2);
    nd.set(this.dists);
    ni.set(this.ids);
    this.dists = nd;
    this.ids = ni;
  }
}

export class HNSWGraph {
  private readonly M: number;
  private readonly M0: number; // layer-0 out-degree cap = 2·M
  private readonly efConstruction: number;
  private readonly mL: number; // level multiplier 1/ln(M)
  private readonly rng: () => number;

  // Corpus, packed row-major; grows geometrically like CpuIndex.
  private data = new Float32Array(0);
  private rows = 0;
  private cap = 0;

  // Graph. Layer-0 adjacency is one flat block per node: [count, id0..id_{M0-1}].
  // Upper layers exist for only ~1/M of nodes, so they get a per-node Int32Array
  // of `level` blocks shaped [count, id0..id_{M-1}] (block i = layer i+1).
  private levels = new Int32Array(0);
  private links0 = new Int32Array(0);
  private upper: (Int32Array | undefined)[] = [];
  private entry = -1;
  private top = -1;

  // Generation-stamped visited set: bump `visitGen` instead of clearing.
  private visited = new Int32Array(0);
  private visitGen = 0;

  // Reused scratch: beam heaps, sorted beam output, heuristic selection.
  private readonly cand = new MinHeap();
  private readonly res = new MinHeap(); // holds -dist → behaves as a max-heap
  private outDists = new Float32Array(0);
  private outIds = new Int32Array(0);
  private readonly selIds: Int32Array;
  private readonly selDists: Float32Array;

  constructor(
    private readonly dim: number,
    private readonly metric: Metric,
    params: HNSWGraphParams = {},
  ) {
    this.M = Math.max(2, params.M ?? 16);
    this.M0 = this.M * 2;
    this.efConstruction = Math.max(this.M, params.efConstruction ?? 200);
    this.mL = 1 / Math.log(this.M);
    this.rng = mulberry32((params.seed ?? 0x6d2b79f5) >>> 0);
    this.selIds = new Int32Array(this.M0);
    this.selDists = new Float32Array(this.M0);
  }

  get size(): number {
    return this.rows;
  }

  /** Top graph layer (-1 while empty). */
  get maxLevel(): number {
    return this.top;
  }

  /** Entry node of the graph (-1 while empty). */
  get entryNode(): number {
    return this.entry;
  }

  /** Layer-0 out-degree cap (2·M) — the fixed slot count of exportLinks(). */
  get degree(): number {
    return this.M0;
  }

  /** Snapshot the graph structure for persistence (M7c). Arrays are tight copies. */
  serializeGraph(): HNSWGraphState {
    let upperLen = 0;
    for (let r = 0; r < this.rows; r++) upperLen += this.levels[r]! * (this.M + 1);
    const upper = new Int32Array(upperLen);
    let w = 0;
    for (let r = 0; r < this.rows; r++) {
      const blocks = this.levels[r]! * (this.M + 1);
      if (blocks > 0) {
        upper.set(this.upper[r]!.subarray(0, blocks), w);
        w += blocks;
      }
    }
    return {
      M: this.M,
      entry: this.entry,
      top: this.top,
      levels: this.levels.slice(0, this.rows),
      links0: this.links0.slice(0, this.rows * (this.M0 + 1)),
      upper,
    };
  }

  /**
   * Restore a persisted graph over its vectors — the load-time counterpart of
   * append(), skipping the O(N·efC) rebuild entirely. Only valid on an empty
   * graph with a matching M (the caller checks and falls back to append()).
   * The level RNG is fast-forwarded one draw per restored row, so inserts after
   * a load draw the same levels they would have on the never-saved store.
   */
  loadGraph(vectors: Float32Array, count: number, state: HNSWGraphState): void {
    if (this.rows !== 0) throw new Error('loadGraph requires an empty graph');
    if (state.M !== this.M) throw new Error(`graph M ${state.M} != configured M ${this.M}`);
    if (vectors.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${vectors.length}`);
    }
    if (state.levels.length !== count || state.links0.length !== count * (this.M0 + 1)) {
      throw new Error('graph state does not match vector count');
    }
    this.ensureCapacity(count);
    this.data.set(vectors, 0);
    this.levels.set(state.levels, 0);
    this.links0.set(state.links0, 0);
    let off = 0;
    for (let r = 0; r < count; r++) {
      const blocks = state.levels[r]! * (this.M + 1);
      if (blocks > 0) {
        this.upper[r] = state.upper.slice(off, off + blocks);
        off += blocks;
      }
    }
    if (off !== state.upper.length) throw new Error('graph upper-links region does not match levels');
    this.entry = state.entry;
    this.top = state.top;
    this.rows = count;
    for (let i = 0; i < count; i++) this.rng();
  }

  /**
   * Export the layer-0 adjacency as a dense rows×2M table for the GPU kernel
   * (M7b): fixed slots per row, 0xFFFFFFFF-padded. The hierarchy stays CPU-side
   * — the flat bottom layer plus spread entry points is what the kernel walks.
   */
  exportLinks(): Uint32Array {
    const K = this.M0;
    const stride = K + 1;
    const out = new Uint32Array(this.rows * K).fill(0xffffffff);
    for (let r = 0; r < this.rows; r++) {
      const off = r * stride;
      const cnt = this.links0[off]!;
      for (let j = 0; j < cnt; j++) out[r * K + j] = this.links0[off + 1 + j]!;
    }
    return out;
  }

  /** Append `count` rows and insert each into the graph (incremental — no rebuild). */
  append(vectors: Float32Array, count: number): void {
    if (vectors.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${vectors.length}`);
    }
    this.ensureCapacity(this.rows + count);
    this.data.set(vectors, this.rows * this.dim);
    const first = this.rows;
    this.rows += count;
    for (let r = first; r < this.rows; r++) this.insert(r);
  }

  /**
   * k-NN search: greedy-descend the upper layers, beam-search layer 0 with width
   * max(ef, k), return the k best as FlatHits (score follows the library's
   * higher-is-closer convention for every metric).
   */
  search(query: Float32Array, k: number, ef: number): FlatHit[] {
    if (this.rows === 0) return [];
    let curr = this.entry;
    let currD = this.distToQuery(curr, query);
    for (let lc = this.top; lc >= 1; lc--) {
      let improved = true;
      while (improved) {
        improved = false;
        const { arr, off, cnt } = this.linksOf(curr, lc);
        for (let j = 1; j <= cnt; j++) {
          const e = arr[off + j]!;
          const d = this.distToQuery(e, query);
          if (d < currD) {
            curr = e;
            currD = d;
            improved = true;
          }
        }
      }
    }
    const n = this.searchLayer(query, curr, currD, Math.max(ef, k), 0);
    const hits: FlatHit[] = [];
    const take = Math.min(k, n);
    for (let i = 0; i < take; i++) {
      hits.push({ row: this.outIds[i]!, score: -this.outDists[i]! });
    }
    return hits;
  }

  // ---- Distances --------------------------------------------------------------
  // Internal convention: smaller = closer. cosine/dot use -dot (vectors arrive
  // normalized for cosine), l2 uses squared L2 — so score = -dist in every case,
  // matching FlatIndex/CpuIndex semantics exactly.

  private distToQuery(row: number, q: Float32Array): number {
    const { dim, data } = this;
    const base = row * dim;
    const tail = dim & ~3;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    let i = 0;
    if (this.metric === 'l2') {
      for (; i < tail; i += 4) {
        const d0 = data[base + i]! - q[i]!;
        const d1 = data[base + i + 1]! - q[i + 1]!;
        const d2 = data[base + i + 2]! - q[i + 2]!;
        const d3 = data[base + i + 3]! - q[i + 3]!;
        a0 += d0 * d0;
        a1 += d1 * d1;
        a2 += d2 * d2;
        a3 += d3 * d3;
      }
      let acc = a0 + a1 + a2 + a3;
      for (; i < dim; i++) {
        const d = data[base + i]! - q[i]!;
        acc += d * d;
      }
      return acc;
    }
    for (; i < tail; i += 4) {
      a0 += data[base + i]! * q[i]!;
      a1 += data[base + i + 1]! * q[i + 1]!;
      a2 += data[base + i + 2]! * q[i + 2]!;
      a3 += data[base + i + 3]! * q[i + 3]!;
    }
    let acc = a0 + a1 + a2 + a3;
    for (; i < dim; i++) acc += data[base + i]! * q[i]!;
    return -acc;
  }

  /** Row-to-row distance for the neighbor-selection heuristic — no subarray copies. */
  private distRows(a: number, b: number): number {
    const { dim, data } = this;
    const ba = a * dim;
    const bb = b * dim;
    const tail = dim & ~3;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    let i = 0;
    if (this.metric === 'l2') {
      for (; i < tail; i += 4) {
        const d0 = data[ba + i]! - data[bb + i]!;
        const d1 = data[ba + i + 1]! - data[bb + i + 1]!;
        const d2 = data[ba + i + 2]! - data[bb + i + 2]!;
        const d3 = data[ba + i + 3]! - data[bb + i + 3]!;
        a0 += d0 * d0;
        a1 += d1 * d1;
        a2 += d2 * d2;
        a3 += d3 * d3;
      }
      let acc = a0 + a1 + a2 + a3;
      for (; i < dim; i++) {
        const d = data[ba + i]! - data[bb + i]!;
        acc += d * d;
      }
      return acc;
    }
    for (; i < tail; i += 4) {
      a0 += data[ba + i]! * data[bb + i]!;
      a1 += data[ba + i + 1]! * data[bb + i + 1]!;
      a2 += data[ba + i + 2]! * data[bb + i + 2]!;
      a3 += data[ba + i + 3]! * data[bb + i + 3]!;
    }
    let acc = a0 + a1 + a2 + a3;
    for (; i < dim; i++) acc += data[ba + i]! * data[bb + i]!;
    return -acc;
  }

  // ---- Insert -----------------------------------------------------------------

  /** Geometric level draw: P(level ≥ l) = M^-l, so layer l holds ~N/M^l nodes. */
  private randomLevel(): number {
    const l = Math.floor(-Math.log(1 - this.rng()) * this.mL);
    return Math.min(l, MAX_LEVEL);
  }

  private insert(row: number): void {
    const q = this.data.subarray(row * this.dim, (row + 1) * this.dim);
    const level = this.randomLevel();
    this.levels[row] = level;
    if (level > 0) this.upper[row] = new Int32Array(level * (this.M + 1));

    if (this.entry === -1) {
      this.entry = row;
      this.top = level;
      return;
    }

    // Greedy-descend the layers above the new node's level to find a close entry.
    let curr = this.entry;
    let currD = this.distToQuery(curr, q);
    for (let lc = this.top; lc > level; lc--) {
      let improved = true;
      while (improved) {
        improved = false;
        const { arr, off, cnt } = this.linksOf(curr, lc);
        for (let j = 1; j <= cnt; j++) {
          const e = arr[off + j]!;
          const d = this.distToQuery(e, q);
          if (d < currD) {
            curr = e;
            currD = d;
            improved = true;
          }
        }
      }
    }

    // Beam-search each layer the node joins, pick diverse neighbors, link both ways.
    for (let lc = Math.min(level, this.top); lc >= 0; lc--) {
      const n = this.searchLayer(q, curr, currD, this.efConstruction, lc);
      // Closest beam result seeds the next layer down — captured now because
      // linkBack below reuses the out*/sel* scratch when a neighbor overflows.
      const nextCurr = this.outIds[0]!;
      const nextCurrD = this.outDists[0]!;
      const maxDeg = lc === 0 ? this.M0 : this.M;
      const selCount = this.selectHeuristic(n, this.M);
      const chosenIds: number[] = [];
      const chosenDists: number[] = [];
      for (let s = 0; s < selCount; s++) {
        chosenIds.push(this.selIds[s]!);
        chosenDists.push(this.selDists[s]!);
      }
      const { arr, off } = this.linksOf(row, lc);
      arr[off] = selCount;
      for (let s = 0; s < selCount; s++) arr[off + 1 + s] = chosenIds[s]!;
      for (let s = 0; s < selCount; s++) {
        this.linkBack(chosenIds[s]!, row, chosenDists[s]!, lc, maxDeg);
      }
      curr = nextCurr;
      currD = nextCurrD;
    }

    if (level > this.top) {
      this.entry = row;
      this.top = level;
    }
  }

  /**
   * Best-first beam search of width `ef` within one layer. Results land sorted
   * ascending-by-distance in outDists/outIds; returns how many. Classic two-heap
   * scheme: `cand` pops the closest unexpanded node, `res` (negated) evicts the
   * worst kept result; stop once the closest candidate is worse than the worst
   * kept result and the beam is full.
   */
  private searchLayer(q: Float32Array, ep: number, epD: number, ef: number, level: number): number {
    const { cand, res, visited } = this;
    cand.clear();
    res.clear();
    const gen = ++this.visitGen;
    visited[ep] = gen;
    cand.push(epD, ep);
    res.push(-epD, ep);

    while (cand.size > 0) {
      const cD = cand.peekDist();
      if (cD > -res.peekDist() && res.size >= ef) break;
      const c = cand.peekId();
      cand.pop();
      const { arr, off, cnt } = this.linksOf(c, level);
      for (let j = 1; j <= cnt; j++) {
        const e = arr[off + j]!;
        if (visited[e] === gen) continue;
        visited[e] = gen;
        const d = this.distToQuery(e, q);
        if (res.size < ef) {
          cand.push(d, e);
          res.push(-d, e);
        } else if (d < -res.peekDist()) {
          cand.push(d, e);
          res.push(-d, e);
          res.pop();
        }
      }
    }

    // Drain the max-heap back-to-front → ascending order, no sort.
    const n = res.size;
    if (this.outDists.length < n) {
      this.outDists = new Float32Array(Math.max(n, this.outDists.length * 2, 256));
      this.outIds = new Int32Array(this.outDists.length);
    }
    for (let i = n - 1; i >= 0; i--) {
      this.outDists[i] = -res.peekDist();
      this.outIds[i] = res.peekId();
      res.pop();
    }
    return n;
  }

  /**
   * The paper's diversity heuristic over the beam output (ascending in
   * outDists/outIds): keep a candidate only if it is closer to the query than to
   * every already-kept neighbor. Prevents all M edges from pointing into one
   * tight cluster, which is what keeps the graph navigable between clusters.
   * Fills selIds/selDists, returns the kept count.
   */
  private selectHeuristic(n: number, maxSel: number): number {
    let kept = 0;
    for (let i = 0; i < n && kept < maxSel; i++) {
      const c = this.outIds[i]!;
      const cD = this.outDists[i]!;
      let ok = true;
      for (let s = 0; s < kept; s++) {
        if (this.distRows(c, this.selIds[s]!) < cD) {
          ok = false;
          break;
        }
      }
      if (ok) {
        this.selIds[kept] = c;
        this.selDists[kept] = cD;
        kept++;
      }
    }
    return kept;
  }

  /**
   * Add `row` to `node`'s adjacency at `level`; when the list overflows `maxDeg`,
   * re-select the best `maxDeg` of (existing ∪ new) with the same diversity
   * heuristic, measured from `node`.
   */
  private linkBack(node: number, row: number, dNodeRow: number, level: number, maxDeg: number): void {
    const { arr, off } = this.linksOf(node, level);
    const cnt = arr[off]!;
    if (cnt < maxDeg) {
      arr[off + 1 + cnt] = row;
      arr[off] = cnt + 1;
      return;
    }
    // Overflow: rank candidates by distance to `node`, insertion-sorted ascending
    // into the shared out* scratch (cnt+1 ≤ M0+1 entries — tiny), then re-select.
    const total = cnt + 1;
    if (this.outDists.length < total) {
      this.outDists = new Float32Array(Math.max(total, 256));
      this.outIds = new Int32Array(this.outDists.length);
    }
    for (let j = 0; j <= cnt; j++) {
      const id = j < cnt ? arr[off + 1 + j]! : row;
      const d = j < cnt ? this.distRows(node, id) : dNodeRow;
      let p = j;
      while (p > 0 && this.outDists[p - 1]! > d) {
        this.outDists[p] = this.outDists[p - 1]!;
        this.outIds[p] = this.outIds[p - 1]!;
        p--;
      }
      this.outDists[p] = d;
      this.outIds[p] = id;
    }
    const kept = this.selectHeuristic(total, maxDeg);
    arr[off] = kept;
    for (let s = 0; s < kept; s++) arr[off + 1 + s] = this.selIds[s]!;
  }

  /** Locate `node`'s count-prefixed adjacency block at `level`. */
  private linksOf(node: number, level: number): { arr: Int32Array; off: number; cnt: number } {
    if (level === 0) {
      const off = node * (this.M0 + 1);
      return { arr: this.links0, off, cnt: this.links0[off]! };
    }
    const arr = this.upper[node]!;
    const off = (level - 1) * (this.M + 1);
    return { arr, off, cnt: arr[off]! };
  }

  private ensureCapacity(rows: number): void {
    if (rows <= this.cap) return;
    const next = Math.max(rows, Math.ceil(this.cap * 1.5), 1024);
    const grow = <T extends Float32Array | Int32Array>(old: T, len: number, make: (n: number) => T): T => {
      const a = make(len);
      a.set(old as never);
      return a;
    };
    this.data = grow(this.data, next * this.dim, (n) => new Float32Array(n));
    this.levels = grow(this.levels, next, (n) => new Int32Array(n));
    this.links0 = grow(this.links0, next * (this.M0 + 1), (n) => new Int32Array(n));
    // Stamps stay valid across growth: new slots are 0, which never equals a live gen.
    this.visited = grow(this.visited, next, (n) => new Int32Array(n));
    this.cap = next;
  }
}
