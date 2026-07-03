// HNSW graph index (M7) — the VectorIndex seam over the CPU graph core.
//
// Where flat/IVF/quant indexes dispatch WGSL kernels, HNSW is inherently
// sequential pointer-chasing (each hop needs the previous hop's distances), so it
// runs on the CPU — which also makes it the first ANN index that works on the
// no-WebGPU fallback path. Following the kmeansTrainer pattern, the graph lives
// in a Web Worker when one is available so builds *and* searches stay off the
// main thread; otherwise the same HNSWGraph runs in-thread with identical,
// seeded-deterministic results. Callers never know which path ran (see
// `trainMode`, surfaced through stats().train like the IVF build).
//
// Unlike IVF there is no lazy rebuild: inserts extend the graph incrementally at
// append time, so the first query after a large ingest pays nothing extra.

import type { HNSWConfig, Metric } from '../types.js';
import type { DeviceContext } from '../engine/device.js';
import type { FlatHit, VectorIndex } from './flat.js';
import type { TrainMode } from './kmeansTrainer.js';
import { HNSWGraph, type HNSWGraphState } from './hnswGraph.js';
import { HNSWGpuSearcher } from './hnswGpu.js';
// Vite base64-inlines the worker into the main bundle (must be a *static* import —
// see the note in quant/encoder.ts). `tsc` resolves it via env.d.ts, and the built
// dist never constructs the Worker unless `typeof Worker !== 'undefined'`.
import InlineHNSWWorker from './hnsw.worker?worker&inline';

const DEFAULT_EF_SEARCH = 64;

interface Pending {
  resolve: (msg: WorkerReply) => void;
  reject: (e: unknown) => void;
}

type WorkerReply =
  | { type: 'appended'; id: number; maxLevel: number }
  | { type: 'hits'; id: number; rows: Int32Array; scores: Float32Array }
  | { type: 'graph'; id: number; links: Uint32Array; entry: number }
  | { type: 'state'; id: number; state: HNSWGraphState }
  | { type: 'loaded'; id: number; maxLevel: number };

export class HNSWIndex implements VectorIndex {
  private rows = 0;
  private readonly efSearch: number;
  private efOverride: number | undefined; // per-query override (see setEf)

  // Exactly one of these is live: the worker proxy or the in-thread graph.
  private worker: Worker | null = null;
  private graph: HNSWGraph | null = null;
  private topLevel = -1; // mirrored from the worker on each append ack
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  // GPU search path (M7b): the CPU/worker still builds the graph; the searcher
  // mirrors corpus + layer-0 adjacency on the GPU and runs the beam kernel.
  // Null when search:'cpu' (default), no device, or after a capacity fallback.
  private gpu: HNSWGpuSearcher | null = null;
  private entryNode = -1; // mirrored with each graph export

  /** Configured out-degree — persisted graphs must match to be restorable. */
  readonly M: number;

  constructor(
    private readonly dim: number,
    metric: Metric,
    params: HNSWConfig,
    ctx: DeviceContext | null = null,
  ) {
    this.M = Math.max(2, params.M ?? 16);
    this.efSearch = Math.max(1, params.efSearch ?? DEFAULT_EF_SEARCH);
    const graphParams = { M: params.M, efConstruction: params.efConstruction, seed: params.seed };
    if (params.search === 'gpu' && ctx) {
      // Throws early on degree > kernel limit (M ≤ 32) — a config error, not a
      // runtime condition, so it should not silently fall back.
      this.gpu = new HNSWGpuSearcher(ctx, dim, metric, 2 * Math.max(2, params.M ?? 16));
    }
    if (typeof Worker !== 'undefined') {
      try {
        const w = new InlineHNSWWorker();
        w.onmessage = (e: MessageEvent<WorkerReply>) => this.onMessage(e);
        w.onerror = () => this.failAll(new Error('hnsw worker crashed'));
        w.postMessage({ type: 'init', dim, metric, params: graphParams });
        this.worker = w;
      } catch {
        // No worker (CSP block, non-Vite runtime) — fall through to in-thread.
      }
    }
    if (!this.worker) this.graph = new HNSWGraph(dim, metric, params);
  }

  get size(): number {
    return this.rows;
  }

  /** Where the graph build (and search) runs — surfaced as stats().train. */
  get trainMode(): TrainMode {
    return this.worker ? 'worker' : 'main-thread';
  }

  /** Top graph layer (-1 while empty) — surfaced as stats().maxLevel. */
  get maxLevel(): number {
    return this.graph ? this.graph.maxLevel : this.topLevel;
  }

  /** Set the ef used by the next query() (consumed once), like IVF's setNprobe. */
  setEf(n: number | undefined): void {
    this.efOverride = n;
  }

  /** Which engine answers queries — surfaced as stats().graphSearch. */
  get searchMode(): 'gpu' | 'cpu' {
    return this.gpu ? 'gpu' : 'cpu';
  }

  append(data: Float32Array, count: number): void | Promise<void> {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    if (this.gpu) {
      try {
        this.gpu.append(data, count);
      } catch {
        // Corpus outgrew one storage buffer — the graph hops to arbitrary rows,
        // so it can't chunk like the linear scans. Fall back to CPU search for
        // good (the worker/in-thread graph is complete either way).
        this.gpu.destroy();
        this.gpu = null;
      }
    }
    this.rows += count;
    if (this.graph) {
      this.graph.append(data, count);
      return;
    }
    // The caller may reuse/retain `data`; transfer an owned copy zero-copy.
    const owned = data.slice();
    return this.rpc({ type: 'append', vectors: owned, count }, [owned.buffer]).then((msg) => {
      if (msg.type === 'appended') this.topLevel = msg.maxLevel;
    });
  }

  async query(queryVec: Float32Array, k: number): Promise<FlatHit[]> {
    if (this.rows === 0) return [];
    const ef = Math.max(this.efOverride ?? this.efSearch, k);
    this.efOverride = undefined;
    if (this.gpu) {
      await this.syncGpuGraph();
      const [hits] = await this.gpu.search(queryVec, 1, k, ef, this.entryNode);
      return hits!;
    }
    if (this.graph) return this.graph.search(queryVec, k, ef);
    // Worker messages are FIFO, so this search runs after every acked append.
    const msg = await this.rpc({ type: 'search', query: queryVec, k, ef });
    if (msg.type !== 'hits') throw new Error('hnsw worker protocol error');
    const hits: FlatHit[] = [];
    for (let i = 0; i < msg.rows.length; i++) {
      hits.push({ row: msg.rows[i]!, score: msg.scores[i]! });
    }
    return hits;
  }

  /**
   * Search `nQ` queries packed row-major in one call. On the GPU path this is a
   * SINGLE dispatch (one workgroup per query) — the regime where the kernel
   * beats the CPU walk, which pays per-query dispatch+readback otherwise.
   * CPU paths just loop. Duck-typed by BrowserVec.queryBatch.
   */
  async queryBatch(queries: Float32Array, nQ: number, k: number): Promise<FlatHit[][]> {
    if (queries.length !== nQ * this.dim) {
      throw new Error(`expected ${nQ * this.dim} floats, got ${queries.length}`);
    }
    const ef = Math.max(this.efOverride ?? this.efSearch, k);
    this.efOverride = undefined;
    if (this.rows === 0) return Array.from({ length: nQ }, () => []);
    if (this.gpu) {
      await this.syncGpuGraph();
      return this.gpu.search(queries, nQ, k, ef, this.entryNode);
    }
    const out: FlatHit[][] = [];
    for (let i = 0; i < nQ; i++) {
      const q = queries.subarray(i * this.dim, (i + 1) * this.dim);
      if (this.graph) {
        out.push(this.graph.search(q, k, ef));
      } else {
        const msg = await this.rpc({ type: 'search', query: q.slice(), k, ef });
        if (msg.type !== 'hits') throw new Error('hnsw worker protocol error');
        const hits: FlatHit[] = [];
        for (let j = 0; j < msg.rows.length; j++) hits.push({ row: msg.rows[j]!, score: msg.scores[j]! });
        out.push(hits);
      }
    }
    return out;
  }

  /** Snapshot the graph structure for persistence (M7c). Null while empty. */
  async exportGraphState(): Promise<HNSWGraphState | null> {
    if (this.rows === 0) return null;
    if (this.graph) return this.graph.serializeGraph();
    const msg = await this.rpc({ type: 'serialize' });
    if (msg.type !== 'state') throw new Error('hnsw worker protocol error');
    return msg.state;
  }

  /**
   * Restore a persisted graph + its vectors in one shot — the load-time
   * counterpart of append() that skips the O(N·efConstruction) rebuild. Only
   * valid on an empty index; the caller has already verified M matches.
   */
  async loadWithGraph(vectors: Float32Array, count: number, state: HNSWGraphState): Promise<void> {
    if (this.rows !== 0) throw new Error('loadWithGraph requires an empty index');
    if (this.gpu) {
      try {
        this.gpu.append(vectors, count);
      } catch {
        this.gpu.destroy();
        this.gpu = null;
      }
    }
    this.rows = count;
    if (this.graph) {
      this.graph.loadGraph(vectors, count, state);
      return;
    }
    const owned = vectors.slice();
    const msg = await this.rpc({ type: 'load', vectors: owned, count, state }, [
      owned.buffer,
      state.levels.buffer,
      state.links0.buffer,
      state.upper.buffer,
    ]);
    if (msg.type !== 'loaded') throw new Error('hnsw worker protocol error');
    this.topLevel = msg.maxLevel;
  }

  /** Re-upload the layer-0 adjacency when appends have outdated the GPU mirror. */
  private async syncGpuGraph(): Promise<void> {
    if (this.gpu!.uploadedRows === this.rows) return;
    if (this.graph) {
      this.entryNode = this.graph.entryNode;
      this.gpu!.uploadGraph(this.graph.exportLinks(), this.rows);
      return;
    }
    const msg = await this.rpc({ type: 'export' });
    if (msg.type !== 'graph') throw new Error('hnsw worker protocol error');
    this.entryNode = msg.entry;
    this.gpu!.uploadGraph(msg.links, this.rows);
  }

  private rpc(payload: Record<string, unknown>, transfer?: Transferable[]): Promise<WorkerReply> {
    const id = this.nextId++;
    return new Promise<WorkerReply>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ ...payload, id }, transfer ?? []);
    });
  }

  private onMessage(e: MessageEvent<WorkerReply>): void {
    const msg = e.data;
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    p.resolve(msg);
  }

  private failAll(err: unknown): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  destroy(): void {
    this.failAll(new Error('index destroyed'));
    this.worker?.terminate();
    this.worker = null;
    this.graph = null;
    this.gpu?.destroy();
    this.gpu = null;
    this.rows = 0;
  }
}
