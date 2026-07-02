// BrowserVec public entry point.
//
// Implemented: (WebGPU flat brute-force, in-memory, cosine/dot/l2, configurable
// dim, CPU top-k) + (OPFS/IndexedDB persistence, versioned snapshot, export/import).
// plus TurboQuant, IVF, on-device embedder, GPU top-k, and a CPU fallback
// (fallback: 'wasm') that keeps exact flat search working with no WebGPU.

import type {
  BrowserVecConfig,
  Embedder,
  EncryptionConfig,
  ExportOptions,
  Metric,
  PersistConfig,
  QueryOptions,
  QueryResult,
  Stats,
  SupportInfo,
  TextRecord,
  Vector,
  VectorRecord,
} from './types.js';
import {
  acquireDevice,
  isWebGPUAvailable,
  WebGPUUnavailableError,
  type DeviceContext,
} from './engine/device.js';
import { FlatIndex, topK, type FlatHit, type VectorIndex } from './index/flat.js';
import { CpuIndex } from './fallback/cpu.js';
import { QuantIndex } from './index/quant.js';
import { IVFIndex } from './index/ivf.js';
import { IVFQuantIndex } from './index/ivfquant.js';
import { Store, normalizeInPlace } from './store/store.js';
import { queryTrace } from './engine/profile.js';
import { selectBackend, type PersistenceBackend } from './persist/backend.js';
import { serialize, deserialize, type Snapshot } from './persist/format.js';
import { encryptSnapshot, decryptSnapshot, isEncrypted } from './persist/crypto.js';

export type {
  BrowserVecConfig,
  PersistConfig,
  EncryptionConfig,
  ExportOptions,
  QuantConfig,
  ANNConfig,
  Embedder,
  TextRecord,
  Metric,
  QueryOptions,
  QueryResult,
  Stats,
  SupportInfo,
  Vector,
  VectorRecord,
  Metadata,
} from './types.js';

export { hashingEmbedder } from './embed/hashing.js';
export type { HashingEmbedderOptions } from './embed/hashing.js';
export { transformersEmbedder } from './embed/transformers.js';
export type { TransformersEmbedderOptions } from './embed/transformers.js';

const DEFAULT_SEED = 0x9e3779b9;

// Decrypt a persisted/imported blob when it's an encrypted envelope. An encrypted
// blob without a passphrase, or a passphrase against a plaintext blob, is a clear
// mismatch worth erroring on rather than silently mis-loading.
async function decryptIfNeeded(buf: ArrayBuffer, passphrase: string | undefined): Promise<ArrayBuffer> {
  const encrypted = isEncrypted(buf);
  if (encrypted && !passphrase) {
    throw new Error('snapshot is encrypted; pass { encryption: { passphrase } } to read it');
  }
  if (!encrypted && passphrase) {
    throw new Error('a passphrase was provided but the snapshot is not encrypted');
  }
  return encrypted ? decryptSnapshot(buf, passphrase!) : buf;
}

// int4 scores are coarse (15 levels), so on clustered/near-duplicate data the true
// neighbours cluster into near-ties and need a wider re-rank pool to survive. int8
// is fine with the standard 4×.
function defaultRerankFactor(quantBits: 0 | 1 | 4 | 8): number {
  // Coarser codes need a wider exact-rerank pool: 1-bit is coarsest of all.
  if (quantBits === 1) return 32;
  return quantBits === 4 ? 16 : 4;
}

function buildIndex(
  ctx: DeviceContext,
  dim: number,
  metric: Metric,
  quantBits: 0 | 1 | 4 | 8,
  quant: BrowserVecConfig['quant'],
  ann: BrowserVecConfig['ann'],
  chunkRows?: number,
): VectorIndex {
  const seed = quant?.seed ?? DEFAULT_SEED;
  const rounds = quant?.rounds ?? 2;
  if (ann) {
    if (metric === 'l2') {
      throw new Error('IVF supports metric cosine/dot only (l2 is later work)');
    }
    if (quantBits === 8 || quantBits === 4 || quantBits === 1) {
      return new IVFQuantIndex(ctx, dim, seed, rounds, ann, quantBits, chunkRows);
    }
    if (quantBits === 0) return new IVFIndex(ctx, dim, metric, ann, chunkRows);
    throw new Error(`quantBits ${quantBits} not supported with ann`);
  }
  if (quantBits === 0) return new FlatIndex(ctx, dim, metric, chunkRows);
  if (quantBits === 8 || quantBits === 4 || quantBits === 1) {
    if (metric === 'l2') {
      throw new Error('quantized mode supports metric cosine/dot only (l2 quantization is later work)');
    }
    return new QuantIndex(ctx, dim, seed, rounds, quantBits, chunkRows);
  }
  throw new Error(`quantBits ${quantBits} not supported (ships 0, 1, 4, 8)`);
}

// Exact CPU flat index for the no-WebGPU fallback (§NFR-7). Quantization and IVF
// are GPU-throughput optimizations that add nothing to an exact CPU scan, so they
// aren't ported to the fallback yet — asking for them without a GPU is an error
// rather than a silent accuracy change.
function buildCpuIndex(dim: number, metric: Metric, quantBits: 0 | 1 | 4 | 8, ann: BrowserVecConfig['ann']): VectorIndex {
  if (quantBits !== 0 || ann) {
    throw new Error(
      'CPU fallback supports fp32 flat only — quantBits/ann require WebGPU. ' +
        'Drop them, or run where WebGPU is available.',
    );
  }
  return new CpuIndex(dim, metric);
}

// Test-only seam: force the CPU path even where WebGPU exists, so the fallback can
// be exercised in a GPU-capable browser (the demo sets this). Never set in prod.
function forceCpu(): boolean {
  return (
    typeof globalThis !== 'undefined' &&
    (globalThis as Record<string, unknown>).__BROWSERVEC_FORCE_CPU__ === true
  );
}

// Acquire a GPU context, or return null to signal the CPU fallback. Null happens
// when WebGPU is genuinely absent and the caller opted into 'wasm', or under the
// force-CPU test seam. A real GPU error with fallback:'error' still throws.
async function acquireContext(
  device: GPUDevice | undefined,
  fallback: 'wasm' | 'error',
): Promise<DeviceContext | null> {
  if (forceCpu()) return null;
  try {
    return await acquireDevice(device);
  } catch (e) {
    if (fallback === 'wasm' && e instanceof WebGPUUnavailableError) return null;
    throw e;
  }
}

// One factory used for both initial build and compact()'s rebuild: GPU indexes
// when a context exists, the CPU index when it doesn't.
function indexFactory(
  ctx: DeviceContext | null,
  dim: number,
  metric: Metric,
  quantBits: 0 | 1 | 4 | 8,
  quant: BrowserVecConfig['quant'],
  ann: BrowserVecConfig['ann'],
  chunkRows: number | undefined,
): () => VectorIndex {
  return ctx
    ? () => buildIndex(ctx, dim, metric, quantBits, quant, ann, chunkRows)
    : () => buildCpuIndex(dim, metric, quantBits, ann);
}

export class BrowserVec {
  private lastQueryMs: number | undefined;
  private lastQueryGpuMs: number | undefined;
  private backend: PersistenceBackend | undefined;
  private persistName: string | undefined;
  private persistPassphrase: string | undefined;
  private quantBits: 0 | 1 | 4 | 8 = 0;
  private rerankFactor = 4;
  private embedder: Embedder | undefined;

  // Rebuilds a fresh, empty index with the same config — used by compact().
  private makeIndex: (() => VectorIndex) | undefined;

  private constructor(
    // Null when running the CPU fallback (§NFR-7) — no GPU device to own.
    private readonly ctx: DeviceContext | null,
    private readonly store: Store,
    private index: VectorIndex,
  ) {}

  /** Capability probe (BrowserVec.isSupported). */
  static isSupported(): SupportInfo {
    return {
      webgpu: isWebGPUAvailable(),
      opfs: typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory,
      wasm: typeof WebAssembly !== 'undefined',
    };
  }

  static async create(config: BrowserVecConfig): Promise<BrowserVec> {
    if (!Number.isInteger(config.dimension) || config.dimension <= 0) {
      throw new Error(`dimension must be a positive integer, got ${config.dimension}`);
    }
    const metric: Metric = config.metric ?? 'cosine';
    const normalize = config.normalize ?? metric === 'cosine';

    const quantBits = config.quantBits ?? 0;
    const ctx = await acquireContext(config.device, config.fallback ?? 'error');
    const store = new Store(config.dimension, metric, normalize);
    const makeIndex = indexFactory(
      ctx, config.dimension, metric, quantBits, config.quant, config.ann, config.chunkRows,
    );
    const index = makeIndex();

    const db = new BrowserVec(ctx, store, index);
    db.quantBits = quantBits;
    db.rerankFactor = config.quant?.rerankFactor ?? defaultRerankFactor(quantBits);
    db.makeIndex = makeIndex;

    if (config.embedder) {
      if (config.embedder.dimension !== config.dimension) {
        throw new Error(
          `embedder dimension ${config.embedder.dimension} != store dimension ${config.dimension}`,
        );
      }
      db.embedder = config.embedder;
    }

    if (config.persist) await db.initPersistence(config.persist);
    return db;
  }

  private async initPersistence(cfg: PersistConfig): Promise<void> {
    this.backend = selectBackend(cfg.backend ?? 'auto');
    this.persistName = cfg.name;
    this.persistPassphrase = cfg.encryption?.passphrase;
    if (cfg.autoLoad === false) return;

    const buf = await this.backend.read(cfg.name);
    if (buf) await this.loadSnapshot(deserialize(await decryptIfNeeded(buf, this.persistPassphrase)));
  }

  get dimension(): number {
    return this.store.dimension;
  }
  get count(): number {
    return this.store.count;
  }
  get metric(): Metric {
    return this.store.metric;
  }

  async add(record: VectorRecord): Promise<void> {
    const vec = this.store.prepare(record.vector);
    const row = this.store.insert(record.id, vec, record.metadata);
    if (row !== this.index.size) throw new Error('internal: store/index row mismatch');
    await this.index.append(vec, 1);
  }

  /** Bulk insert — the primary ingestion path. Packs into one upload. */
  async addBatch(records: VectorRecord[]): Promise<void> {
    if (records.length === 0) return;
    const dim = this.store.dimension;
    const packed = new Float32Array(records.length * dim);
    for (let i = 0; i < records.length; i++) {
      const r = records[i]!;
      const vec = this.store.prepare(r.vector);
      const row = this.store.insert(r.id, vec, r.metadata);
      if (row !== this.index.size + i) throw new Error('internal: row mismatch in batch');
      packed.set(vec, i * dim);
    }
    await this.index.append(packed, records.length);
  }

  // ---- Text convenience (M5) ------------------------------------------------

  private requireEmbedder(): Embedder {
    if (!this.embedder) {
      throw new Error('text methods require an embedder; pass { embedder } to create()');
    }
    return this.embedder;
  }

  /** Embed `text` on-device and insert it (FR — text ingestion). */
  async addText(record: TextRecord): Promise<void> {
    const [vector] = await this.requireEmbedder().embed([record.text]);
    const r: VectorRecord = { id: record.id, vector: vector! };
    if (record.metadata !== undefined) r.metadata = record.metadata;
    await this.add(r);
  }

  /** Embed and insert a batch of texts in one model call + one GPU upload. */
  async addTexts(records: TextRecord[]): Promise<void> {
    if (records.length === 0) return;
    const vectors = await this.requireEmbedder().embed(records.map((r) => r.text));
    await this.addBatch(
      records.map((r, i) => {
        const out: VectorRecord = { id: r.id, vector: vectors[i]! };
        if (r.metadata !== undefined) out.metadata = r.metadata;
        return out;
      }),
    );
  }

  /** Embed `text` and run a query (the offline text→retrieval path). */
  async queryText(text: string, opts: QueryOptions = {}): Promise<QueryResult[]> {
    const [vector] = await this.requireEmbedder().embed([text]);
    return this.query(vector!, opts);
  }

  async query(query: Vector, opts: QueryOptions = {}): Promise<QueryResult[]> {
    const k = opts.k ?? 10;
    const q = query instanceof Float32Array ? new Float32Array(query) : Float32Array.from(query);
    if (q.length !== this.store.dimension) {
      throw new Error(`query dim ${q.length} != store dim ${this.store.dimension}`);
    }
    if (this.store.normalize) normalizeInPlace(q);

    // IVF: let a per-query nprobe override the configured default.
    if (opts.nprobe !== undefined && 'setNprobe' in this.index) {
      (this.index as { setNprobe(n: number): void }).setNprobe(opts.nprobe);
    }

    // Deleted rows are tombstones still scored by the GPU, so over-fetch by the
    // deleted count and filter them out — guarantees ≥k live results survive.
    // When nothing is deleted, kEff == k and this is a no-op (unchanged path).
    const deleted = this.store.deletedCount;
    const total = this.index.size;
    const kEff = Math.min(total, k + deleted);

    queryTrace.reset();
    const start = performance.now();
    let hits: FlatHit[];
    if (this.quantBits > 0) {
      const doRerank = opts.rerank ?? this.rerankFactor > 1;
      const fetch = doRerank ? Math.min(total, Math.max(kEff, kEff * this.rerankFactor)) : kEff;
      const approx = await this.index.query(q, fetch);
      hits = doRerank ? this.rerankExact(q, approx, kEff) : approx.slice(0, kEff);
    } else {
      hits = await this.index.query(q, kEff);
    }
    this.lastQueryMs = performance.now() - start;
    this.lastQueryGpuMs = queryTrace.gpuWaitMs;

    const results: QueryResult[] = [];
    for (const h of hits) {
      if (deleted && this.store.isDeleted(h.row)) continue;
      const entry = this.store.entryByRow(h.row)!;
      const out: QueryResult = { id: entry.id, score: h.score };
      if (entry.metadata !== undefined) out.metadata = entry.metadata;
      results.push(out);
      if (results.length === k) break;
    }
    return results;
  }

  /**
   * Delete a vector by id. Returns false if the id isn't present. The row is
   * tombstoned (filtered from results immediately) and its GPU memory is reclaimed
   * the next time the store is persisted + reloaded (snapshots are compacted).
   */
  delete(id: string): boolean {
    return this.store.delete(id);
  }

  /**
   * Replace the vector (and metadata) for `id`, or insert it if new (upsert).
   * Returns true if an existing row was replaced. Implemented as tombstone + append,
   * so repeated updates accumulate tombstones — call {@link compact} to reclaim them.
   */
  async update(record: VectorRecord): Promise<boolean> {
    const existed = this.store.delete(record.id);
    await this.add(record);
    return existed;
  }

  /**
   * Physically drop tombstoned rows: rebuild the store and GPU index from the live
   * vectors only, reclaiming their memory without a save/reload round-trip. Returns
   * the number of rows removed. No-op (returns 0) when nothing is deleted.
   */
  async compact(): Promise<number> {
    const removed = this.store.deletedCount;
    if (removed === 0) return 0;
    if (!this.makeIndex) throw new Error('internal: compact() needs an index factory');

    const dim = this.store.dimension;
    const entries = this.store.liveEntries();
    const vectors = this.store.liveVectors(); // compacted, already prepared/normalized

    // Rebuild the store dense (rows 0..live-1, no tombstones).
    this.store.clear();
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i]!;
      this.store.insert(e.id, vectors.subarray(i * dim, i * dim + dim), e.metadata);
    }

    // Rebuild the GPU index from scratch and re-ingest the live vectors.
    this.index.destroy();
    this.index = this.makeIndex();
    if (entries.length > 0) await this.index.append(vectors, entries.length);
    return removed;
  }

  /** Fetch a stored record by id, including its vector. */
  get(id: string): VectorRecord | null {
    const entry = this.store.entryById(id);
    if (!entry) return null;
    const vector = this.store.vectorAt(entry.row)!;
    const out: VectorRecord = { id, vector };
    if (entry.metadata !== undefined) out.metadata = entry.metadata;
    return out;
  }

  /**
   * Re-score approximate candidates against the exact fp32 vectors and keep the
   * true top-k. The quantized kernel narrows 1M rows
   * to a few hundred candidates cheaply; exact scoring those few recovers recall.
   * Valid for cosine/dot (the only quantized metrics): exact score = dot product.
   */
  private rerankExact(q: Float32Array, candidates: FlatHit[], k: number): FlatHit[] {
    const scores = new Float32Array(candidates.length);
    for (let c = 0; c < candidates.length; c++) {
      // Score straight from the store's packed buffer — no per-candidate copy.
      scores[c] = this.store.dotRow(candidates[c]!.row, q);
    }
    // topK returns rows as indices into `scores`; remap to original corpus rows.
    return topK(scores, candidates.length, k).map((h) => ({
      row: candidates[h.row]!.row,
      score: h.score,
    }));
  }

  // ---- Persistence (M2) -----------------------------------------------------

  /** Flush the current store to the configured persistence backend. */
  async save(): Promise<void> {
    if (!this.backend || !this.persistName) {
      throw new Error('save() requires a persist config; pass { persist: { name } } to create()');
    }
    let bytes = this.snapshotBytes();
    if (this.persistPassphrase) bytes = await encryptSnapshot(bytes, this.persistPassphrase);
    await this.backend.write(this.persistName, bytes);
  }

  /** Serialize the whole store to a single Blob. Optionally encrypted. */
  async export(opts?: ExportOptions): Promise<Blob> {
    let bytes = this.snapshotBytes();
    const passphrase = opts?.encryption?.passphrase;
    if (passphrase) bytes = await encryptSnapshot(bytes, passphrase);
    return new Blob([bytes], { type: 'application/octet-stream' });
  }

  /** Build a store from an exported snapshot Blob. */
  static async import(
    blob: Blob,
    config?: Partial<
      Pick<BrowserVecConfig, 'device' | 'persist' | 'quantBits' | 'quant' | 'ann' | 'chunkRows' | 'fallback'>
    > & {
      encryption?: EncryptionConfig;
    },
  ): Promise<BrowserVec> {
    const raw = await blob.arrayBuffer();
    const snap = deserialize(await decryptIfNeeded(raw, config?.encryption?.passphrase));
    const quantBits = config?.quantBits ?? 0;
    const ctx = await acquireContext(config?.device, config?.fallback ?? 'error');
    const store = new Store(snap.dimension, snap.metric, snap.normalize);
    const makeIndex = indexFactory(
      ctx, snap.dimension, snap.metric, quantBits, config?.quant, config?.ann, config?.chunkRows,
    );
    const index = makeIndex();
    const db = new BrowserVec(ctx, store, index);
    db.quantBits = quantBits;
    db.rerankFactor = config?.quant?.rerankFactor ?? defaultRerankFactor(quantBits);
    db.makeIndex = makeIndex;
    await db.loadSnapshot(snap);
    if (config?.persist) {
      db.backend = selectBackend(config.persist.backend ?? 'auto');
      db.persistName = config.persist.name;
    }
    return db;
  }

  private snapshotBytes(): ArrayBuffer {
    // Compact out tombstoned rows so persisted/exported snapshots hold only live
    // vectors (and a reload reclaims their memory). liveEntries/liveVectors are
    // row-aligned.
    const live = this.store.liveEntries();
    const entries = live.map((e) =>
      e.metadata !== undefined ? { id: e.id, metadata: e.metadata } : { id: e.id },
    );
    return serialize({
      dimension: this.store.dimension,
      metric: this.store.metric,
      normalize: this.store.normalize,
      count: live.length,
      entries,
      vectors: this.store.liveVectors(),
    });
  }

  private async loadSnapshot(snap: Snapshot): Promise<void> {
    if (this.store.count > 0) throw new Error('cannot load into a non-empty store');
    if (snap.dimension !== this.store.dimension) {
      throw new Error(`snapshot dim ${snap.dimension} != store dim ${this.store.dimension}`);
    }
    if (snap.metric !== this.store.metric) {
      throw new Error(`snapshot metric ${snap.metric} != store metric ${this.store.metric}`);
    }
    const dim = snap.dimension;
    for (let row = 0; row < snap.count; row++) {
      const e = snap.entries[row]!;
      const vec = snap.vectors.subarray(row * dim, row * dim + dim);
      // Vectors were stored already prepared (normalized if applicable) — insert as-is.
      this.store.insert(e.id, vec, e.metadata);
    }
    await this.index.append(snap.vectors.subarray(0, snap.count * dim), snap.count);
  }

  stats(): Stats {
    const out: Stats = {
      count: this.store.count,
      dimension: this.store.dimension,
      metric: this.store.metric,
      device: this.ctx ? 'webgpu' : 'wasm',
      quantBits: this.quantBits,
    };
    if (this.store.deletedCount > 0) out.deleted = this.store.deletedCount;
    if (this.lastQueryMs !== undefined) {
      out.lastQueryMs = this.lastQueryMs;
      const gpu = Math.min(this.lastQueryGpuMs ?? 0, this.lastQueryMs);
      out.lastQueryGpuMs = gpu;
      out.lastQueryCpuMs = this.lastQueryMs - gpu;
    }
    if (this.backend) out.persist = this.backend.kind;
    if ('nlist' in this.index) {
      const nlist = (this.index as { nlist: number }).nlist;
      if (nlist > 0) out.nlist = nlist;
    }
    if ('ingestMode' in this.index) {
      const mode = (this.index as { ingestMode: 'worker' | 'main-thread' | 'pending' }).ingestMode;
      if (mode !== 'pending') out.ingest = mode;
    }
    if ('chunkCount' in this.index) {
      const chunks = (this.index as { chunkCount: number }).chunkCount;
      if (chunks > 1) out.chunks = chunks;
    }
    if ('trainMode' in this.index) {
      const mode = (this.index as { trainMode: 'worker' | 'main-thread' | 'pending' }).trainMode;
      if (mode !== 'pending') out.train = mode;
    }
    return out;
  }

  /** Free GPU resources (a no-op device teardown when running the CPU fallback). */
  destroy(): void {
    this.index.destroy();
    this.ctx?.device.destroy();
  }
}

export default BrowserVec;
