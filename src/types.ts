// Public type surface for BrowserVec.

export type Metric = 'cosine' | 'dot' | 'l2';

export type Vector = Float32Array | number[];

export type Metadata = Record<string, string | number | boolean | null>;

export interface BrowserVecConfig {
  /** Embedding dimension. Any positive int; 384/768/1024/1536 are fast-pathed. */
  dimension: number;
  /** Similarity metric. Default 'cosine'. */
  metric?: Metric;
  /** Normalize vectors on insert (recommended for cosine). Default: metric === 'cosine'. */
  normalize?: boolean;
  /** Reuse an existing WebGPU device instead of requesting one. */
  device?: GPUDevice;
  /**
   * Behavior when WebGPU is unavailable (§NFR-7). Default 'error' (throw on
   * create). 'wasm' transparently falls back to an exact CPU scan with identical
   * results at reduced throughput — the library keeps working with no GPU. The
   * CPU fallback currently supports fp32 flat only; `quantBits`/`ann` still need
   * WebGPU and throw under fallback.
   */
  fallback?: 'wasm' | 'error';
  /** Enable persistence to OPFS/IndexedDB. */
  persist?: PersistConfig;
  /**
   * On-device text embedder (M5). When set, `addText`/`addTexts`/`queryText`
   * become available. Its `dimension` must equal `dimension`.
   */
  embedder?: Embedder;
  /**
   * Quantization bit-width. 0 = fp32 flat (default). 8 = TurboQuant int8 (~4×
   * less memory). 4 = sub-byte int4 (~8× less). 1 = binary (~32× less — sign bits
   * only, coarsest; leans hardest on re-rank). Rotated + asymmetric kernel + exact
   * fp32 re-rank. Requires metric cosine/dot. Binary (1) is flat-only for now
   * (no IVF combo yet).
   */
  quantBits?: 0 | 1 | 4 | 8;
  /** TurboQuant tuning (only used when quantBits > 0). */
  quant?: QuantConfig;
  /**
   * Enable an approximate (ANN) index. Omit for exact flat search.
   * - `{ type: 'ivf' }` (or omitted `type`, M4): k-means clusters, queries scan
   *   only the `nprobe` nearest of `nlist` — GPU-only, fp32 + cosine/dot.
   * - `{ type: 'hnsw' }` (M7): CPU graph index searched by greedy descent —
   *   sub-linear queries with incremental inserts, works with or without WebGPU
   *   and on any metric. Currently fp32 only.
   */
  ann?: ANNConfig;
  /**
   * Force the corpus to split into chunks of at most this many rows.
   * Normally chunking triggers automatically once a single GPU buffer would
   * exceed the device limit; this override exists mainly to exercise the chunked
   * path cheaply in tests/demos. Applies to flat fp32 (`quantBits: 0`), flat
   * quantized (`quantBits: 4 | 8`), the IVF×quant combo (`ann` + quantBits), and
   * fp32 IVF (`ann` alone). Ignored if larger than one buffer holds.
   */
  chunkRows?: number;
}

export type ANNConfig = IVFConfig | HNSWConfig;

export interface IVFConfig {
  /** Index family. IVF is the default when `type` is omitted. */
  type?: 'ivf';
  /** Number of clusters. Default ≈ sqrt(count), clamped to [16, 4096]. */
  nlist?: number;
  /** Clusters scanned per query. Default ≈ 5% of nlist. Higher = better recall, slower. */
  nprobe?: number;
  /** Reservoir sample size used to train k-means. Default 50_000. */
  sampleSize?: number;
  /** Lloyd iterations during build. Default 12. */
  iters?: number;
  /** Seed for sampling + k-means, for reproducible builds. */
  seed?: number;
}

export interface HNSWConfig {
  /** Index family: HNSW graph (M7). */
  type: 'hnsw';
  /** Graph out-degree per layer (layer 0 keeps 2·M). Higher = better recall, more memory + slower build. Default 16. */
  M?: number;
  /** Candidate-list width while building. Higher = better graph quality, slower ingest. Default 200. */
  efConstruction?: number;
  /** Default candidate-list width at query time (clamped to ≥ k). Higher = better recall, slower. Default 64. */
  efSearch?: number;
  /** Seed for the level RNG, for reproducible builds. */
  seed?: number;
  /**
   * Where queries run (M7b). 'cpu' (default): graph walk in the Worker/in-thread.
   * 'gpu': single-dispatch beam-search kernel — the whole search runs inside one
   * compute dispatch, one workgroup per query, so `queryBatch` searches every
   * query concurrently. Requires WebGPU, M ≤ 32, efSearch ≤ 256, and the corpus
   * within one storage buffer (falls back to 'cpu' past that; see
   * stats().graphSearch). Single queries pay fixed dispatch+readback latency —
   * batches and large corpora are where 'gpu' wins. Ignored on the CPU fallback.
   */
  search?: 'cpu' | 'gpu';
}

export interface QuantConfig {
  /** Rotation seed. Fixed default so reloads re-quantize identically. */
  seed?: number;
  /** Randomized-Hadamard rounds (more = better mixing, slower build). Default 2. */
  rounds?: number;
  /**
   * Exact fp32 re-rank: fetch k·factor approximate candidates from the GPU, then
   * re-score them exactly on the CPU. Default 4 for int8, 16 for int4, 32 for
   * binary (coarser scores need a wider pool on near-duplicate data). Set 1 to
   * disable re-rank.
   */
  rerankFactor?: number;
}

export interface PersistConfig {
  /** Snapshot name (file/key). Required for save()/auto-load. */
  name: string;
  /** Storage backend. Default 'auto' (OPFS, else IndexedDB). */
  backend?: 'opfs' | 'indexeddb' | 'auto';
  /**
   * Load an existing snapshot on create() if one exists for `name`.
   * Default true. A dimension/metric mismatch throws.
   */
  autoLoad?: boolean;
  /**
   * Encrypt persisted snapshots at rest. When set, `save()` writes an
   * AES-256-GCM envelope (passphrase stretched via PBKDF2-SHA256) and auto-load
   * decrypts it — a wrong passphrase or tampered blob throws rather than
   * returning garbage. Requires WebCrypto (`crypto.subtle`).
   */
  encryption?: EncryptionConfig;
}

export interface EncryptionConfig {
  /** Passphrase used to derive the AES-GCM key. Non-empty. */
  passphrase: string;
}

export interface ExportOptions {
  /** Encrypt the exported blob with this passphrase. */
  encryption?: EncryptionConfig;
}

export interface VectorRecord {
  id: string;
  vector: Vector;
  metadata?: Metadata;
}

/**
 * Turns text into vectors on-device. Bundled implementations: `hashingEmbedder`
 * (zero-dependency, offline, non-semantic — good for tests/demos) and
 * `transformersEmbedder` (real semantic model via @xenova/transformers, lazy-loaded).
 * Bring your own by implementing this interface.
 */
export interface Embedder {
  /** Output vector length — must match the store's `dimension`. */
  readonly dimension: number;
  /** Embed a batch of texts; returns one vector per input, in order. */
  embed(texts: string[]): Promise<Float32Array[]>;
}

export interface TextRecord {
  id: string;
  text: string;
  metadata?: Metadata;
}

export interface QueryResult {
  id: string;
  /** Metric-dependent: higher = closer for cosine/dot, lower = closer for l2. */
  score: number;
  metadata?: Metadata;
}

export interface QueryOptions {
  /** Number of neighbors to return. Default 10. */
  k?: number;
  /**
   * Override exact re-rank for this query (quantized stores only). Defaults to
   * the store's configured behavior. Has no effect on fp32 stores.
   */
  rerank?: boolean;
  /**
   * Clusters to scan for this query (IVF stores only). Overrides the configured
   * default — higher = better recall, slower. No effect on flat stores.
   */
  nprobe?: number;
  /**
   * Candidate-list width for this query (HNSW stores only). Overrides the
   * configured `efSearch` — higher = better recall, slower. Clamped to ≥ k.
   */
  efSearch?: number;
}

export interface SupportInfo {
  webgpu: boolean;
  opfs: boolean;
  wasm: boolean;
}

export interface Stats {
  /** Live (non-deleted) vector count. */
  count: number;
  /** Tombstoned rows awaiting compaction on the next save/export (0 if none). */
  deleted?: number;
  dimension: number;
  metric: Metric;
  /** 'webgpu' on the GPU path, 'wasm' when running the CPU fallback (§NFR-7). */
  device: 'webgpu' | 'wasm';
  /** Wall-clock of the most recent query() in ms. */
  lastQueryMs?: number;
  /**
   * Of lastQueryMs: time spent awaiting GPU work — kernel execution + readback
   * transfer + queue scheduling (see engine/profile.ts). 0 on the CPU fallback.
   */
  lastQueryGpuMs?: number;
  /** Of lastQueryMs: JS-side time (prep, candidate gather, exact re-rank, top-k merge). */
  lastQueryCpuMs?: number;
  /** Persistence backend in use, if configured. */
  persist?: 'opfs' | 'indexeddb';
  /** Quantization bit-width in use (0 = fp32). */
  quantBits: 0 | 1 | 4 | 8;
  /** IVF cluster count, if an IVF index is in use and built. */
  nlist?: number;
  /** HNSW top graph layer, if an HNSW index is in use and non-empty. */
  maxLevel?: number;
  /** HNSW query engine: 'gpu' (beam-search kernel) or 'cpu' (graph walk). */
  graphSearch?: 'gpu' | 'cpu';
  /** Number of GPU buffers the corpus spans. >1 once it overflows one buffer. */
  chunks?: number;
  /**
   * Where quantized ingest ran: 'worker' (rotate+quantize offloaded off the main
   * thread) or 'main-thread' (no Worker available — ran in-thread).
   * Absent for fp32 stores and before the first quantized append.
   */
  ingest?: 'worker' | 'main-thread';
  /**
   * Where the ANN index build ran (§NFR-8): 'worker' (offloaded off the main
   * thread) or 'main-thread' (no Worker — ran in-thread). For IVF this is the
   * k-means centroid mean-update; for HNSW it is the graph construction.
   * Absent for flat stores and before the index's first build.
   */
  train?: 'worker' | 'main-thread';
}
