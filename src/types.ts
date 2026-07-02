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
   * Enable the IVF approximate index (M4). When set, queries scan only the
   * `nprobe` nearest of `nlist` clusters instead of the whole corpus — trading a
   * little recall for a large latency drop at scale. Omit for exact flat search.
   * Currently fp32 + cosine/dot only.
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

export interface ANNConfig {
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
  /** Persistence backend in use, if configured. */
  persist?: 'opfs' | 'indexeddb';
  /** Quantization bit-width in use (0 = fp32). */
  quantBits: 0 | 1 | 4 | 8;
  /** IVF cluster count, if an ANN index is in use and built. */
  nlist?: number;
  /** Number of GPU buffers the corpus spans. >1 once it overflows one buffer. */
  chunks?: number;
  /**
   * Where quantized ingest ran: 'worker' (rotate+quantize offloaded off the main
   * thread) or 'main-thread' (no Worker available — ran in-thread).
   * Absent for fp32 stores and before the first quantized append.
   */
  ingest?: 'worker' | 'main-thread';
  /**
   * Where the IVF k-means centroid mean-update ran (§NFR-8): 'worker'
   * (offloaded off the main thread) or 'main-thread' (no Worker — ran in-thread).
   * Absent for non-IVF stores and before the index's first build.
   */
  train?: 'worker' | 'main-thread';
}
