# Requirements Document — BrowserVec: In-Browser WebGPU Vector Store

**Status:** Draft v0.4
**Date:** 2026-06-30
**Owner:** skss.vfx@gmail.com

---

## 1. Overview

BrowserVec is a client-side vector database plugin that stores, indexes, and
retrieves high-dimensional embeddings entirely within the browser. Similarity
search is executed on the GPU via **custom WebGPU compute kernels** to maximize
throughput and minimize query latency. The library targets in-browser session
memory, offline retrieval-augmented generation (RAG), and privacy-preserving
semantic search where no data leaves the device.

Because the primary target is **large local corpora (> 1M vectors)** across
**all major browsers including mobile**, the design treats **quantization and an
approximate (ANN) index as first-class, not optional** — see §6 (TurboQuant).

### 1.1 Problem Statement
Existing JS vector libraries run similarity math on the CPU (single-threaded or
WASM-SIMD), which caps throughput on large corpora and drains battery. Server-
side vector DBs require network round-trips, leak user data, and don't work
offline. BrowserVec fills the gap: a GPU-accelerated, zero-network, persistent
vector store usable from any web app, scaling to millions of vectors on consumer
hardware including phones.

### 1.2 Goals
- Sub-10 ms top-k query over 1M vectors (768-dim) on mid-tier laptop GPU using ANN + quantization.
- Fully offline operation after initial load (no network at query time).
- Persistent across sessions via OPFS / IndexedDB.
- Drop-in JS/TS API; framework-agnostic; runs in a Web Worker.
- Configurable embedding dimension with performance-tuned kernels for common sizes.

### 1.3 Non-Goals
- Distributed / multi-device sync (future).
- Training or fine-tuning embedding models.
- Server-side execution (Node) as a primary target.

---

## 2. Target Users & Use Cases

| User | Use Case |
|------|----------|
| Web app developer | Add semantic search over local documents without a backend |
| AI feature builder | In-browser RAG: retrieve context chunks for an on-device or API LLM |
| Privacy-focused app | Keep all embeddings on-device; never transmit user data |
| Offline / PWA | Retrieval works with no connectivity (airplane, field work) |
| Chat/agent UX | Session memory: store conversation turns, recall relevant history |
| Large-corpus search | Full document libraries / knowledge bases (> 1M chunks) on-device |

---

## 3. Configurable Dimension Strategy

**Decision:** Dimension is fully configurable at store creation; kernels are
optimized for common sizes but correct for any dimension, with an explicit
performance contract.

- **FR-DIM-1** Accept any positive integer dimension at init.
- **FR-DIM-2** Ship **specialized (fast-path) kernels** with compile-time tiling
  for the common sizes **384, 768, 1024, 1536** (768 is the default).
- **FR-DIM-3** Provide a **generic kernel** for arbitrary dimensions; it is
  correct but may be slower than the specialized paths.
- **FR-DIM-4** Kernels are generated/specialized via WGSL constants (workgroup
  size, tile width, unroll factor) selected per dimension and per device limits.
- **NFR-DIM-1** Performance contract: specialized sizes meet the §5 latency
  targets; generic dimensions target within ~1.5–2× of the nearest specialized
  size. Document the measured cost per dimension.
- **NFR-DIM-2** Higher dimensions increase memory and bandwidth linearly;
  surface guidance (and quantization recommendations) when a chosen dimension ×
  corpus size approaches device buffer limits.

---

## 4. Functional Requirements

### 4.1 Core API
- **FR-1** Initialize a store with configurable dimension, metric, index type, and capacity.
- **FR-2** `add(id, vector, metadata?)` — insert single vector with optional JSON metadata.
- **FR-3** `addBatch(records[])` — bulk insert (primary ingestion path).
- **FR-4** `query(vector, k, filter?)` — return top-k nearest neighbors with scores.
- **FR-5** `delete(id)` / `update(id, vector)` — mutate existing entries.
- **FR-6** `get(id)` — fetch vector + metadata by id.
- **FR-7** Metadata filtering: pre/post-filter results by metadata predicates.

### 4.2 Similarity Metrics
- **FR-8** Support cosine similarity, dot product, and Euclidean (L2) distance.
- **FR-9** Optional vector normalization on insert (cosine via dot product).

### 4.3 Indexing (scale-first)
- **FR-10** Brute-force (flat) GPU scan as the exact baseline / ground-truth and for small stores.
- **FR-11** **Approximate (ANN) index required** for the >1M target — IVF (inverted file / cluster-based) as the primary approach; HNSW evaluated as an alternative.
- **FR-12** **Quantization is a core feature**, not optional — see §6.
- **FR-13** Index build runs incrementally and off the main thread; support adding to an existing index without full rebuild where feasible.

### 4.4 Persistence
- **FR-14** Persist vectors + metadata + index to OPFS (preferred) or IndexedDB fallback.
- **FR-15** Lazy / streamed load of quantized vectors and index into GPU buffers on startup.
- **FR-16** Export / import the store as a single binary blob (versioned format).

### 4.5 WebGPU Kernels
- **FR-17** Custom WGSL compute shaders for batched distance computation over quantized data.
- **FR-18** GPU-side top-k selection (partial sort / reduction) to avoid CPU readback of full score arrays.
- **FR-19** Buffer pooling and reuse to avoid per-query allocation.
- **FR-20** Tunable workgroup size and tiling for device-specific optimization (incl. mobile).
- **FR-21** Corpus chunking across multiple dispatches to respect per-buffer size limits.

### 4.6 Embeddings (bundled + pluggable)
- **FR-22** Define a stable `Embedder` interface: `embed(texts[]) -> vectors[]`, exposing dimension and metric.
- **FR-23** **Bundle a default on-device embedder** (e.g. a small transformers.js model running on WebGPU/WASM) for true end-to-end offline use.
- **FR-24** Allow users to **bring their own embedder** (remote API, custom model, or precomputed vectors) via the same interface.
- **FR-25** Default embedder is lazy-loaded / code-split so apps that supply their own vectors pay no bundle cost.
- **FR-26** Validate that embedder output dimension matches store dimension; clear error on mismatch.

---

## 5. Non-Functional Requirements

### 5.1 Performance
- **NFR-1** Query latency: < 10 ms for top-10 over **1M × 768-dim** via ANN + quantization (target laptop GPU); < 5 ms over 100k.
- **NFR-2** Ingestion: > 50k vectors/sec batch insert (excluding embedding generation).
- **NFR-3** Cold start (load 1M quantized vectors + index from OPFS to GPU): < 1.5 s.
- **NFR-4** Memory: 1M × 768-dim fits via quantization (~750 MB int8, ~96 MB at ~1 bit) with chunking across GPU buffers; fp32 (~3 GB) only for small stores.
- **NFR-5** Recall: ANN recall@10 ≥ 0.95 vs. exact flat search at the default quantization setting; tunable accuracy/speed knob.

### 5.2 Compatibility
- **NFR-6** Support **Chrome/Edge, Safari, and Firefox** with WebGPU, plus **mobile browsers** (iOS/Android).
- **NFR-7** Graceful **CPU/WASM fallback** when WebGPU is unavailable or device-limited — identical results, reduced throughput. This is a hard requirement given the broad/mobile target.
- **NFR-8** Runs in Web Workers (off main thread) — required for non-blocking UI.
- **NFR-9** Detect and adapt to **mobile GPU limits** (smaller `maxStorageBufferBindingSize`, `maxBufferSize`, workgroup sizes); auto-tune tiling and chunk counts per device.

### 5.3 Reliability & Limits
- **NFR-10** Respect `maxStorageBufferBindingSize` / `maxBufferSize` / `maxComputeWorkgroup*`; auto-chunk large corpora across dispatches.
- **NFR-11** Handle GPU device loss (`device.lost`) with recovery / re-upload from OPFS source of truth.

### 5.4 Developer Experience
- **NFR-12** TypeScript types, ESM module, < 50 KB gzipped core (excl. WASM fallback and bundled embedder, which are code-split).
- **NFR-13** Zero required build config; usable via `<script type=module>` or npm.
- **NFR-14** Async, promise-based API; no blocking calls.

### 5.5 Privacy & Security
- **NFR-15** No network calls at runtime; all data local (default embedder runs on-device).
- **NFR-16** Optional encryption-at-rest for persisted data.

---

## 6. Quantization — TurboQuant Integration

BrowserVec adopts **TurboQuant** (Zandieh, Daliri, Hadian, Mirrokni —
*"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*,
arXiv:2504.19874) as its primary quantization scheme.

### 6.1 Why it fits this project
- **Data-oblivious** — uses a random rotation + per-coordinate scalar
  quantizers, requiring **no training data or fitted codebook**. Critical in a
  browser: we cannot ship/recompute a calibrated codebook per corpus, and new
  vectors can be quantized online as they arrive.
- **GPU-friendly** — per-coordinate **scalar** quantization dequantizes with
  branchless arithmetic, unlike Product Quantization which needs per-subspace
  lookup tables and gather/scatter that stall WebGPU compute.
- **Near-optimal distortion across all bit-widths** — supports a single
  accuracy/memory knob from ~1 bit/dim up to 8 bit/dim, matching NFR-4/NFR-5.
- **Handles both MSE and inner-product distortion** — covers our cosine/dot and
  L2 metrics; the paper reports strong nearest-neighbor results vs. PQ and
  quality-neutral results at low bit-widths.

### 6.2 Requirements
- **FR-Q1** Implement TurboQuant: (a) random rotation of vectors, (b) optimal
  per-coordinate scalar quantization, (c) two-stage inner-product estimation
  (MSE quant + residual 1-bit Quantized-JL) for dot/cosine queries.
- **FR-Q2** Configurable bit-width per store (e.g. 1, 2, 4, 8 bits/dim) exposing
  the accuracy↔memory↔speed trade-off.
- **FR-Q3** Random rotation implemented as a fast transform (e.g. structured /
  Hadamard-style) so it is cheap on GPU and WASM and reproducible from a stored seed.
- **FR-Q4** WGSL distance kernels operate **directly on quantized codes** (dequant in-shader), keeping memory bandwidth — the GPU bottleneck — minimal.
- **FR-Q5** Asymmetric distance: full-precision (or higher-bit) query vs.
  quantized corpus for better recall.
- **FR-Q6** Optional exact **re-ranking** stage — refine top-N quantized
  candidates against fp32/higher-precision vectors to recover recall.
- **NFR-Q1** Validate measured distortion / recall against the paper's reported
  rates and against an fp32 brute-force reference in CI.

### 6.3 Licensing / IP assessment (researched 2026-06-30)

**Findings**
- **Paper license:** arXiv:2504.19874 is published under **CC BY 4.0**. We may
  freely read, reproduce, and build on the *described method* with attribution.
  ⚠️ Important: CC BY covers the *paper text/figures*, **not patent rights** — a
  permissive paper license does not grant a patent license.
- **Venue / authorship:** Accepted to **ICLR 2026**. Authors (Zandieh, Daliri,
  Hadian, **Mirrokni**) are associated with **Google Research** (Mirrokni leads
  algorithms research there). This raises a **non-trivial patent-risk flag** —
  large labs routinely file patents on quantization methods even when the paper
  is open. No patent was confirmed in our search, but absence of evidence ≠
  clearance.
- **Reference code:** The authors published **no official implementation**.
  Several community implementations exist (Python, Rust, etc.), e.g.
  `yashkc2025/turboquant`, `OmarHory/turboquant`, `jonpojonpo/turbo-vec` (Rust).
  Licenses vary per repo and must be checked individually before reuse — we
  cannot assume MIT/Apache across them.

**Conclusion / actions**
- ✅ **Clean-room reimplementation from the paper is the safe path** — implement
  our own WGSL/TS version from the algorithm description; do not copy GPL/unknown
  community code. Attribute the paper (CC BY).
- ⏳ **Patent search before commercial GA** — run a Google Patents / USPTO search
  for the authors + "quantization" / "quantized JL transform"; if BrowserVec ships
  commercially, have counsel confirm no blocking claim. *Blocking item before a
  GA release; not blocking for M3 prototyping/research use.*
- 🔁 **Design hedge** — keep quantization behind the `quantBits` interface so
  TurboQuant can be swapped for an unencumbered scheme (plain scalar/int8, or
  RaBitQ-style) if an IP problem surfaces. Low switching cost.

---

## 7. Architecture (High Level)

```
┌─────────────────────────────────────────────────────────┐
│                   Public JS/TS API                       │
│      add / addBatch / query / delete / persist           │
│      + optional Embedder (bundled or user-supplied)      │
└───────────────┬─────────────────────────────────────────┘
                │  (runs in Web Worker)
        ┌───────┴────────┐
        │  Store Manager  │  id map, metadata, filtering, dim check
        └───┬─────┬───────┘
            │     │
   ┌────────┘     └──────────────┐
┌──┴──────────────┐     ┌─────────┴──────────┐
│  WebGPU Engine   │     │ Persistence Layer   │
│  WGSL kernels    │     │  OPFS / IndexedDB    │
│  (dim-specialized│     │  versioned binary    │
│   + generic)     │     │  serializer          │
│  TurboQuant      │     └─────────────────────┘
│  dequant in-shader│
│  ANN (IVF) + top-k│
│  buffer pools     │
└──────┬───────────┘
       │  fallback (mobile / no-WebGPU)
┌──────┴────────┐     ┌──────────────────────┐
│  WASM/CPU path │     │  Embedder (code-split)│
└───────────────┘     │  default on-device    │
                      │  model + BYO adapter   │
                      └───────────────────────┘
```

### 7.1 Query Pipeline (ANN + quantized)
1. Embed query (if text) → fp32 query vector; rotate (TurboQuant transform).
2. IVF coarse stage: pick nearest cluster centroids (GPU or CPU).
3. Distance kernel over **quantized** candidates in selected clusters; dequant in-shader (asymmetric query).
4. GPU top-k reduction → small candidate list (ids + approx scores).
5. Optional exact re-rank of top-N against higher-precision vectors.
6. Read back only final k results — minimal transfer.

---

## 8. Key Technical Risks

| Risk | Mitigation |
|------|------------|
| WebGPU buffer size limits (esp. mobile) | Chunk corpus, multi-dispatch, per-device auto-tuning |
| GPU top-k non-trivial vs CPU sort | Start GPU-distance + CPU top-k; optimize to GPU reduction |
| Browser/driver/mobile variance | Feature-detect limits; tunable workgroups; CI across devices; WASM fallback |
| ANN recall vs. speed | Tunable nprobe/bit-width + exact re-rank stage; CI recall gates |
| TurboQuant impl correctness | Validate distortion/recall vs. paper + fp32 reference |
| TurboQuant licensing/patent | Clear IP before M3 (§6.3) |
| Device loss / context reset | OPFS is source of truth; re-upload on recovery |
| Bundle size with embedder | Code-split embedder + WASM fallback; core stays < 50 KB |

---

## 9. Milestones

- **M1 — MVP:** Flat GPU brute-force, in-memory, cosine/dot, configurable dim (768 default + specialized sizes), Worker API.
- **M2 — Persistence:** OPFS storage, versioned export/import, streamed cold-start load.
- **M3 — Quantization (TurboQuant):** Scalar quant + in-shader dequant, configurable bit-width, asymmetric distance + re-rank. *(IP cleared first.)*
- **M4 — ANN at scale:** IVF index, 1M-vector support, incremental build, metadata filtering, GPU top-k reduction.
- **M5 — Embeddings:** Pluggable `Embedder` interface + bundled code-split on-device embedder.
- **M6 — Cross-browser & mobile hardening:** Safari/Firefox/mobile tuning, WASM fallback parity, device-loss recovery, encryption, docs.

---

## 10. Acceptance Criteria

- Top-10 query over 1M × 768-dim returns in < 10 ms (p50) on reference GPU with recall@10 ≥ 0.95 vs. exact.
- Configurable dimension verified for 384/768/1024/1536 (fast) and an arbitrary size (generic, within perf contract).
- Quantized results validated against fp32 brute-force within tolerance / recall gate.
- Store survives reload and restores (1M, quantized) in < 1.5 s.
- Runs in a Web Worker without blocking the main thread.
- Works in Chrome/Edge, Safari, Firefox, and at least one mobile browser; falls back to WASM with identical results where WebGPU is absent.
- End-to-end offline path works using the bundled embedder (no network).

---

## 11. Open Questions

1. IVF vs. HNSW as the primary ANN structure for GPU + browser memory constraints?
2. Default TurboQuant bit-width for the shipped preset (balance recall vs. memory on mobile)?
3. Which model to bundle as the default on-device embedder, and at what dimension?
4. Encryption-at-rest: required for v1 or deferred to post-M6?
5. ~~TurboQuant IP/licensing clearance~~ — **resolved (§6.3):** CC BY paper, clean-room reimplement; patent search before commercial GA, not before M3. Quantization kept swappable as a hedge.
6. Mobile memory ceiling — do we cap corpus size or degrade gracefully (spill to OPFS, partial index)?

---

## 12. Tech Stack & Implementation Plan

### 12.1 Technology Choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Language | **TypeScript**, compiled to ESM | Types for a public API; tree-shakeable modules |
| GPU | **WebGPU** (raw API, no wrapper) | Max control over buffers/pipelines; wrappers add overhead and hide limits |
| Shaders | **WGSL**, generated per-dimension via template + `override` constants | Specialized fast paths (§3) without hand-writing every variant |
| Threading | **Web Worker** + `Comlink`-style RPC (or hand-rolled, to stay dependency-light) | Keep main thread free; all heavy work off-thread |
| Persistence | **OPFS** (`navigator.storage.getDirectory`) primary, **IndexedDB** fallback | OPFS gives fast, large, synchronous-in-worker file access; IDB for Safari gaps |
| CPU fallback | **WASM (Rust → wasm-pack)** with SIMD, or hand-written wasm | Parity path for no-WebGPU / mobile; SIMD for acceptable speed |
| Embedder (bundled) | **transformers.js** (ONNX Runtime Web, WebGPU/WASM backends), code-split | Mature on-device embedding; reuses the same WebGPU device |
| Build | **Vite** (lib mode) + **Vitest** | Fast, ESM-native, good worker/wasm support |
| Testing GPU | **Playwright** across Chrome/Edge/Firefox + headless GPU; real-device lab for Safari/mobile | WebGPU behavior is driver-specific; needs real browsers |
| Bench/recall | Custom harness vs. **fp32 brute-force ground truth** | Gate recall@k and latency in CI |

### 12.2 Module Layout

```
browservec/
├─ src/
│  ├─ index.ts                 # public API surface (§13), main-thread proxy
│  ├─ worker/
│  │  ├─ worker.ts             # RPC entry; owns GPU device + store state
│  │  └─ rpc.ts                # message protocol, transferables
│  ├─ engine/
│  │  ├─ device.ts             # adapter/device init, limits, device-loss recovery
│  │  ├─ buffers.ts            # buffer pool, chunking against maxBufferSize
│  │  ├─ pipelines.ts          # pipeline cache keyed by (dim, metric, bits)
│  │  └─ wgsl/
│  │     ├─ distance.wgsl.ts   # templated distance kernel (dequant in-shader)
│  │     ├─ topk.wgsl.ts       # GPU top-k reduction
│  │     └─ rotate.wgsl.ts     # TurboQuant structured rotation (Hadamard-style)
│  ├─ quant/
│  │  ├─ turboquant.ts         # rotation + scalar quant + JL residual (§6)
│  │  └─ codec.ts              # pack/unpack quantized codes
│  ├─ index/
│  │  ├─ flat.ts               # exact brute-force (baseline + ground truth)
│  │  └─ ivf.ts                # coarse quantizer / inverted lists, nprobe
│  ├─ store/
│  │  ├─ store.ts              # id map, metadata, filtering, dim validation
│  │  └─ filter.ts             # metadata predicate evaluation
│  ├─ persist/
│  │  ├─ opfs.ts               # OPFS read/write, streamed load
│  │  ├─ indexeddb.ts          # fallback
│  │  └─ format.ts             # versioned binary (header, dim, bits, seed, lists)
│  ├─ embed/
│  │  ├─ types.ts              # Embedder interface
│  │  └─ transformers.ts       # bundled default (code-split, lazy)
│  └─ fallback/
│     └─ wasm.ts               # WASM/CPU distance + top-k parity path
├─ tests/                      # unit + recall/latency benches
└─ bench/                      # device matrix harness
```

### 12.3 Phased Plan (maps to §9 milestones)

1. **M1 — MVP / flat GPU**
   - Device init + limits probe; buffer pool with chunking.
   - Templated WGSL distance kernel for 384/768/1024/1536 + generic; CPU top-k first.
   - Worker RPC; in-memory store; `add/addBatch/query`.
   - Recall harness vs. naive JS reference. *Exit:* exact top-k correct; <5 ms over 100k×768.

2. **M2 — Persistence**
   - OPFS writer/reader + versioned format; IndexedDB fallback.
   - Streamed cold-start load into GPU buffers. *Exit:* survive reload; documented load times.

3. **M3 — TurboQuant** *(IP cleared first, §6.3)*
   - Structured rotation kernel; scalar quant codec (1/2/4/8 bit); in-shader dequant.
   - Asymmetric query distance + optional fp32 re-rank. *Exit:* recall@10 ≥ 0.95 at chosen preset vs. fp32; memory within NFR-4.

4. **M4 — ANN at scale**
   - IVF coarse quantizer (k-means on device/worker), inverted lists, `nprobe` knob.
   - GPU top-k reduction; incremental insert; metadata filtering. *Exit:* <10 ms top-10 over 1M×768, recall gate held.

5. **M5 — Embeddings**
   - `Embedder` interface; code-split transformers.js default; dim validation. *Exit:* offline E2E text→search with no network.

6. **M6 — Cross-browser / mobile hardening**
   - Per-device auto-tuning; Safari/Firefox/mobile passes; WASM fallback parity; device-loss recovery; encryption; docs. *Exit:* acceptance criteria (§10) across the device matrix.

### 12.4 Cross-Cutting Concerns
- **Pipeline cache** keyed by `(dim, metric, bits, deviceLimits)` — compile shaders once.
- **Auto-tuning**: micro-bench workgroup/tile candidates on first run per device; persist the winner.
- **Backpressure**: cap concurrent dispatches; queue large `addBatch` to avoid OOM.
- **Determinism**: store the rotation seed so quantization is reproducible across loads.

---

## 13. TypeScript API Surface (Draft)

> Illustrative public types for v1. The main-thread `index.ts` proxies these to
> the Worker; all methods are async.

```ts
// ---- Core types -------------------------------------------------------------

export type Metric = 'cosine' | 'dot' | 'l2';
export type IndexType = 'flat' | 'ivf';
export type QuantBits = 0 | 1 | 2 | 4 | 8; // 0 = no quantization (fp32)

export interface BrowserVecConfig {
  dimension: number;                 // any positive int; 384/768/1024/1536 fast-pathed
  metric?: Metric;                   // default 'cosine'
  index?: IndexType;                 // default 'flat'; 'ivf' for large corpora
  quantBits?: QuantBits;             // default 0; e.g. 4 for >1M
  normalize?: boolean;               // normalize on insert (cosine)
  persist?: {
    name: string;                    // OPFS/IDB store name
    backend?: 'opfs' | 'indexeddb' | 'auto'; // default 'auto'
    encryptionKey?: CryptoKey;       // optional encryption-at-rest
  };
  ivf?: {
    nlist?: number;                  // number of clusters (build)
    nprobe?: number;                 // clusters scanned per query (search)
  };
  embedder?: Embedder | 'default';   // 'default' lazy-loads bundled model
  device?: GPUDevice;                // reuse an existing WebGPU device
  fallback?: 'wasm' | 'error';       // behavior when WebGPU unavailable
}

export type Vector = Float32Array | number[];
export type Metadata = Record<string, string | number | boolean | null>;

export interface VectorRecord {
  id: string;
  vector?: Vector;                   // optional if `text` + embedder provided
  text?: string;                     // embedded via configured Embedder
  metadata?: Metadata;
}

export interface QueryResult {
  id: string;
  score: number;                     // metric-dependent (higher = closer for cosine/dot)
  metadata?: Metadata;
  vector?: Vector;                   // included only if requested
}

export interface QueryOptions {
  k?: number;                        // default 10
  filter?: MetadataFilter;           // metadata predicate
  nprobe?: number;                   // override IVF nprobe for this query
  rerank?: boolean;                  // exact re-rank of quantized candidates
  includeVectors?: boolean;
}

// Mongo-ish predicate; AND of fields, with simple operators.
export type MetadataFilter = {
  [field: string]:
    | string | number | boolean
    | { $eq?: unknown; $ne?: unknown; $in?: unknown[];
        $gt?: number; $gte?: number; $lt?: number; $lte?: number };
};

// ---- Embedder interface (bundled + pluggable, §4.6) -------------------------

export interface Embedder {
  readonly dimension: number;
  readonly metric?: Metric;
  embed(texts: string[], opts?: { signal?: AbortSignal }): Promise<Float32Array[]>;
  ready?(): Promise<void>;           // warm up / load model weights
  dispose?(): void;
}

// ---- Main class -------------------------------------------------------------

export declare class BrowserVec {
  static isSupported(): { webgpu: boolean; opfs: boolean; wasm: boolean };
  static create(config: BrowserVecConfig): Promise<BrowserVec>;

  readonly dimension: number;
  readonly count: number;            // number of stored vectors
  readonly index: IndexType;

  add(record: VectorRecord): Promise<void>;
  addBatch(records: VectorRecord[], opts?: { signal?: AbortSignal;
    onProgress?: (done: number, total: number) => void }): Promise<void>;

  query(query: Vector | string, opts?: QueryOptions): Promise<QueryResult[]>;
  queryBatch(queries: Array<Vector | string>, opts?: QueryOptions): Promise<QueryResult[][]>;

  get(id: string, opts?: { includeVector?: boolean }): Promise<VectorRecord | null>;
  update(id: string, vector: Vector, metadata?: Metadata): Promise<void>;
  delete(id: string): Promise<void>;

  buildIndex(opts?: { signal?: AbortSignal }): Promise<void>; // (re)build IVF
  optimize(): Promise<void>;          // compaction / auto-tune persist

  save(): Promise<void>;              // flush to OPFS/IDB
  export(): Promise<Blob>;            // single-blob snapshot
  static import(blob: Blob, config?: Partial<BrowserVecConfig>): Promise<BrowserVec>;

  stats(): Promise<{
    count: number; dimension: number; bytes: number;
    quantBits: QuantBits; index: IndexType;
    device: 'webgpu' | 'wasm';
    lastQueryMs?: number; estRecall?: number;
  }>;

  dispose(): Promise<void>;          // free GPU buffers, terminate worker
}
```

### 13.1 Usage Sketches

```ts
// Bring-your-own vectors, small in-memory store
const db = await BrowserVec.create({ dimension: 768, metric: 'cosine' });
await db.addBatch(records);                       // records have .vector
const hits = await db.query(queryVector, { k: 5 });

// Large persisted, quantized, IVF + bundled embedder (offline RAG)
const rag = await BrowserVec.create({
  dimension: 768, index: 'ivf', quantBits: 4,
  ivf: { nlist: 4096, nprobe: 32 },
  persist: { name: 'docs' },
  embedder: 'default',                            // lazy-loads on-device model
  fallback: 'wasm',
});
await rag.addBatch(chunks.map(c => ({ id: c.id, text: c.text, metadata: c.meta })));
await rag.buildIndex();
await rag.save();
const ctx = await rag.query('how do I reset my password?',
  { k: 8, rerank: true, filter: { lang: 'en' } });
```

---

## 14. How Custom WebGPU Kernels Deliver the Speed

The performance targets in §5 are not reachable with off-the-shelf JS/WASM or a
generic GPU library. They come from hand-written WGSL compute kernels tuned to
the exact shape of this problem. The core insight: **vector search is
memory-bandwidth bound, not compute bound** — for each query we stream millions
of vectors through simple multiply-adds. Every optimization below exists to move
*fewer bytes* and *keep the GPU's ALUs fed*.

### 14.1 The baseline we're beating
- **CPU/JS:** one core, no SIMD width to speak of — ~1M×768 dot products is tens
  of ms to >100 ms, and it blocks the main thread.
- **WASM+SIMD:** ~4× over scalar JS, still single-digit-to-tens of ms, still CPU
  memory bandwidth bound.
- **Generic GPU libs (e.g. matmul via a tensor lib):** correct, but they
  round-trip full fp32 buffers, allocate per call, read back the entire score
  array to the CPU, and can't fuse quantization or top-k. The PCIe readback alone
  can dominate.

Custom kernels win by attacking each of those costs directly:

### 14.2 The seven levers

1. **Quantized data + in-shader dequant (biggest win).**
   The kernel reads **int8/4-bit/1-bit codes**, not fp32. At 4 bits that is
   **~8× less memory traffic** than fp32 for the same corpus. Since search is
   bandwidth bound, throughput scales almost linearly with bytes-moved. Dequant
   happens in-register with a couple of ALU ops — essentially free relative to
   the bandwidth saved. *This is why TurboQuant (§6) is core, not optional.*

2. **Kernel fusion — no intermediate buffers.**
   Distance compute → partial reduction → top-k candidate tracking happen in
   **one dispatch**, keeping intermediates in registers/workgroup shared memory.
   A naive pipeline writes a full N-length score array to VRAM and reads it back;
   fusion never materializes it. Fewer round-trips, less VRAM pressure.

3. **GPU-side top-k — minimal readback.**
   Each workgroup keeps its k best in shared memory via a reduction; only the
   final `k` (ids + scores) cross back to the CPU. For 1M candidates and k=10
   that's **~1M× less data** off the GPU than reading every score. Readback
   latency, not math, is the usual silent killer — this removes it.

4. **Dimension-specialized shaders (the "custom" in custom kernel).**
   We generate WGSL per dimension using `override` constants so the compiler
   sees the dimension as a **compile-time constant**: loops fully unroll, the
   accumulator vectorizes to `vec4<f32>`, and tile width matches the dimension.
   A 768-dim kernel issues a tight unrolled `fma` chain with no loop overhead and
   no bounds checks — typically **1.5–2× over the generic loop** that must treat
   dimension as runtime data. (Generic kernel still exists for arbitrary dims.)

5. **Tiling + workgroup-shared memory reuse.**
   The query vector is loaded **once into shared memory** per workgroup and
   reused across the whole tile of corpus vectors, so it isn't re-fetched from
   global memory for every comparison. Coalesced, aligned loads of the corpus
   codes keep every memory transaction full-width — critical because uncoalesced
   access can waste most of the bus.

6. **Buffer pooling — zero per-query allocation.**
   GPU buffers (query, scratch, results) are pooled and reused. Allocating/
   freeing GPU memory per query stalls the pipeline; a warm pool makes a query a
   write-dispatch-read with no allocation on the hot path. Pipelines are compiled
   **once** and cached by `(dim, metric, bits, deviceLimits)`.

7. **Device-adaptive tuning + ANN to shrink the work.**
   Workgroup size, tile width, and dispatch chunking are micro-benched per device
   on first run (desktop vs. mobile have very different sweet spots and buffer
   limits) and the winner is persisted. Orthogonally, the **IVF index** means a
   query only scans `nprobe` clusters, not all 1M vectors — cutting the candidate
   set by 10–100× *before* the kernel even runs. Custom kernels make each
   comparison cheap; ANN makes there be far fewer comparisons.

### 14.3 Where the speed comes from, stacked

```
1M × 768 top-10, conceptual contribution (illustrative, to validate in bench):

  CPU/JS scalar ................................ baseline (1×)
  + WASM SIMD .................................. ~4×
  + GPU fp32 brute force (custom kernel) ....... ~10–20× more  (massively parallel)
  + quantize 4-bit (8× less bandwidth) ......... ~4–8× more
  + fused GPU top-k (no full readback) ......... removes readback cliff
  + dimension-specialized + tiling ............. ~1.5–2× more
  + IVF (scan ~1–3% of corpus) ................. ~30–100× fewer candidates
  ────────────────────────────────────────────────────────────────
  Net: tens-of-ms / >100 ms  →  < 10 ms target (§5, NFR-1)
```

> These multipliers are **design hypotheses to confirm in the §12.1 bench
> harness**, not measured results. They are listed so each is independently
> benchmarkable and so we can see which lever pays off on which device. The
> recall gate (NFR-5) guards the quantization/ANN levers from trading too much
> accuracy for speed.

### 14.4 Kernel sketch (illustrative WGSL shape)

```wgsl
// Specialized for DIM via override; corpus stored as packed int8 codes.
override DIM: u32 = 768u;
override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read> codes: array<u32>;   // packed quantized corpus
@group(0) @binding(1) var<storage, read> query: array<f32>;   // fp32 (asymmetric)
@group(0) @binding(2) var<storage, read_write> partial: array<KScore>; // per-wg top-k

var<workgroup> q_shared: array<f32, 768>;   // query loaded once per workgroup

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  // 1) cooperatively load query into shared memory (reused across the tile)
  for (var i = lid.x; i < DIM; i += WG) { q_shared[i] = query[i]; }
  workgroupBarrier();

  // 2) each thread scores one corpus vector: dequant in-register, unrolled FMA
  let row = gid.x;
  var acc = 0.0;
  // (loop fully unrolls because DIM is a compile-time override)
  for (var d = 0u; d < DIM; d += 4u) {
    let v = dequant4(codes, row, d);     // vec4<f32>, branchless
    acc = fma(v.x, q_shared[d],   acc);
    acc = fma(v.y, q_shared[d+1u], acc);
    acc = fma(v.z, q_shared[d+2u], acc);
    acc = fma(v.w, q_shared[d+3u], acc);
  }

  // 3) workgroup reduction keeps only top-k in shared mem; write k results, not N
  wg_topk_insert(row, acc);
  // ... only `k` (id,score) per workgroup leave the GPU
}
```

The point of the sketch: a **constant `DIM`**, a **query reused from shared
memory**, **dequant fused into the inner loop**, and **only `k` results leaving
the GPU** — each maps directly to a lever in §14.2.
