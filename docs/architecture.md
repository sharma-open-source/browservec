# Architecture

## Layers

```
BrowserVec (src/index.ts)          orchestration: config → device/index selection,
                                    query re-rank, persistence, delete/compact
  ├── Store (src/store/store.ts)   CPU-side id↔row map, tombstones, raw fp32 vectors
  │                                 — the persistence source of truth
  ├── VectorIndex (interface)      one implementation is live per BrowserVec instance:
  │     ├── FlatIndex               fp32 brute-force            (src/index/flat.ts)
  │     ├── QuantIndex               fp32 → int8/int4/1-bit      (src/index/quant.ts)
  │     ├── IVFIndex                 fp32 clustered              (src/index/ivf.ts)
  │     ├── IVFQuantIndex            clustered + quantized       (src/index/ivfquant.ts)
  │     ├── HNSWIndex                graph ANN, CPU/Worker-built (src/index/hnsw.ts)
  │     │     └── HNSWGpuSearcher    optional GPU beam search    (src/index/hnswGpu.ts)
  │     └── CpuIndex                 WASM-SIMD/scalar fallback   (src/fallback/cpu.ts)
  └── PersistenceBackend           OPFS / IndexedDB               (src/persist/*)
```

`BrowserVec` never talks to WGSL or GPU buffers directly — everything GPU-shaped
goes through a `VectorIndex`. This is what lets `create()` pick `FlatIndex` vs.
`CpuIndex` (or any of the quantized/IVF variants) behind one interface:

```ts
// src/index/flat.ts
export interface VectorIndex {
  readonly size: number;
  append(data: Float32Array, count: number): void | Promise<void>;
  query(queryVec: Float32Array, k: number): Promise<FlatHit[]>;
  destroy(): void;
}
```

## Startup: which index gets built

`BrowserVec.create()` (src/index.ts):

1. Validates `dimension`, resolves `metric`/`normalize` defaults.
2. `acquireContext(device, fallback)` — tries `acquireDevice()`
   (src/engine/device.ts): reuses an injected `GPUDevice`, or requests an
   adapter (`powerPreference: 'high-performance'`) and device with the
   largest limits the adapter allows (`pickLimits`, capped at 1 GiB storage
   buffers). Returns `null` (not a throw) when WebGPU is genuinely absent
   *and* `fallback: 'wasm'` was requested — a real GPU error under
   `fallback: 'error'` (the default) still throws `WebGPUUnavailableError`.
3. `indexFactory(...)` picks the concrete `VectorIndex` constructor based on
   `(quantBits, ann, ctx)` — see the decision table below — but doesn't
   build it yet. The same factory is reused later by `compact()` to rebuild
   a fresh empty index.
4. Builds the index, wires up persistence (`initPersistence`, which
   auto-loads a snapshot if one exists), and wires the optional `embedder`.

### Index selection (`buildIndex` / `buildCpuIndex` in src/index.ts)

| `ctx` | `ann` | `quantBits` | Index |
|---|---|---|---|
| GPU | no | `0` | `FlatIndex` |
| GPU | no | `1\|4\|8` | `QuantIndex` |
| GPU | `type: 'ivf'` (or omitted `type`) | `0` | `IVFIndex` |
| GPU | `type: 'ivf'` (or omitted `type`) | `1\|4\|8` | `IVFQuantIndex` |
| GPU | `type: 'hnsw'` | `0` | `HNSWIndex` (CPU/Worker graph; + `HNSWGpuSearcher` when `search: 'gpu'`) |
| any | `type: 'hnsw'` | `1\|4\|8` | throws — HNSW × quant is future work |
| `null` (CPU fallback) | no | `0` | `CpuIndex` |
| `null` (CPU fallback) | `type: 'hnsw'` | `0` | `HNSWIndex` — the graph index is CPU-native, so it needs no GPU |
| `null` (CPU fallback) | IVF, or `1\|4\|8` | — | throws — quant/IVF are GPU-throughput features |

`l2` metric is rejected for IVF and quantized modes at this same point
(cosine/dot only; flat and HNSW accept all three metrics) — it's a validation
error, not a silent fallback to flat.

## Ingest path

`add`/`addBatch` (src/index.ts) both go through `Store.prepare()` (dimension
check + optional L2 normalize) and `Store.insert()` (assigns the next row,
appends to the CPU-side `raw` buffer — the persistence source of truth) before
calling `index.append(packedVectors, count)`. `addBatch` packs all records into
one `Float32Array` so there's one GPU upload regardless of batch size.

For `FlatIndex`, `append` writes straight into a `ChunkedCorpus`
(src/engine/buffers.ts) — a set of GPU buffers, each below
`maxStorageBufferBindingSize`, so a corpus that would overflow one buffer
transparently spans several (§NFR-10; see `Stats.chunks`). Quantized indexes
instead route through the rotate+quantize encoder (see
[internals.md](./internals.md#quantization)), which may run on a Worker.

## Query path

`BrowserVec.query()` (src/index.ts):

1. Validates/normalizes the query vector.
2. If the store has tombstones, over-fetches (`k + deletedCount`, capped at
   `index.size`) so at least `k` live results survive filtering — a no-op
   when nothing is deleted.
3. Quantized stores (`quantBits > 0`) fetch `kEff * rerankFactor` approximate
   candidates from the GPU, then `rerankExact()` re-scores them against the
   exact fp32 vectors in `Store` (CPU dot product) and keeps the true top-k.
   This is what recovers recall lost to quantization — the GPU kernel only
   has to narrow millions of rows to a few hundred candidates cheaply.
4. Results are mapped from GPU row indices back to `{ id, score, metadata }`
   via `Store.entryByRow`, skipping any row still marked deleted.

Inside a `VectorIndex.query()`: write the query vector to a GPU buffer,
dispatch the distance kernel (WGSL, specialized per `dim`/`metric` at
pipeline-build time — see `src/engine/wgsl/`), then reduce to top-k. Above
`GPU_TOPK_MIN_ROWS` the top-k reduction itself runs on the GPU
(`src/index/gpuTopk.ts`, §14.2 lever 3) so only a short candidate list is
read back to the CPU; below that threshold the full score array is read back
and reduced with a CPU `topK()`.

## Delete / update / compact

Deletes are tombstones (`Store.delete`): the id is removed from the id→row
map immediately (so queries filter it out and `count` drops), but the row's
vector stays in the CPU buffer and the GPU index until compaction — cheap,
O(1) delete at the cost of stale GPU memory. `update()` is
tombstone-then-`add()` (so repeated updates accumulate tombstones).
`compact()` rebuilds `Store` dense (row 0..live-1) and calls
`index.destroy()` + `makeIndex()` + `append()` on the live vectors only — the
only way to reclaim GPU memory without a save/reload round-trip.

## Persistence

`save()`/`export()` serialize a *compacted* snapshot (`Store.liveEntries()` +
`Store.liveVectors()` — tombstones are never persisted) via
`src/persist/format.ts`, optionally passed through
`src/persist/crypto.ts`'s AES-256-GCM envelope, and written through a
`PersistenceBackend` (`src/persist/backend.ts`) selected by
`{ backend: 'opfs' | 'indexeddb' | 'auto' }`. Loading is the same pipeline in
reverse: decrypt-if-needed → deserialize → `Store.insert()` each row in order
→ one `index.append()` of the whole snapshot. HNSW stores embed their graph in
the snapshot (format v2, M7c); when the loaded config matches (`type: 'hnsw'`,
same `M`), load calls `index.loadWithGraph()` instead of `append()`, restoring
the graph directly and skipping the O(N·efConstruction) rebuild — any mismatch,
or a snapshot taken with pending tombstones, falls back to the append path. See
[internals.md](./internals.md#persistence-format) for the byte layout.

## CPU / WASM fallback

When `acquireContext` returns `null` (see startup above), `BrowserVec` builds
a `CpuIndex` (src/fallback/cpu.ts) instead of a GPU `VectorIndex` — same
`VectorIndex` interface, so nothing else in `BrowserVec` branches on it. The
hot loop is a hand-written WASM-SIMD kernel (src/fallback/simd.ts) with a
scalar-JS fallback for engines lacking WASM/SIMD; both are bit-exact with the
GPU path. See [internals.md](./internals.md#cpu-fallback) for the kernel
details.

## Worker offload

Three CPU-heavy operations can run on a Worker instead of the main thread —
quantized ingest (rotate+quantize, `src/quant/encoder.ts` +
`src/quant/quantize.worker.ts`), IVF k-means centroid mean-updates
(`src/index/kmeansTrainer.ts` + `src/index/kmeans.worker.ts`), and the HNSW
graph (`src/index/hnsw.ts` + `src/index/hnsw.worker.ts` — here the Worker *owns*
the graph and corpus copy: appends stream in, only top-k results cross back, so
build **and** CPU search stay off the main thread). All follow the same seam
pattern: try to spin up a Worker, transparently fall back to running in-thread
if Workers aren't available, and report which path ran via `Stats.ingest` /
`Stats.train`. See [internals.md](./internals.md#worker-offload-seam).

## How it maps to the design

The authoritative file↔spec-section index — use it to jump from a
[REQUIREMENTS.md](../REQUIREMENTS.md) section (e.g. `§NFR-10`) to the file that
implements it.

| Code | REQUIREMENTS.md |
|------|-----------------|
| [src/engine/wgsl/distance.ts](../src/engine/wgsl/distance.ts) | §14.2 levers 4 & 5 — `DIM`/`WG` baked as constants → unrolled `vec4` FMA; query staged in workgroup shared memory |
| [src/index/flat.ts](../src/index/flat.ts) | §9 M1 — flat GPU brute-force; GPU top-k past 4k rows, CPU top-k below |
| [src/engine/wgsl/topk.ts](../src/engine/wgsl/topk.ts) / [src/index/gpuTopk.ts](../src/index/gpuTopk.ts) | §14.2 lever 3 — on-GPU top-k reduction (segment argmax → short readback), on the flat fp32 + quantized int8/int4 paths |
| [src/engine/buffers.ts](../src/engine/buffers.ts) | §NFR-10 — `ChunkedCorpus`: corpus split across GPU buffers past the per-buffer limit |
| [src/engine/device.ts](../src/engine/device.ts) | §NFR-11 — device acquisition + device-loss wiring |
| [src/persist/format.ts](../src/persist/format.ts) | §FR-16 — versioned binary snapshot codec |
| [src/persist/opfs.ts](../src/persist/opfs.ts) / [indexeddb.ts](../src/persist/indexeddb.ts) | §NFR-6 — OPFS primary, IndexedDB fallback |
| [src/persist/crypto.ts](../src/persist/crypto.ts) | §M6 — AES-256-GCM + PBKDF2 snapshot encryption at rest |
| [src/fallback/cpu.ts](../src/fallback/cpu.ts) / [simd.ts](../src/fallback/simd.ts) | §NFR-7 / M6 — exact CPU flat scan for the no-WebGPU fallback (identical results); WASM-SIMD `f32x4` kernel (~7×), scalar JS fallback |
| [src/store/store.ts](../src/store/store.ts) | §NFR-11 — CPU raw vectors are the persistence source of truth |
| [src/quant/rotator.ts](../src/quant/rotator.ts) | §6 — data-oblivious randomized Hadamard rotation |
| [src/quant/codec.ts](../src/quant/codec.ts) | §6 / §9 M3b — int8 snorm codec (matches `unpack4x8snorm`) + int4 nibble + 1-bit sign pack/unpack |
| [src/engine/wgsl/distanceQ8.ts](../src/engine/wgsl/distanceQ8.ts) | §6.2 FR-Q4 / §14.2 lever 1 — int8 kernel, dequant in-shader |
| [src/engine/wgsl/distanceQ4.ts](../src/engine/wgsl/distanceQ4.ts) | §9 M3b — int4 kernel, manual nibble unpack (~8× less memory) |
| [src/engine/wgsl/distanceQ1.ts](../src/engine/wgsl/distanceQ1.ts) | §9 M3b — 1-bit binary kernel, sign-bit unpack + asymmetric scoring (~32× less memory) |
| [src/index/quant.ts](../src/index/quant.ts) | §6 — quantized index + asymmetric query |
| [src/index/kmeans.ts](../src/index/kmeans.ts) | §9 M4 — k-means (CPU helpers + init/update for the GPU-assisted loop) |
| [src/index/kmeans.worker.ts](../src/index/kmeans.worker.ts) | §NFR-8 — k-means mean-update Worker; inlined via `?worker&inline` |
| [src/index/kmeansTrainer.ts](../src/index/kmeansTrainer.ts) | §NFR-8 — trainer seam: Worker offload + in-thread fallback (reproducible either way) |
| [src/engine/wgsl/assign.ts](../src/engine/wgsl/assign.ts) | §9 M4 — GPU centroid-assignment kernel (argmax dot) |
| [src/engine/wgsl/distance.ts](../src/engine/wgsl/distance.ts) (indexed) | §9 M4 — indexed scan: score rows from a candidate-id list |
| [src/index/ivf.ts](../src/index/ivf.ts) | §9 M4 / §NFR-10 — IVF index: reservoir sample, GPU-assisted build, probe→gather→chunked scan→GPU top-k |
| [src/engine/wgsl/assignQ8.ts](../src/engine/wgsl/assignQ8.ts) | §9 M4 — quantized centroid-assignment kernel (dequant + argmax dot) |
| [src/engine/wgsl/assignQ4.ts](../src/engine/wgsl/assignQ4.ts) | §9 M3b+M4 — 4-bit quantized centroid-assignment kernel |
| [src/engine/wgsl/assignQ1.ts](../src/engine/wgsl/assignQ1.ts) | §9 M3b+M4 — 1-bit binary centroid-assignment kernel (IVF×1-bit combo) |
| [src/index/ivfquant.ts](../src/index/ivfquant.ts) | §9 M4 / §NFR-10 — IVF × int8/int4 combo: clustered + quantized + chunked + GPU top-k, the 1M path |
| [src/embed/hashing.ts](../src/embed/hashing.ts) | §9 M5 — zero-dep feature-hashing embedder (offline, lexical) |
| [src/embed/transformers.ts](../src/embed/transformers.ts) | §9 M5 — optional transformers.js adapter (lazy, real semantics) |
| [src/quant/encode.ts](../src/quant/encode.ts) | §NFR-8 — shared BatchEncoder (rotate+quantize), one impl for both threads |
| [src/quant/quantize.worker.ts](../src/quant/quantize.worker.ts) | §NFR-8 — ingest Worker; inlined into the bundle via `?worker&inline` |
| [src/quant/encoder.ts](../src/quant/encoder.ts) | §NFR-8 — encoder seam: Worker offload + in-thread fallback |
| [src/index/hnswGraph.ts](../src/index/hnswGraph.ts) | M7 (post-spec) — HNSW graph core: typed-array adjacency, beam search, diversity heuristic, serialize/load |
| [src/index/hnsw.worker.ts](../src/index/hnsw.worker.ts) | M7 / §NFR-8 — graph Worker (owns graph + corpus copy; build and CPU search off-thread) |
| [src/index/hnsw.ts](../src/index/hnsw.ts) | M7 — HNSWIndex seam: Worker/in-thread/GPU dispatch, batch queries, graph persistence hooks |
| [src/engine/wgsl/graphSearch.ts](../src/engine/wgsl/graphSearch.ts) | M7b — CAGRA-style beam-search kernel: whole search in one dispatch, one workgroup per query |
| [src/index/hnswGpu.ts](../src/index/hnswGpu.ts) | M7b — GPU graph executor: corpus/adjacency mirrors, batched dispatch, readback dedup |
