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
| GPU | yes | `0` | `IVFIndex` |
| GPU | yes | `1\|4\|8` | `IVFQuantIndex` |
| `null` (CPU fallback) | — | `0` | `CpuIndex` |
| `null` (CPU fallback) | any | `1\|4\|8` or `ann` set | throws — CPU fallback is fp32-flat only |

`l2` metric is rejected for `ann` and quantized modes at this same point
(cosine/dot only) — it's a validation error, not a silent fallback to flat.

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
→ one `index.append()` of the whole snapshot. See
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

Two CPU-heavy operations can run on a Worker instead of the main thread —
quantized ingest (rotate+quantize, `src/quant/encoder.ts` +
`src/quant/quantize.worker.ts`) and IVF k-means centroid mean-updates
(`src/index/kmeansTrainer.ts` + `src/index/kmeans.worker.ts`). Both follow the
same seam pattern: try to spin up a Worker, transparently fall back to running
in-thread if Workers aren't available, and report which path ran via
`Stats.ingest` / `Stats.train`. See
[internals.md](./internals.md#worker-offload-seam).

## Where to look for a given requirement

The README's [How it maps to the design](../README.md#how-it-maps-to-the-design)
table is the authoritative file↔spec-section index — use it to jump from a
`REQUIREMENTS.md` section (e.g. `§NFR-10`) to the implementing file.
