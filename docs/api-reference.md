# API reference

Full public surface exported from `browservec` ([src/index.ts](../src/index.ts),
types in [src/types.ts](../src/types.ts)). This is a reference — for narrative
walkthroughs with runnable examples, see the [README](../README.md) sections
linked from each entry below.

## `BrowserVec`

### Static

| Member | Signature | Notes |
|---|---|---|
| `isSupported()` | `() => SupportInfo` | Capability probe: `{ webgpu, opfs, wasm }`. Call before `create()` to decide whether to pass `fallback: 'wasm'`. |
| `create(config)` | `(config: BrowserVecConfig) => Promise<BrowserVec>` | Acquires a device (or falls back to CPU), builds the store + index, optionally auto-loads a persisted snapshot. |
| `import(blob, config?)` | `(blob: Blob, config?) => Promise<BrowserVec>` | Rebuilds a store from a blob produced by `export()`. `config` accepts a subset of `BrowserVecConfig` (`device`, `persist`, `quantBits`, `quant`, `ann`, `chunkRows`, `fallback`) plus `encryption`. |

### Instance — ingestion

| Member | Signature | Notes |
|---|---|---|
| `add(record)` | `(record: VectorRecord) => Promise<void>` | Single insert. Prefer `addBatch` for >1 record — one GPU upload instead of N. |
| `addBatch(records)` | `(records: VectorRecord[]) => Promise<void>` | Primary ingestion path. Packs all vectors into one `Float32Array` before appending. |
| `addText(record)` | `(record: TextRecord) => Promise<void>` | Requires `config.embedder`. Embeds then `add()`s. |
| `addTexts(records)` | `(records: TextRecord[]) => Promise<void>` | Requires `config.embedder`. One embed call + one `addBatch`. |

### Instance — query

| Member | Signature | Notes |
|---|---|---|
| `query(vector, opts?)` | `(vector: Vector, opts?: QueryOptions) => Promise<QueryResult[]>` | Core search. Handles tombstone over-fetch, quantized re-rank, and per-query `nprobe`/`efSearch` overrides internally. |
| `queryBatch(vectors, opts?)` | `(vectors: Vector[], opts?: QueryOptions) => Promise<QueryResult[][]>` | Many queries at once. On an HNSW store with `search: 'gpu'` this is ONE compute dispatch (one workgroup per query); elsewhere it loops `query()`. Results are per-query, input order. |
| `queryText(text, opts?)` | `(text: string, opts?: QueryOptions) => Promise<QueryResult[]>` | Requires `config.embedder`. Embeds then `query()`s. |
| `get(id)` | `(id: string) => VectorRecord \| null` | Fetch a stored record (including its vector) by id. |

### Instance — mutation

| Member | Signature | Notes |
|---|---|---|
| `delete(id)` | `(id: string) => boolean` | Tombstones the row. `false` if `id` wasn't present. See [Deleting vectors](./features.md#deleting-vectors). |
| `update(record)` | `(record: VectorRecord) => Promise<boolean>` | Tombstone + re-add (upsert). Returns whether an existing row was replaced. |
| `compact()` | `() => Promise<number>` | Physically rebuilds the store + GPU index from live rows only, dropping tombstones. Returns rows removed (0 if nothing to do). |

### Instance — persistence

| Member | Signature | Notes |
|---|---|---|
| `save()` | `() => Promise<void>` | Requires `config.persist`. Writes a (compacted, optionally encrypted) snapshot to the configured backend. |
| `export(opts?)` | `(opts?: ExportOptions) => Promise<Blob>` | Serializes the whole store to a `Blob`, independent of any configured backend. |

### Instance — lifecycle / introspection

| Member | Signature | Notes |
|---|---|---|
| `dimension` | `get => number` | |
| `count` | `get => number` | Live (non-deleted) row count. |
| `metric` | `get => Metric` | |
| `stats()` | `() => Stats` | Point-in-time snapshot: counts, device, quant/IVF/chunking/worker-offload state. See table below. |
| `destroy()` | `() => void` | Frees GPU resources. No-op device teardown under the CPU fallback. |

## Config types

### `BrowserVecConfig` (passed to `create()`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `dimension` | `number` | required | Positive integer. 384/768/1024/1536 are fast-pathed. |
| `metric` | `'cosine' \| 'dot' \| 'l2'` | `'cosine'` | IVF and quantization currently require cosine/dot (`l2` works on flat and HNSW). |
| `normalize` | `boolean` | `metric === 'cosine'` | Normalize vectors on insert. |
| `device` | `GPUDevice` | — | Reuse an existing device instead of requesting a new one. |
| `fallback` | `'wasm' \| 'error'` | `'error'` | `'wasm'` degrades to a CPU path (exact fp32 flat scan, or HNSW) when WebGPU is unavailable, instead of throwing. |
| `persist` | `PersistConfig` | — | Enables `save()` / auto-load. |
| `embedder` | `Embedder` | — | Enables `addText`/`addTexts`/`queryText`. Its `dimension` must match. |
| `quantBits` | `0 \| 1 \| 4 \| 8` | `0` | `0` = fp32. `8`/`4`/`1` = TurboQuant int8/int4/binary. Requires cosine/dot. |
| `quant` | `QuantConfig` | — | Tuning for quantized mode (seed, rounds, rerank factor). |
| `ann` | `ANNConfig` | — | Enables an approximate index — IVF (`type` omitted or `'ivf'`) or HNSW (`type: 'hnsw'`). Omit for exact flat search. |
| `chunkRows` | `number` | auto | Force corpus chunking below the auto-trigger threshold (mainly for tests/demos). |

### `ANNConfig` = `IVFConfig | HNSWConfig`

A discriminated union on `type`. Omitting `type` (or `'ivf'`) selects IVF, so pre-M7 configs are unchanged.

**`IVFConfig`** — `type?: 'ivf'`, `nlist` (default ≈ `sqrt(count)`, clamped `[16, 4096]`), `nprobe` (default: the auto-tuned value when `targetRecall` is set, else ≈ 5% of `nlist`; setting it explicitly disables auto-tuning), `targetRecall` (0–1, e.g. `0.95` — auto-tune `nprobe`: after each build, recall@10 is estimated on ≤ 32 sample queries drawn from the corpus, one exact scan each, and the smallest `nprobe` meeting the target becomes the default; the result surfaces as `stats().nprobe` / `stats().tunedRecall`), `sampleSize` (default `50_000`, reservoir sample for k-means training), `iters` (default `8` Lloyd iterations), `seed`.

**`HNSWConfig`** — `type: 'hnsw'`, `M` (graph out-degree per layer, layer 0 keeps 2·M; default `16`), `efConstruction` (build beam width; default `200`), `efSearch` (query beam width, clamped ≥ k; default `64`), `seed` (level RNG, reproducible builds), `search` (`'cpu' | 'gpu'`, default `'cpu'` — `'gpu'` runs the single-dispatch beam-search kernel; needs WebGPU, `M ≤ 32`, `efSearch ≤ 256`, corpus within one storage buffer, and shines on `queryBatch`).

### `QuantConfig`

`seed` (fixed default so reloads re-quantize identically), `rounds` (default `2` Hadamard rounds), `rerankFactor` (default `4` for int8, `16` for int4, `32` for binary — set `1` to disable re-rank).

### `PersistConfig`

`name` (required snapshot key), `backend` (`'opfs' | 'indexeddb' | 'auto'`, default `'auto'`), `autoLoad` (default `true`), `encryption` (`{ passphrase }` — AES-256-GCM + PBKDF2).

### `QueryOptions`

`k` (default `10`), `filter` (metadata predicate — see [`MetadataFilter`](#metadatafilter)), `rerank` (override the store's default re-rank behavior, quantized stores only), `nprobe` (per-query override, IVF stores only), `efSearch` (per-query beam-width override, HNSW stores only — higher = better recall, slower).

### `MetadataFilter`

Mongo-ish predicate over record metadata (FR-7). Fields AND together; a bare
value is shorthand for `{ $eq: value }`:

```ts
await db.query(q, { k: 8, filter: { lang: 'en', year: { $gte: 2020, $lt: 2025 } } });
```

Operators: `$eq`, `$ne` (also matches records missing the field), `$in` (array
of values), and numeric `$gt` / `$gte` / `$lt` / `$lte` (only match stored
numbers). Unknown operators throw.

Execution picks a strategy per query:

- **Tiny match set (≤ ~4k rows)** — bypasses the index: matching rows are scored
  exactly against the store's CPU-side fp32 vectors (`O(matches · dim)`,
  independent of corpus size). Exact on *every* index type, including
  quantized/IVF/HNSW.
- **Flat stores (fp32 or quantized), larger match sets** — **in-index GPU
  filtering**: a mask pass stamps non-matching rows' scores to `-FLT_MAX`
  between the distance kernel and the top-k reduction, so the GPU returns
  exactly the top-k *matching* rows with no over-fetch. Full GPU speed at any
  selectivity; exact for fp32, and quantized stores keep their usual exact
  re-rank. Mask upload is 1 bit/row (~125 KB per query at 1M rows).
- **IVF/HNSW/CPU-fallback stores, larger match sets** — selective filters take
  the exact CPU scan; filters matching nearly everything stay on the index
  path, over-fetching by the excluded count and post-filtering (the tombstone
  mechanism). In-index filtering for IVF and HNSW (masked cluster scans /
  filtered graph traversal) is future work.

## Result / status types

### `QueryResult`

`{ id: string, score: number, metadata?: Metadata }` — score is metric-dependent: higher = closer for cosine/dot, lower = closer for l2.

### `Stats` (from `stats()`)

`count`, `deleted?`, `dimension`, `metric`, `device: 'webgpu' | 'wasm'`, `lastQueryMs?`, `lastQueryGpuMs?` (GPU kernel portion), `lastQueryCpuMs?` (CPU re-rank/filtering portion), `persist?: 'opfs' | 'indexeddb'`, `quantBits`, `nlist?` (IVF cluster count once built), `nprobe?` (IVF default clusters-per-query — explicit, auto-tuned, or the 5% heuristic — once built), `tunedRecall?` (estimated recall@10 at the auto-tuned `nprobe`; only present when `targetRecall` tuning ran), `maxLevel?` (HNSW top graph layer once non-empty), `graphSearch?: 'gpu' | 'cpu'` (which engine answers HNSW queries), `chunks?` (>1 once the corpus spans multiple GPU buffers), `ingest?: 'worker' | 'main-thread'` (where quantized rotate+quantize ran), `train?: 'worker' | 'main-thread'` (where the ANN index build ran — IVF k-means updates, or HNSW graph construction).

### `SupportInfo` (from `isSupported()`)

`{ webgpu: boolean, opfs: boolean, wasm: boolean }`.

## Bundled embedders

| Export | Notes |
|---|---|
| `hashingEmbedder(options?)` | Zero-dependency feature-hashing embedder. Offline, non-semantic (lexical) — good for tests/demos. See [`HashingEmbedderOptions`](../src/embed/hashing.ts). |
| `transformersEmbedder(options?)` | Lazy-loaded `@xenova/transformers` adapter for real semantic embeddings. See [`TransformersEmbedderOptions`](../src/embed/transformers.ts). |

Bring your own by implementing the `Embedder` interface: `{ dimension: number, embed(texts: string[]): Promise<Float32Array[]> }`.
