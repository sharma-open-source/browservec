# Configuration guide

How to choose the right index type and parameters for your use case.

## Decision tree

```
Is `fallback: 'wasm'` needed? (iOS, locked-down browser, etc.)
├── YES → CPU fallback. fp32 flat, or HNSW for sub-linear queries.
│            exact:  quantBits: 0, ann: omit
│            ANN:    quantBits: 0, ann: { type: 'hnsw' }
│            See [CPU fallback](../features.md#cpu-fallback--no-webgpu-nfr-7--m6)
│
└── NO (WebGPU available) →
    |
    How many vectors?  (< 100k is "small"; > 100k is "large")
    |
    ├── Small corpus, exact results needed
    │   └── flat fp32 — simplest, no tuning
    │       quantBits: 0, ann: omit
    │       [Basic flat example](../../examples/01-basic-flat.html)
    │
    ├── Memory-constrained or large corpus
    │   └── Quantized flat
    │       quantBits: 8 (int8, ~4× memory) or 4 (int4, ~8×)
    │       ann: omit
    │       [Quantization example](../../examples/02-quantization.html)
    │
    ├── Large corpus, speed over exact recall
    │   ├── Batch/rebuild-tolerant ingest → IVF fp32
    │   │       quantBits: 0, ann: { nlist: 1024 }
    │   │       [IVF example](../../examples/03-ivf-index.html)
    │   └── Continuous inserts, or l2 metric → HNSW
    │           quantBits: 0, ann: { type: 'hnsw' }
    │           add search: 'gpu' if queries arrive in batches (queryBatch)
    │           [HNSW example](../../examples/18-hnsw-index.html) ·
    │           [GPU graph search](../../examples/19-hnsw-gpu.html)
    │
    └── Large corpus + memory constrained
        └── IVF × quant (the 1M path — HNSW is fp32-only for now)
            quantBits: 8 (or 4), ann: { nlist: 4096 }
            [IVF×quant example](../../examples/04-ivf-quant-combo.html)
```

## Parameter tuning

### `dimension`

Typical embedding model output dimensions:

| Model | dim | Notes |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | Default sentence-transformers model |
| all-mpnet-base-v2 | 768 | Higher quality, larger |
| OpenAI text-embedding-3-small | 1536 | |
| OpenAI text-embedding-3-large | 3072 | |

Dims 384, 768, 1024, and 1536 have WGSL kernel fast-paths. Other dims work
too but the kernel falls back to a general loop — slightly slower per row.

### `metric`

| Metric | Score direction | When to use |
|---|---|---|
| `'cosine'` (default) | higher = closer | General semantic similarity. Vectors are auto-normalized. |
| `'dot'` | higher = closer | When vectors are already normalized or you want raw dot product. |
| `'l2'` | higher = closer (negated squared distance) | When Euclidean distance matters. **Flat and HNSW only** — IVF and quantized modes don't support l2 yet. |

### `quantBits`

| Value | Memory vs fp32 | Recall (with re-rank) | Best for |
|---|---|---|---|
| `0` (default) | 1× | 1.000 | Small corpora, exact results |
| `8` (int8) | ~4× | 1.000 | General purpose — no recall loss at 4× savings |
| `4` (int4) | ~8× | ~0.94+ | Tight memory, can accept slight recall loss |
| `1` (binary) | ~32× | ~0.98 (on clustered data) | Extreme compression, wide re-rank needed |

**Important:** Quantization is a memory lever, not a speed lever at small to
medium N. The unpack kernels are ALU-bound, so int4 and 1-bit queries can be
*s lower* than fp32 at 20k rows. The speed advantage only appears at scale
where the scan becomes bandwidth-bound.

All quantized modes require `metric: 'cosine'` or `'dot'`.

See [Quantization feature](../features.md#quantization--turboquant-int8-m3a) for details.

### `quant.rerankFactor`

Controls how many approximate candidates are fetched before the exact fp32
re-rank. Defaults depend on `quantBits`:

| quantBits | Default rerankFactor |
|---|---|
| 8 | 4 |
| 4 | 16 |
| 1 | 32 |

Higher values recover more recall at the cost of CPU re-rank time. Set to
`1` to disable re-rank entirely.

### `ann` (IVF)

| Parameter | Default | Effect |
|---|---|---|
| `nlist` | ≈ `sqrt(count)`, clamped [16, 4096] | More clusters = finer partitioning. Higher = better recall, but also more compute during build. |
| `nprobe` | ≈ 5% of nlist, min 1 | Clusters scanned per query. Higher = better recall, slower query. |
| `sampleSize` | 50,000 | Reservoir sample for k-means training. Increase for very large corpora. |
| `iters` | 8 | Lloyd iterations. More = better cluster quality, slower build. |

Tuning advice:

- **`nprobe`** is the main recall/latency knob. Start with the default, then
  increase until recall is acceptable.
- **`nlist`** should be large enough that each cluster has manageable size
  (e.g., at 1M rows, `nlist: 4096` gives ~244 rows/cluster on average).
- IVF requires `metric: 'cosine'` or `'dot'` — l2 is not supported.

### `ann` (HNSW)

Select with `ann: { type: 'hnsw' }` (fp32 only — `quantBits` must be 0):

| Parameter | Default | Effect |
|---|---|---|
| `M` | 16 | Out-degree per graph layer (layer 0 keeps 2·M). Higher = better recall and more robust graphs, but more memory and slower builds. 12–48 is the practical range; ≤ 32 required for `search: 'gpu'`. |
| `efConstruction` | 200 | Beam width while inserting. Higher = better graph quality, slower ingest. |
| `efSearch` | 64 | Beam width at query time (clamped to ≥ k; ≤ 256 on `search: 'gpu'`). The main recall/latency knob — overridable per query. |
| `seed` | fixed | Level RNG seed; same inserts + same seed = identical graph. |
| `search` | `'cpu'` | `'gpu'` runs the one-dispatch beam-search kernel — pick it when queries arrive in **batches** (`queryBatch`); single queries are faster on `'cpu'`. |

Tuning advice:

- **`efSearch`** is the knob to sweep first (like IVF's `nprobe`). On clustered
  data, recall@10 is typically ≳0.98 by `efSearch: 10–40`.
- Build cost is paid **at ingest** (in a Worker, so the UI stays responsive) —
  unlike IVF there is no rebuild after appends, so HNSW suits continuously
  growing stores; IVF suits bulk-load-then-query.
- HNSW supports **all metrics** including `l2`, and runs without WebGPU
  (`fallback: 'wasm'`).
- With `persist`/`export`, the graph is stored in the snapshot — reloads
  restore it directly (~180× faster than rebuilding) as long as `M` matches.

### `chunkRows`

GPU storage buffers have a device-dependent size cap
(`maxStorageBufferBindingSize` — default 128 MiB on many devices, 4 GiB on
Apple Metal). Once the corpus exceeds this limit, chunking activates
automatically and is transparent.

You only need `chunkRows` to force chunking at a smaller size for testing:

```ts
const db = await BrowserVec.create({
  dimension: 768,
  chunkRows: 25_000, // force chunking at 25k rows
});
```

## Persistence

| Config | When to use |
|---|---|
| `persist: { name: 'my-store' }` | Auto-save/load to OPFS (or IndexedDB). Data survives page reload. |
| `persist: { name: 'my-store', backend: 'indexeddb' }` | Force IndexedDB (e.g., when OPFS is unavailable). |
| `persist: { name: 'my-store', autoLoad: false }` | Configure persistence but skip auto-load on create. |
| `export()` / `import()` | Share or backup snapshots as blobs. |
| `{ encryption: { passphrase } }` | Encrypt at rest. Combine with any persist backend or export. |

## Per-query overrides

`QueryOptions` supports per-query overrides of store-level config:

```ts
db.query(q, {
  k: 20,           // return 20 neighbors (default 10)
  nprobe: 64,      // scan 64 clusters (IVF only)
  efSearch: 128,   // widen the search beam (HNSW only)
  rerank: false,   // skip exact re-rank (quantized only)
});

// Many queries in one call — a single GPU dispatch on an HNSW store
// with search: 'gpu'; a sequential loop everywhere else.
await db.queryBatch([q1, q2, q3], { k: 10, efSearch: 64 });
```
