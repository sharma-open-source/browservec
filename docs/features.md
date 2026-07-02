# Feature deep-dives

Moved out of the top-level README to keep that page skimmable. Each section
below was originally a README subsection under **API** — same content, same
code samples, just here instead. See the [README](../README.md) for the
quick-start and top-level feature list, and [architecture.md](./architecture.md)
for how these pieces fit together.

## Deleting vectors

Deletes are **tombstones**: the id is removed from results immediately (cheap — no
GPU rewrite), and the row's memory is physically reclaimed when the store is next
persisted + reloaded (snapshots are compacted to live rows only). Queries transparently
over-fetch past tombstones so they still return a full `k`.

```ts
db.delete('doc-42');          // → true (false if the id wasn't present)
db.count;                     // live count, excludes deleted
db.stats().deleted;           // tombstones awaiting compaction
db.query(q, { k: 10 });       // 'doc-42' never appears; still returns 10 hits

await db.save();              // snapshot written without the deleted rows
// re-adding a deleted id is allowed and creates a fresh row.
```

Works across every index (flat, quantized, IVF) — filtering happens above the
kernels, so no index type needs special handling. `update(record)` replaces a
vector (tombstone + append; upserts if new), and `await db.compact()` physically
drops tombstones in place — rebuilding the store + index from live vectors only,
reclaiming GPU memory without a save/reload round-trip.

> **Try it:** [`examples/08-deletion-compaction.html`](../examples/08-deletion-compaction.html)

## Persistence (M2)

```ts
// Auto-loads an existing snapshot named 'docs' on create (cold start).
const db = await BrowserVec.create({
  dimension: 768,
  persist: { name: 'docs', backend: 'auto' }, // 'auto' = OPFS, else IndexedDB
});

await db.addBatch(records);
await db.save();             // flush snapshot to storage

// later / new tab: same name → restored automatically, no re-ingest
const blob = await db.export();              // single-blob snapshot
const copy = await BrowserVec.import(blob);  // rebuild from a blob
```

The snapshot is a versioned binary blob (`BVEC` magic + header + metadata JSON +
packed Float32 vectors). Vectors are persisted **already normalized** (as
searched), so a reload reproduces query results exactly. A dimension/metric
mismatch on load throws rather than silently corrupting results. Byte layout:
[internals.md#persistence-format](./internals.md#persistence-format-srcpersistformatts).

> **Try it:** [`examples/06-persistence.html`](../examples/06-persistence.html)

## Encryption at rest (M6)

The point of an in-browser store is that data never leaves the device — but a
persisted snapshot (OPFS/IndexedDB) or an exported blob is otherwise **plaintext
embeddings + metadata**, which leak content if the device is shared or the storage
is exfiltrated. Pass a passphrase and snapshots are sealed in an AES-256-GCM
envelope (key stretched from the passphrase with PBKDF2-SHA256); auto-load
decrypts, and a wrong passphrase or any tampering **fails loudly** (GCM auth)
rather than returning garbage:

```ts
const db = await BrowserVec.create({
  dimension: 768,
  persist: { name: 'docs', encryption: { passphrase } }, // encrypted on save()
});
await db.addBatch(records);
await db.save();                       // AES-GCM envelope written to storage

// later / new tab — same passphrase auto-decrypts; wrong passphrase throws.
const reopened = await BrowserVec.create({
  dimension: 768,
  persist: { name: 'docs', encryption: { passphrase } },
});

// export/import blobs can be encrypted too.
const blob = await db.export({ encryption: { passphrase } });
const copy = await BrowserVec.import(blob, { encryption: { passphrase } });
```

Each save uses a fresh random salt + IV, so the same data encrypts differently
every time. Requires WebCrypto (`crypto.subtle`), available in browsers and Node
18+. The demo's **Encrypted persist** button round-trips a store and shows that a
wrong passphrase, a missing passphrase, and a tampered blob are all rejected.
Envelope byte layout: [internals.md#encryption-envelope](./internals.md#encryption-envelope-srcpersistcryptots).

> **Try it:** [`examples/07-encryption.html`](../examples/07-encryption.html)

## Quantization — TurboQuant int8 (M3a)

```ts
const db = await BrowserVec.create({
  dimension: 768,
  metric: 'cosine',          // quantization supports cosine/dot only
  quantBits: 8,              // 0 = fp32 (default), 8 = int8 (~4×), 4 = int4 (~8×), 1 = binary (~32×)
  quant: { seed: 0x9e3779b9, rounds: 2, rerankFactor: 4 },
});

await db.addBatch(records);                 // rotated + quantized on insert
const hits = await db.query(q, { k: 10 });  // exact re-rank by default
const raw  = await db.query(q, { k: 10, rerank: false }); // raw quantized
```

The corpus is **rotated** (data-oblivious randomized Hadamard — depends only on
the seed, never the data) and stored as packed int8 codes with a per-row scale.
Queries rotate the same way and run **asymmetrically** (fp32 query × int8 corpus)
through a kernel that dequantizes 4 coordinates per `unpack4x8snorm`. The GPU
narrows the corpus to `k·rerankFactor` candidates, then an **exact fp32 re-rank**
(against the retained vectors) recovers recall. Rotation is orthonormal, so it
preserves dot/cosine exactly — quantization is the only error source. The
default seed is fixed, so a persisted store re-quantizes identically on reload.
See [REQUIREMENTS.md §6](../REQUIREMENTS.md) for the TurboQuant background and IP note.

> **Try it:** [`examples/02-quantization.html`](../examples/02-quantization.html)

**Sub-byte (`quantBits: 4`, M3b).** int4 packs 8 coords per word — ~8× less memory
than fp32 (6× after the 768→1024 pad). With no hardware unpack the kernel extracts
nibbles by hand. int4's 15 levels are coarse, so on near-duplicate data the true
neighbours cluster into near-tied scores and need a **wider re-rank pool** to
survive — the default `rerankFactor` is therefore **16 for int4** (vs 4 for int8).
With that, int4×IVF reaches **recall@10 ≈ 0.94** at 8× memory (int8×IVF is 1.0 at
4×); raising `rerankFactor` further trades CPU re-rank time for more recall. int4
is the pick when memory is the binding constraint; int8 when you want recall 1.0.

**Binary (`quantBits: 1`, M3b).** The extreme rung: keep only each rotated
coordinate's **sign**, so 32 coords pack into one u32 — **~32× less memory** than
fp32 (16× after the 768→1024 pad, ~2 KB/row → 128 B). A row reconstructs as
`sign · scale` where `scale = mean|coord|` (the magnitude that minimizes sign-
quantization error), and the kernel scores asymmetrically — accumulating
`Σ sign·query` against the fp32 rotated query, then multiplying by the row scale.
Binary is the coarsest code, so it leans hardest on re-rank: the default
`rerankFactor` is **32**. With that, flat 1-bit reaches **recall@10 ≈ 0.98** on
clustered data (validated in a Node harness that mirrors the WGSL kernel and cross-
checks it against the reference dequant to float precision). Rotation is what makes
this work — it spreads energy across coordinates so the sign bits carry signal.
Binary composes with IVF too (`quantBits: 1` + `ann`) — a binary centroid-assign
kernel buckets rows by their sign codes, agreeing with fp32 assignment ~92% of the
time, and the binary scan within probed lists is within ~0.005 recall of an exact
scan of those same candidates (Node harness). It's the pick for the largest corpora
on the tightest memory budgets.

**`quantBits` is a memory lever, not a speed lever at small/medium N.** The unpack
kernels are ALU-bound: extracting nibbles/sign bits by hand costs more per row than
fp32's straight `vec4` loads, so on a device-report matrix (Apple M-series, 20k×384)
query time actually *rises* as bits shrink — int8 ≈ fp32 < int4 < 1-bit. The tighter
codes only win throughput at scale, where the corpus is large enough to be *bandwidth*-
bound and moving 8×/32× fewer bytes dominates. So choose the bit-width for the memory
budget you need (fit 1M in RAM/one buffer), not for per-query latency at 20k rows;
below the bandwidth crossover, fp32 or int8 is the faster scan. See
[Benchmarks](../README.md#benchmarks) for measured numbers.

`metric: 'cosine'` normalizes vectors on insert and the query, so the kernel
computes a plain dot product (cosine = dot of unit vectors). `'l2'` returns
negative squared distance so that **higher score = closer** holds for every
metric and a single top-k path works.

> **Try it:** [`examples/13-all-metrics.html`](../examples/13-all-metrics.html)

## Approximate search — IVF (M4)

```ts
const db = await BrowserVec.create({
  dimension: 768,
  metric: 'cosine',           // IVF supports cosine/dot
  ann: { nlist: 1024 },       // omit nlist to auto-pick ≈ sqrt(count)
});

await db.addBatch(records);
const hits = await db.query(q, { k: 10 });            // default nprobe
const wide = await db.query(q, { k: 10, nprobe: 32 }); // scan more clusters → higher recall
```

Instead of scoring every row, the corpus is clustered into `nlist` cells with
**k-means** (assignment runs on the GPU — at dim≈768 a CPU k-means would be far
too slow; the centroid-mean update runs on the CPU, offloaded to a Worker so the
build stays interactive — see [Worker offload](#worker-ingest-offload-nfr-8)). A query scores
its query against the centroids, picks the `nprobe` nearest cells, and the GPU
scans **only those rows** through an *indexed* distance kernel (one dispatch over
a candidate-id list — no physical reorder). `nprobe` trades recall for latency;
the true neighbour is missed only when it lands in an unprobed cell. The index
builds lazily on the first query after an append — a one-time `O(nlist·N)` cost.
Real embeddings cluster well, so recall@10 ≥ 0.95 is reachable at a small
`nprobe`; uniform-random vectors are a worst case (the demo uses clustered data).

> **Try it:** [`examples/03-ivf-index.html`](../examples/03-ivf-index.html)

**IVF × int8 (the 1M path).** Add `quantBits: 8` to an `ann` store and the corpus
is both clustered *and* int8-quantized — so ~1M×768 fits in a single ~1 GB buffer
*and* each query scans only the probed lists:

```ts
const db = await BrowserVec.create({
  dimension: 768,
  metric: 'cosine',
  quantBits: 8,    // int8 codes
  ann: {},         // + IVF clustering
});
```

Rows are clustered in the rotated space and assigned by their *quantized*
representation (the same one they're scored with). The query rotates, picks
`nprobe` cells, the GPU scans only those rows through the indexed dequant kernel,
then the usual exact fp32 re-rank recovers recall. Works at **`quantBits: 8` (4×),
`4` (8×), or `1` (binary, ~32×)** — at int4 the corpus packs into ~0.5 GB for 1M×768
and re-rank does the heavy lifting (validated: raw ~0.86 → re-rank ~1.0); at 1-bit
it's ~128 B/row, so 1M×768 fits in well under 200 MB and the binary scan+rerank
tracks the exact scan of the probed set to within ~0.005 recall.

> **Try it:** [`examples/04-ivf-quant-combo.html`](../examples/04-ivf-quant-combo.html)

## Text retrieval — on-device embedder (M5)

Pass an `embedder` and you get `addText`/`addTexts`/`queryText` — text in, results
out, fully offline:

```ts
import { BrowserVec, hashingEmbedder } from 'browservec';

const db = await BrowserVec.create({
  dimension: 384,
  metric: 'cosine',
  embedder: hashingEmbedder({ dimension: 384 }), // zero-dep, offline, lexical
});

await db.addTexts([
  { id: 'd1', text: 'WebGPU runs compute kernels on the GPU', metadata: { text: '…' } },
  { id: 'd2', text: 'int8 quantization saves memory bandwidth' },
]);

const hits = await db.queryText('gpu compute', { k: 5 });
```

`hashingEmbedder` needs no model download (feature hashing → keyword-level
matching), so it's perfect for tests and offline-first demos but **not semantic**.
For real meaning, swap in `transformersEmbedder` — same store/query code:

```ts
import { transformersEmbedder } from 'browservec';
// npm install @xenova/transformers  (loaded lazily; NOT in the core bundle)
const embedder = await transformersEmbedder({ model: 'Xenova/all-MiniLM-L6-v2' });
const db = await BrowserVec.create({ dimension: 384, metric: 'cosine', embedder });
```

The model weights download once and are cached by the browser, so later sessions
run offline. Any object with `{ dimension, embed(texts) }` works as an embedder.

> **Try it:** [`examples/05-text-retrieval.html`](../examples/05-text-retrieval.html) · [`examples/11-custom-embedder.html`](../examples/11-custom-embedder.html) · [`examples/12-transformers-embedder.html`](../examples/12-transformers-embedder.html)

## Worker ingest offload (NFR-8)

Quantized ingest is dominated by a pure-CPU transform — the randomized Hadamard
rotation (an FWHT per row) plus scalar quantization. On a large corpus that's
seconds of work, and run on the main thread it would freeze the tab. So for any
quantized store (`quantBits: 4 | 8`, flat **or** IVF) that work is handed to a Web
Worker; the main thread only does the light packing and the GPU upload and stays
responsive:

```ts
const db = await BrowserVec.create({ dimension: 768, metric: 'cosine', quantBits: 8 });
await db.addBatch(records);          // rotate+quantize runs in a Worker
db.stats().ingest;                   // 'worker' — or 'main-thread' if no Worker
```

The worker is **base64-inlined into the single-file bundle** (via Vite
`?worker&inline`), so there's no separate asset to host — importing `browservec`
is enough. The rotation is data-oblivious (seed-only), so the worker needs no
corpus state, and the exact same `BatchEncoder` runs in both the worker and the
in-thread fallback — the offloaded path can't drift from the synchronous one (a
Node parity test asserts they're bit-identical). Where `Worker` is unavailable
(older runtimes, strict CSP, SSR) it transparently falls back to encoding in-thread
and `stats().ingest` reports `'main-thread'`. The demo's **Worker ingest** button
ingests in chunks while a `requestAnimationFrame` heartbeat measures the worst
main-thread stall — smooth frames show the UI would stay interactive.

The **IVF k-means build** is offloaded the same way. The GPU already does the
per-iteration point *assignment*; the CPU-side centroid **mean-update** (an
O(trainCount·dim) sweep that jank-blocks the main thread at dim≈768) now runs in
its own inlined Worker, which owns a copy of the training sample so only the small
centroid array crosses back each Lloyd iteration. `stats().train` reports `'worker'`
or `'main-thread'`. Both paths call the same deterministic helpers on the same
inputs, so a build is byte-for-byte reproducible regardless of where the mean-update
ran (a Node check asserts the fresh-array trainer contract matches the old in-place
loop exactly). The **Worker ingest** button now also builds an IVF index under a
heartbeat to show the build stays interactive. Implementation notes:
[internals.md#worker-offload-seam](./internals.md#worker-offload-seam-srcquantencoderts-srcindexkmeanstrainerts).

> **Try it:** [`examples/09-worker-ingest.html`](../examples/09-worker-ingest.html)

## Corpus chunking (NFR-10)

A single GPU storage buffer is capped at `maxStorageBufferBindingSize` — **128 MiB
by default on many devices**, so a flat fp32×768 store overflows one buffer at only
~40k rows, and an int8×768 store (the memory-efficient path) at ~500k. The flat
index — **fp32 and quantized (int8/int4)** — now spreads the corpus across several
fixed-stride **chunks** (each ≤ the limit) and a query dispatches the distance
kernel once per chunk, each writing its scores into the right slice of one shared
scores buffer (the kernel takes a `params.y` base-row offset; for quant, codes are
addressed per-chunk while the small scales buffer stays global). Results are
identical to a single-buffer scan — chunking changes *where* rows live, not the math:

```ts
// Chunking is automatic once the corpus would overflow one buffer. The knob below
// just forces it early, to exercise the path without allocating gigabytes.
const db = await BrowserVec.create({ dimension: 768, metric: 'cosine', chunkRows: 25_000 });
await db.addBatch(records);
db.stats().chunks; // e.g. 4 — number of GPU buffers the corpus spans
```

Every chunk but the last holds exactly `rowsPerChunk` rows, so global row *r* maps
to chunk `⌊r / rowsPerChunk⌋` at local row `r % rowsPerChunk`; the current chunk
grows geometrically up to that size before a new one opens, so a small store still
uses a small buffer.

**IVF chunking (fp32 and ×quant).** The IVF paths are chunked too — the real path to
1M, since int8 codes for 1M×768 are ~1 GB (8× the 128 MiB default cap). Inverted
lists hold *global* row ids that span different corpus buffers, so a query buckets
the gathered candidates by chunk and issues one indexed dispatch per bucket, binding
that chunk's rows with **local** candidate ids (`params.z` = the bucket's offset in
the score output; the quant kernel also takes `params.y` = chunk base for the global
per-row scales). The final assignment during build is chunked the same way — the
assign kernel addresses rows locally, so each chunk scores its own rows and the
result is placed at the chunk's global base. Because k-means is deterministically
seeded and rotation is data-oblivious, a chunked IVF store produces results
*identical* to a single-buffer one — even under a *different* chunk layout, since the
clustering doesn't depend on where rows physically live. The demo's **Chunked corpus**
button forces a small chunk size and verifies fp32, int8, fp32×IVF, and int8×IVF all
match their single-buffer references exactly. Every query path now scales past one
GPU buffer.

> **Try it:** [`examples/10-chunking.html`](../examples/10-chunking.html)

## GPU top-k (§14.2 lever 3)

The distance kernel leaves one score per row in a dense buffer. Reading *all* of
it back and sorting on the CPU is an O(N) transfer plus an O(N·k) scan per query —
at 1M×768 that's a ~4 MB copy and a million-element pass on the main thread every
query. Past `GPU_TOPK_MIN_ROWS` (4096) the flat index reduces on the GPU instead:

```ts
// Transparent — no config. A large flat store just reduces on the GPU.
const db = await BrowserVec.create({ dimension: 768, metric: 'cosine' });
await db.addBatch(millionRecords);
await db.query(q, { k: 10 }); // top-k reduced on-GPU; only a short list read back

> **Try it:** [`examples/01-basic-flat.html`](../examples/01-basic-flat.html) (the GPU top-k path is exercised automatically at 50k+ rows)
```

One workgroup owns a contiguous 256-score segment and extracts its local top-k by
`k` rounds of parallel argmax in shared memory — each round tree-reduces to the
segment max, records it, then invalidates that slot with −FLT_MAX before the next
round. Each workgroup emits `k` (score, row) pairs, so the readback shrinks from
**N floats to ⌈N/256⌉·k pairs** (≈ a few KB), and the CPU only merges that short
list. `k` is a uniform, not a baked constant, so one compiled pipeline serves every
`k`. Scores are already dense-global (chunk offsets applied by the distance kernel),
so the reduction is oblivious to chunking. It's exact: the same neighbors a full
CPU sort returns — the demo's **GPU top-k** button checks it against a brute-force
reference across sizes straddling the switch-over threshold. Small corpora keep the
simpler full-readback path, where a second dispatch wouldn't pay for itself.

The **quantized flat path** (int8/int4) reuses the same reduction over its quantized
score buffer, so a large quantized store also skips the N-score readback before its
exact re-rank. The switch is gated by `GpuTopK.beneficial(n, k)`, which engages only
when the partials list (⌈N/256⌉·k pairs) is actually smaller than reading all N
scores — for int8's k·4 over-fetch it is; for int4's wider k·16 over-fetch it isn't,
so that case correctly keeps the full readback.

The **IVF and IVF×quant paths** use the same reduction over their *candidate* score
buffer (here `n` is the candidate count, not the corpus size), with the dense scan
slot remapped back to the original row on the CPU afterward — so at the 1M scale IVF
targets, a wide `nprobe` no longer pays an all-candidates readback per query. Every
query path now reduces on the GPU when it pays off.

## CPU fallback — no WebGPU (§NFR-7 / M6)

WebGPU is still absent on some browsers and locked-down/mobile environments, so
NFR-7 makes it a hard requirement that the library keep working there — same
results, lower throughput. Pass `fallback: 'wasm'` and, when WebGPU can't be
acquired, `create()` transparently uses an exact CPU brute-force scan instead of
throwing:

```ts
const db = await BrowserVec.create({ dimension: 768, metric: 'cosine', fallback: 'wasm' });
// ...where WebGPU exists this is the GPU path; where it doesn't, an exact CPU scan.
db.stats().device; // 'webgpu' or 'wasm' — tells you which path you got
```

The fallback ([src/fallback/cpu.ts](../src/fallback/cpu.ts)) mirrors the GPU flat
index's metric semantics exactly (dot for cosine/dot with vectors normalized on
insert; negative squared-L2 so "higher = closer"), so its top-k is **identical**
to the GPU path — it's literally the reference the GPU results are checked against.
It's a `VectorIndex` behind the same seam as the GPU indexes, so delete, `update`,
`compact`, persistence, and export/import all work on it unchanged; only the scan
kernel differs. Default `fallback: 'error'` still throws where WebGPU is missing,
so opting into CPU is explicit.

The hot loop is a hand-assembled **WASM-SIMD** kernel ([src/fallback/simd.ts](../src/fallback/simd.ts),
source [kernel.wat](../src/fallback/kernel.wat)): 526 bytes, `f32x4` four-lane FMA with a
scalar tail for `dim % 4`, ~**7× faster** than the unrolled scalar JS loop and matched
to f32 precision. The corpus lives *inside* the kernel's own linear memory, so a query
scores in place with no copy. Engines lacking WASM/SIMD transparently drop to the scalar
loop — identical results, just slower. The 526-byte module ships as a base64 constant, so
there's no build-time or runtime WASM toolchain and the single-file inlined dist is preserved.

Scope for now: the CPU path serves **fp32 flat only** — `quantBits`/`ann` are
GPU-throughput optimizations that add nothing to an exact CPU scan, so requesting
them without a GPU is a clear error rather than a silent accuracy change. The demo's
**CPU fallback** button flips an internal force-CPU seam to run the fallback in a
WebGPU browser and verifies it returns the same neighbors as the GPU on identical data.

> **Try it:** the demo's **CPU fallback** button (exercises the internal `__BROWSERVEC_FORCE_CPU__` seam)
