# BrowserVec

[![npm](https://img.shields.io/npm/v/browservec)](https://www.npmjs.com/package/browservec)
[![license](https://img.shields.io/npm/l/browservec)](./LICENSE)

In-browser WebGPU vector store with custom WGSL kernels, for fast offline /
in-session retrieval. See [REQUIREMENTS.md](./REQUIREMENTS.md) for the full spec,
or [docs/](./docs/) for architecture notes, the full API reference, and
per-subsystem internals (quantization, IVF, persistence, Worker offload,
CPU fallback) aimed at contributors.

> **Status: M1 + M2 + M3 + M4 + M5** — WebGPU flat brute-force (cosine/dot/L2,
> configurable dim, CPU top-k), persistence to OPFS/IndexedDB with a versioned
> snapshot + export/import, **TurboQuant quantization at int8, int4, and 1-bit binary** (randomized
> Hadamard rotation + asymmetric dequant kernels + exact fp32 re-rank), an **IVF
> approximate index** + **IVF×quant combo** (int8/int4/1-bit; GPU-assisted k-means —
> scans only the nearest clusters, and fits ~1M×768 in one buffer), an **on-device text
> embedder** (zero-dep hashing + optional transformers.js), **Web Worker
> offload** (NFR-8 — the CPU-heavy rotate+quantize *and* the IVF k-means centroid
> update run off the main thread so ingest and index builds don't freeze the UI),
> **corpus chunking** (NFR-10 — every
> query path, flat fp32/int8/int4/1-bit **and IVF / IVF×int8/int4/1-bit**, spreads rows across
> multiple GPU buffers so they can exceed the per-buffer device limit — the practical
> path to 1M on default-limit devices), **snapshot encryption at rest** (M6 — AES-256-GCM
> with a PBKDF2 passphrase), and a **CPU/WASM-SIMD fallback** (M6) for devices without
> WebGPU. Cross-device recall/latency tuning (M6) is in progress — see
> [CHANGELOG.md](./CHANGELOG.md) for the full per-release breakdown.

## Install

```bash
npm install browservec
```

```ts
import { BrowserVec } from 'browservec';
```

Requires a browser with [WebGPU](https://caniuse.com/webgpu) for the GPU-accelerated
path; falls back to a WASM-SIMD/scalar CPU path (exact fp32 flat search) where WebGPU
is unavailable — see [CPU fallback](#cpu-fallback--no-webgpu-nfr-7--m6) below.

## Development

```bash
npm install
npm run dev        # open the printed URL → demo/index.html (needs a WebGPU browser)
```

The demo builds a random corpus, runs a GPU top-k query, and checks recall
against a CPU brute-force reference — exercising the M1 exit criterion
(exact top-k correct vs. reference).

## API (M1)

```ts
import { BrowserVec } from 'browservec';

BrowserVec.isSupported(); // { webgpu, opfs, wasm }

const db = await BrowserVec.create({ dimension: 768, metric: 'cosine' });

await db.addBatch([
  { id: 'a', vector: vecA, metadata: { lang: 'en' } },
  { id: 'b', vector: vecB },
]);

const hits = await db.query(queryVec, { k: 5 });
// → [{ id, score, metadata? }, ...]  (higher score = closer)

db.get('a');     // → { id, vector, metadata? } | null
db.delete('a');  // tombstone by id → true/false (compacted on save)
await db.update({ id: 'a', vector: v2 }); // replace/upsert a vector
await db.compact();                       // physically drop tombstones (no reload)
db.stats();      // { count, deleted?, dimension, metric, device, lastQueryMs, persist? }
db.destroy();    // free GPU resources
```

### Deleting vectors

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

### Persistence (M2)

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
mismatch on load throws rather than silently corrupting results.

### Encryption at rest (M6)

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

### Quantization — TurboQuant int8 (M3a)

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
See [REQUIREMENTS.md §6](./REQUIREMENTS.md) for the TurboQuant background and IP note.

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
below the bandwidth crossover, fp32 or int8 is the faster scan.

`metric: 'cosine'` normalizes vectors on insert and the query, so the kernel
computes a plain dot product (cosine = dot of unit vectors). `'l2'` returns
negative squared distance so that **higher score = closer** holds for every
metric and a single top-k path works.

### Approximate search — IVF (M4)

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

### Text retrieval — on-device embedder (M5)

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

### Worker ingest offload (NFR-8)

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
heartbeat to show the build stays interactive.

### Corpus chunking (NFR-10)

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

### GPU top-k (§14.2 lever 3)

The distance kernel leaves one score per row in a dense buffer. Reading *all* of
it back and sorting on the CPU is an O(N) transfer plus an O(N·k) scan per query —
at 1M×768 that's a ~4 MB copy and a million-element pass on the main thread every
query. Past `GPU_TOPK_MIN_ROWS` (4096) the flat index reduces on the GPU instead:

```ts
// Transparent — no config. A large flat store just reduces on the GPU.
const db = await BrowserVec.create({ dimension: 768, metric: 'cosine' });
await db.addBatch(millionRecords);
await db.query(q, { k: 10 }); // top-k reduced on-GPU; only a short list read back
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

### CPU fallback — no WebGPU (§NFR-7 / M6)

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

The fallback ([src/fallback/cpu.ts](src/fallback/cpu.ts)) mirrors the GPU flat
index's metric semantics exactly (dot for cosine/dot with vectors normalized on
insert; negative squared-L2 so "higher = closer"), so its top-k is **identical**
to the GPU path — it's literally the reference the GPU results are checked against.
It's a `VectorIndex` behind the same seam as the GPU indexes, so delete, `update`,
`compact`, persistence, and export/import all work on it unchanged; only the scan
kernel differs. Default `fallback: 'error'` still throws where WebGPU is missing,
so opting into CPU is explicit.

The hot loop is a hand-assembled **WASM-SIMD** kernel ([src/fallback/simd.ts](src/fallback/simd.ts),
source [kernel.wat](src/fallback/kernel.wat)): 526 bytes, `f32x4` four-lane FMA with a
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

## Benchmarks

Numbers below are from the demo's **M6 device report** tool (fixed-seed 20k×384
corpus, `recall@10` against an exact fp32 reference). This is a small, growing
device matrix, not an exhaustive one — generate your own with the same tool
(`npm run dev` → demo → **M6 device report**) or the interactive
[perf-benchmark example](./examples/perf-benchmark.html), which sweeps corpus
size/dimension/index type directly in the browser.

**Chrome 149 / macOS, Apple M-series (Metal-3)** — WebGPU, OPFS, WASM-SIMD, and
Worker offload all available; `maxStorageBufferBindingSize` = 4 GiB.

| Config | recall@10 | Query latency |
|---|---|---|
| flat fp32 | 1.000 | 1.71 ms/q |
| flat int8 | 1.000 | 1.52 ms/q |
| flat int4 | 1.000 | 1.69 ms/q |
| flat 1-bit | 1.000 | 2.45 ms/q |
| IVF fp32 | 1.000 | 0.51 ms/q |
| IVF int8 | 1.000 | 0.63 ms/q |
| IVF int4 | 1.000 | 0.93 ms/q |
| IVF 1-bit | 1.000 | 1.11 ms/q |
| CPU fallback (WASM-SIMD), 8k rows | — | 1.76 ms/q |

**Chrome (CriOS 149) / iPhone 15, iOS 26.5** — no WebGPU (iOS third-party
browsers are WebKit under the hood, so WebGPU isn't exposed), no OPFS
(IndexedDB is used instead), WASM-SIMD and Worker both available.

| Config | Query latency |
|---|---|
| CPU fallback (WASM-SIMD), 8k rows | 0.40 ms/q |

Takeaways so far (see [docs/internals.md](./docs/internals.md) for the kernel
detail behind these):

- **Sub-byte quantization is a memory lever, not a speed lever at this scale.**
  The quantized kernels are ALU-bound (manual nibble/sign unpack costs more than
  a plain fp32 `vec4` load), so query time actually *rises* as bit-width shrinks
  at 20k rows (`fp32 ≈ int8 < int4 < 1-bit`). The payoff is memory (int8 ~4×,
  int4 ~8×, 1-bit ~32× smaller) and, at much larger corpora, bandwidth — tighter
  codes only start winning on throughput once the scan is bandwidth-bound rather
  than ALU-bound.
- **`maxStorageBufferBindingSize` varies a lot by GPU** — 4 GiB on Apple Metal
  vs. a much more conservative default on many other adapters. Corpus chunking
  (§NFR-10) triggers off the device's actual reported limit, so this needs no
  configuration — but where the chunking crossover happens is device-dependent.
- **iOS is CPU-fallback-only today**: no WebGPU means quantization/IVF are
  unavailable and only exact fp32-flat search runs, over IndexedDB persistence
  (no OPFS on iOS). Pass `fallback: 'wasm'` explicitly when targeting iOS Chrome
  or Safari, or `BrowserVec.create()` will throw.

Still missing from the matrix: Android Chrome (has WebGPU — a real gap),
Windows + NVIDIA/AMD (likely a much smaller buffer-size cap, which would
actually exercise chunking), and desktop Firefox/Safari. Contributions of a
device-report JSON block from any of these are welcome — see
[Contributing](#contributing).

## How it maps to the design

| Code | REQUIREMENTS.md |
|------|-----------------|
| [src/engine/wgsl/distance.ts](src/engine/wgsl/distance.ts) | §14.2 levers 4 & 5 — `DIM`/`WG` baked as constants → unrolled `vec4` FMA; query staged in workgroup shared memory |
| [src/index/flat.ts](src/index/flat.ts) | §9 M1 — flat GPU brute-force; GPU top-k past 4k rows, CPU top-k below |
| [src/engine/wgsl/topk.ts](src/engine/wgsl/topk.ts) / [src/index/gpuTopk.ts](src/index/gpuTopk.ts) | §14.2 lever 3 — on-GPU top-k reduction (segment argmax → short readback), on the flat fp32 + quantized int8/int4 paths |
| [src/engine/buffers.ts](src/engine/buffers.ts) | §NFR-10 — `ChunkedCorpus`: corpus split across GPU buffers past the per-buffer limit |
| [src/engine/device.ts](src/engine/device.ts) | §NFR-11 — device acquisition + device-loss wiring |
| [src/persist/format.ts](src/persist/format.ts) | §FR-16 — versioned binary snapshot codec |
| [src/persist/opfs.ts](src/persist/opfs.ts) / [indexeddb.ts](src/persist/indexeddb.ts) | §NFR-6 — OPFS primary, IndexedDB fallback |
| [src/persist/crypto.ts](src/persist/crypto.ts) | §M6 — AES-256-GCM + PBKDF2 snapshot encryption at rest |
| [src/fallback/cpu.ts](src/fallback/cpu.ts) / [simd.ts](src/fallback/simd.ts) | §NFR-7 / M6 — exact CPU flat scan for the no-WebGPU fallback (identical results); WASM-SIMD `f32x4` kernel (~7×), scalar JS fallback |
| [src/store/store.ts](src/store/store.ts) | §NFR-11 — CPU raw vectors are the persistence source of truth |
| [src/quant/rotator.ts](src/quant/rotator.ts) | §6 — data-oblivious randomized Hadamard rotation |
| [src/quant/codec.ts](src/quant/codec.ts) | §6 / §9 M3b — int8 snorm codec (matches `unpack4x8snorm`) + int4 nibble + 1-bit sign pack/unpack |
| [src/engine/wgsl/distanceQ8.ts](src/engine/wgsl/distanceQ8.ts) | §6.2 FR-Q4 / §14.2 lever 1 — int8 kernel, dequant in-shader |
| [src/engine/wgsl/distanceQ4.ts](src/engine/wgsl/distanceQ4.ts) | §9 M3b — int4 kernel, manual nibble unpack (~8× less memory) |
| [src/engine/wgsl/distanceQ1.ts](src/engine/wgsl/distanceQ1.ts) | §9 M3b — 1-bit binary kernel, sign-bit unpack + asymmetric scoring (~32× less memory) |
| [src/index/quant.ts](src/index/quant.ts) | §6 — quantized index + asymmetric query |
| [src/index/kmeans.ts](src/index/kmeans.ts) | §9 M4 — k-means (CPU helpers + init/update for the GPU-assisted loop) |
| [src/index/kmeans.worker.ts](src/index/kmeans.worker.ts) | §NFR-8 — k-means mean-update Worker; inlined via `?worker&inline` |
| [src/index/kmeansTrainer.ts](src/index/kmeansTrainer.ts) | §NFR-8 — trainer seam: Worker offload + in-thread fallback (reproducible either way) |
| [src/engine/wgsl/assign.ts](src/engine/wgsl/assign.ts) | §9 M4 — GPU centroid-assignment kernel (argmax dot) |
| [src/engine/wgsl/distance.ts](src/engine/wgsl/distance.ts) (indexed) | §9 M4 — indexed scan: score rows from a candidate-id list |
| [src/index/ivf.ts](src/index/ivf.ts) | §9 M4 / §NFR-10 — IVF index: reservoir sample, GPU-assisted build, probe→gather→chunked scan→GPU top-k |
| [src/engine/wgsl/assignQ8.ts](src/engine/wgsl/assignQ8.ts) | §9 M4 — quantized centroid-assignment kernel (dequant + argmax dot) |
| [src/engine/wgsl/assignQ4.ts](src/engine/wgsl/assignQ4.ts) | §9 M3b+M4 — 4-bit quantized centroid-assignment kernel |
| [src/engine/wgsl/assignQ1.ts](src/engine/wgsl/assignQ1.ts) | §9 M3b+M4 — 1-bit binary centroid-assignment kernel (IVF×1-bit combo) |
| [src/index/ivfquant.ts](src/index/ivfquant.ts) | §9 M4 / §NFR-10 — IVF × int8/int4 combo: clustered + quantized + chunked + GPU top-k, the 1M path |
| [src/embed/hashing.ts](src/embed/hashing.ts) | §9 M5 — zero-dep feature-hashing embedder (offline, lexical) |
| [src/embed/transformers.ts](src/embed/transformers.ts) | §9 M5 — optional transformers.js adapter (lazy, real semantics) |
| [src/quant/encode.ts](src/quant/encode.ts) | §NFR-8 — shared BatchEncoder (rotate+quantize), one impl for both threads |
| [src/quant/quantize.worker.ts](src/quant/quantize.worker.ts) | §NFR-8 — ingest Worker; inlined into the bundle via `?worker&inline` |
| [src/quant/encoder.ts](src/quant/encoder.ts) | §NFR-8 — encoder seam: Worker offload + in-thread fallback |

## Not yet here (by milestone)

- **M3b** ✅ done — int8, int4, and 1-bit binary, each as flat **and** IVF×quant; binary is ~32× less memory (~128 B/row)
- **NFR-10** ✅ done — GPU top-k on all query paths + corpus chunking on all paths (flat fp32/int8/int4, IVF, IVF×quant)
- **NFR-8** ✅ done — Worker offload for both quantized ingest (rotate+quantize) and the IVF k-means centroid mean-update; transparent in-thread fallback, reproducible either way
- **M6 (in progress)** exact fp32-flat CPU fallback + WASM-SIMD kernel + encryption done. Cross-browser/mobile tuning is underway via the demo's **M6 device report** button: a fixed-seed capability probe + full config matrix (fp32/int8/int4/1-bit × flat/IVF, recall + latency + memory, plus WebGPU adapter limits and OPFS/WASM-SIMD/Worker support) that emits one paste-back JSON block per device, so a device matrix can be assembled from real hardware and fed back into tuning.

## Contributing

Issues and PRs are welcome. Before opening a PR, run:

```bash
npm run typecheck
npm run build
```

There's no automated test suite yet — the demo (`npm run dev`) exercises the
GPU vs. CPU-reference recall check described above; changes touching kernels
or indexes should be verified there before submitting.

## License

[MIT](./LICENSE) © Sharma SK
