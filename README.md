# BrowserVec

[![npm](https://img.shields.io/npm/v/browservec)](https://www.npmjs.com/package/browservec)
[![license](https://img.shields.io/npm/l/browservec)](./LICENSE)

In-browser WebGPU vector store with custom WGSL kernels, for fast offline /
in-session retrieval — embeddings, similarity search, and persistence, all
client-side, no server round-trip.

**Contents:** [Status](#status) · [Install](#install) · [Quick start](#quick-start) ·
[Core API](#core-api) · [Features](#features) · [Benchmarks](#benchmarks) ·
[Docs](#docs--further-reading) · [Contributing](#contributing) · [License](#license)

## Status

M1–M5 and M7 complete, M6 mostly complete (encryption, CPU/WASM fallback done;
cross-device tuning in progress). In short: flat brute-force + two ANN families —
IVF clustering and an HNSW graph index (with an optional GPU beam-search kernel
and batched queries) — fp32/int8/int4/1-bit quantization (and every IVF×quant
combination), OPFS/IndexedDB persistence (HNSW graphs persist too, so loads skip
the rebuild) with optional AES-256-GCM encryption, an on-device text embedder,
Worker-offloaded ingest and index builds, and a WASM-SIMD CPU fallback for
devices without WebGPU. See [CHANGELOG.md](./CHANGELOG.md) for the release
history and [Not yet here](#not-yet-here) below for open milestone work.

## Install

```bash
npm install browservec
```

```ts
import { BrowserVec } from 'browservec';
```

Requires a browser with [WebGPU](https://caniuse.com/webgpu) for the GPU-accelerated
path; falls back to a WASM-SIMD/scalar CPU path (exact fp32 flat search, plus the
HNSW graph index) where WebGPU is unavailable — see
[CPU fallback](./docs/features.md#cpu-fallback--no-webgpu-nfr-7--m6).

## Quick start

```bash
npm install
npm run dev        # open the printed URL → demo/index.html (needs a WebGPU browser)
```

The demo builds a random corpus, runs a GPU top-k query, and checks recall
against a CPU brute-force reference — exercising the M1 exit criterion
(exact top-k correct vs. reference).

## Core API

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

const batches = await db.queryBatch([q1, q2, q3], { k: 5 });
// → QueryResult[][] — one GPU dispatch on an HNSW store with search: 'gpu'

db.get('a');     // → { id, vector, metadata? } | null
db.delete('a');  // tombstone by id → true/false (compacted on save)
await db.update({ id: 'a', vector: v2 }); // replace/upsert a vector
await db.compact();                       // physically drop tombstones (no reload)
db.stats();      // { count, deleted?, dimension, metric, device, lastQueryMs, persist? }
db.destroy();    // free GPU resources
```

Full method/type reference: [docs/api-reference.md](./docs/api-reference.md).

## Features

Each links to a short guide with runnable code:

| Feature | What it does |
|---|---|
| [Deleting vectors](./docs/features.md#deleting-vectors) | Tombstone-based delete/update/compact — cheap deletes, GPU memory reclaimed on compact or reload. |
| [Persistence](./docs/features.md#persistence-m2) | Versioned binary snapshots to OPFS (or IndexedDB), auto-load on `create()`, export/import as a `Blob`. |
| [Encryption at rest](./docs/features.md#encryption-at-rest-m6) | AES-256-GCM + PBKDF2 passphrase envelope for persisted/exported snapshots. |
| [Quantization (TurboQuant)](./docs/features.md#quantization--turboquant-int8-m3a) | int8/int4/1-bit codes via randomized Hadamard rotation + exact fp32 re-rank — ~4×/8×/32× less memory. |
| [Approximate search (IVF)](./docs/features.md#approximate-search--ivf-m4) | GPU-assisted k-means clustering; queries scan only the nearest `nprobe` clusters. Combines with quantization for the ~1M-row path. |
| [Graph search (HNSW)](./docs/features.md#graph-search--hnsw-m7) | Layered proximity graph with O(log N) beam search — incremental inserts (no rebuild), all metrics, works without WebGPU. Graph persists in snapshots, so loads skip the rebuild (~180×). |
| [GPU graph search](./docs/features.md#gpu-graph-search-m7b) | CAGRA-style WGSL kernel: the whole beam search in one dispatch, one workgroup per query — `queryBatch()` searches many queries concurrently (~4× the CPU walk at 60k×768×128). |
| [Text retrieval / embedder](./docs/features.md#text-retrieval--on-device-embedder-m5) | `addText`/`queryText` via a zero-dep hashing embedder or an optional real semantic model (transformers.js). |
| [Worker ingest offload](./docs/features.md#worker-ingest-offload-nfr-8) | Rotate+quantize and IVF k-means mean-updates run off the main thread so ingest doesn't freeze the UI. |
| [Corpus chunking](./docs/features.md#corpus-chunking-nfr-10) | Corpus spreads across multiple GPU buffers once it would exceed the device's per-buffer limit — transparent, same results. |
| [GPU top-k](./docs/features.md#gpu-top-k-142-lever-3) | Top-k reduction runs on the GPU past 4k rows, so only a short candidate list is read back per query. |
| [CPU fallback](./docs/features.md#cpu-fallback--no-webgpu-nfr-7--m6) | Exact WASM-SIMD flat scan when WebGPU is unavailable — same results, bit-identical to the GPU path. |

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

<details>
<summary><strong>Takeaways so far</strong> (see <a href="./docs/internals.md">docs/internals.md</a> for the kernel detail behind these)</summary>

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
  unavailable; exact fp32-flat search and the HNSW graph index (M7 — CPU-native,
  so it works here) run over IndexedDB persistence (no OPFS on iOS). Pass
  `fallback: 'wasm'` explicitly when targeting iOS Chrome or Safari, or
  `BrowserVec.create()` will throw.

Still missing from the matrix: Android Chrome (has WebGPU — a real gap),
Windows + NVIDIA/AMD (likely a much smaller buffer-size cap, which would
actually exercise chunking), and desktop Firefox/Safari. Contributions of a
device-report JSON block from any of these are welcome — see
[Contributing](#contributing).

</details>

## Not yet here

- **M6 (in progress)** — exact fp32-flat CPU fallback + WASM-SIMD kernel +
  encryption are done. Cross-browser/mobile tuning is underway via the demo's
  **M6 device report** button: a fixed-seed capability probe + full config
  matrix (fp32/int8/int4/1-bit × flat/IVF, recall + latency + memory, plus
  WebGPU adapter limits and OPFS/WASM-SIMD/Worker support) that emits one
  paste-back JSON block per device, feeding the [Benchmarks](#benchmarks)
  table above.

- **HNSW × quantization** — the graph index is fp32-only for now; combining it
  with the TurboQuant codes (quantized distances inside the traversal) is
  future work. IVF×quant remains the memory-constrained ~1M path.

Everything else (M1–M5, M7 graph search, plus M6's other pieces) is done — see
[CHANGELOG.md](./CHANGELOG.md) for what shipped in each release.

## Docs & further reading

- [docs/](./docs/) — architecture overview, full API reference, and
  per-subsystem internals (quantization codec, IVF/k-means, persistence
  format, Worker offload, CPU fallback) for contributors.
- [docs/architecture.md](./docs/architecture.md) — includes the file↔spec
  mapping table (which source file implements which `REQUIREMENTS.md`
  section).
- [REQUIREMENTS.md](./REQUIREMENTS.md) — the original design spec.
- [CHANGELOG.md](./CHANGELOG.md) — release history.

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
