# Internals

Subsystem-level notes for contributors touching kernels, the persistence
format, or the Worker-offload seams. For the high-level flow, read
[architecture.md](./architecture.md) first.

## Device acquisition (`src/engine/device.ts`)

- `DeviceContext { adapter, device, limits, lost }` — `lost` flips `true`
  asynchronously when `device.lost` resolves (`wireDeviceLost`); a
  `'destroyed'` reason is an intentional teardown (silent), anything else
  logs a console warning.
- `acquireDevice(existing?)`: an injected `GPUDevice` is wrapped directly
  (its `adapter` is left `undefined`); otherwise requests an adapter with
  `powerPreference: 'high-performance'` and a device with `requiredLimits`
  from `pickLimits`.
- `pickLimits` asks for `maxStorageBufferBindingSize`/`maxBufferSize` = 1 GiB
  (`1 << 30`), `maxComputeWorkgroupStorageSize` = 16384,
  `maxComputeInvocationsPerWorkgroup` = 256 — each clamped to what the
  adapter actually reports.
- `WebGPUUnavailableError` is thrown when `navigator.gpu` is missing or no
  adapter is returned; this is the error type `fallback: 'wasm'` catches in
  `src/index.ts`.

## Store (`src/store/store.ts`)

- Two parallel maps: `byId: Map<string, RowEntry>` and `byRow: RowEntry[]`,
  plus `deletedRows: Set<number>` for tombstones and a packed
  `raw: Float32Array` (post-normalization vectors — the persistence source
  of truth).
- `insert()` throws on duplicate id; `raw` grows geometrically (1.5×, floor
  1024 rows) via `ensureRawCapacity`.
- `prepare(vector)` validates dimension, then `normalizeInPlace` (L2
  normalize) if the store's `normalize` flag is set.
- `liveEntries()`/`liveVectors()` filter out tombstoned rows — this is what
  `compact()` and snapshot serialization use; `rawView()` (all rows,
  including tombstones) is what a fresh `VectorIndex.append()` gets fed on
  load.

## Persistence format (`src/persist/format.ts`)

Magic `0x43455642` ("BVEC" LE), `FORMAT_VERSION = 2`, `HEADER_BYTES = 32`.

| Offset | Field |
|---|---|
| `0..4` | magic |
| `4..8` | version (`1`, or `2` when a graph section is present) |
| `8..12` | dimension |
| `12..16` | metric code (`0`=cosine, `1`=dot, `2`=l2) |
| `16..20` | flags (bit 0 = normalize) |
| `20..24` | count |
| `24..28` | metadata JSON byte length |
| `28..32` | v1: reserved (0). v2: byte offset of the HNSW graph section (0 = none) |
| `32..` | metadata JSON (`Array<{id, metadata?}>`, row order), 4-byte padded |
| after that | row-major `Float32Array` vectors (`count * dim * 4` bytes) |

**v2 graph section (M7c)** — starts 4-aligned right after the vectors, and holds
the HNSW graph so loads restore it instead of rebuilding:

| Field | Size |
|---|---|
| header: magic `"HNSW"` (`0x57534e48`), graph version (=1), `M`, `entry`, `top` (maxLevel), `upperLen`, 2× reserved | 8 × u32 |
| `levels` — per-node top layer | i32 × count |
| `links0` — layer-0 adjacency, count-prefixed `(2M+1)`-wide blocks | i32 × count·(2M+1) |
| `upper` — upper-layer blocks, concatenated in row order (node *n* owns `levels[n]·(M+1)`) | i32 × upperLen |

`serialize()` writes **v1 whenever there is no graph** (flat/IVF/quant stores),
so older builds keep reading every snapshot they could before — only the new
feature pays the version bump; `deserialize()` reads 1..2. The graph is skipped
at write time when tombstones are pending (compaction renumbers rows), and
ignored at read time on any config mismatch — both fall back to the ordinary
rebuild-via-append load path. `deserialize()` validates magic/version/count
consistency and truncation, and returns a `Snapshot`. The vector and graph
regions are `.slice()`d out on read so they don't pin the whole input buffer
in memory.

## Encryption envelope (`src/persist/crypto.ts`)

Magic `0x45435642` ("BVCE" LE), `CRYPTO_VERSION = 1`, `HEADER_BYTES = 44`,
`PBKDF2_ITERATIONS = 210_000`.

| Offset | Field |
|---|---|
| `0..4` | magic |
| `4..8` | version |
| `8..12` | PBKDF2 iteration count |
| `12..16` | reserved |
| `16..32` | 16-byte salt |
| `32..44` | 12-byte GCM IV |
| `44..` | ciphertext (WebCrypto appends its own 16-byte GCM auth tag) |

Key derivation is PBKDF2-SHA256 → a non-extractable AES-256-GCM `CryptoKey`.
`decryptSnapshot` catches any GCM auth failure (wrong passphrase or
tampering) and rethrows a generic "decryption failed" error — no oracle that
would let an attacker distinguish "wrong key" from "corrupted data".
`isEncrypted(buf)` is a cheap 4-byte magic check used by `src/index.ts` to
branch between plaintext and encrypted snapshot loading.

## Persistence backend selection (`src/persist/backend.ts`)

`PersistenceBackend { kind: 'opfs' | 'indexeddb', write, read, remove }`.
`selectBackend('opfs' | 'indexeddb')` throws if that specific backend isn't
available; `selectBackend('auto')` prefers OPFS (faster, larger quota) and
falls back to IndexedDB, throwing only if neither is available.

## Flat index (`src/index/flat.ts`)

- The shared `VectorIndex` interface (`size`, `append`, `query`, `destroy`)
  is implemented by every index type, including `CpuIndex`.
- `FlatIndex` owns a `ChunkedCorpus` (GPU buffers split once a single buffer
  would exceed `maxStorageBufferBindingSize`) and a compute pipeline
  specialized per `(dim, metric)` at build time (`WORKGROUP_SIZE = 64`).
- `query()` writes the query vector once, dispatches the distance kernel per
  corpus chunk (each chunk's scores land at global slot `baseRow + gid.x`),
  then either uses `GpuTopK` (GPU-side reduction, gated by
  `GpuTopK.beneficial(rows, k)`) or reads back all scores and runs a CPU
  `topK()` (O(n·k) partial selection).

## Quantization (`src/quant/`)

**Rotation** (`rotator.ts`) implements TurboQuant's randomized Hadamard
rotation (data-oblivious mixing so a per-coordinate scalar quantizer works
well on arbitrary embedding distributions — see arXiv:2504.19874):
`padToPow2` rounds the dimension up to a power of two ≥ 4; `fwht` is an
in-place fast Walsh-Hadamard transform; `Rotator(dim, seed, rounds=2)`
precomputes `rounds` independent `±1` sign vectors (each seeded
`seed ^ (0x9e3779b9 * (r+1))`) and applies sign-flip → FWHT → scale-by-
`1/sqrt(paddedDim)` per round. The rotation is orthonormal, so dot products
are preserved exactly (`⟨Rx,Ry⟩ = ⟨x,y⟩`) — both corpus rows and the query
are rotated identically.

**Codec** (`codec.ts`) — all three widths share `QuantizedRow { words:
Uint32Array, scale: number }`:

- **int8**: per-row scale = `max(|coord|)`, each coordinate maps to a signed
  byte in `[-127, 127]` matching WGSL's `pack4x8snorm`/`unpack4x8snorm`
  semantics exactly, 4 bytes packed per `u32`.
- **int4**: signed nibble in `[-7, 7]`, 8 coordinates packed per `u32`
  (`paddedDim` must be a multiple of 8) — the kernel unpacks nibbles
  manually since there's no hardware unpack for this width.
- **1-bit**: only the sign per coordinate survives, 32 coords/`u32`
  (LSB-first, bit set when coord ≥ 0), `paddedDim` must be a multiple of 32.
  Scale is `mean(|coord|)` (not max — this minimizes reconstruction error
  for a binary code) and scoring is asymmetric (fp32 query × ±1 corpus).

## IVF (`src/index/ivf.ts`)

`IVFParams`: `nlist` (default ≈ `sqrt(rows)`, clamped `[16, 4096]`), `nprobe`
(default ≈ 5% of `nlist`, minimum 1), `sampleSize` (default 50,000 —
reservoir cap), `trainSize` (default 16,384 fed into k-means), `iters`
(default 8 Lloyd iterations).

- Cosine/dot only — throws for `l2`.
- Ingest updates a streaming reservoir sample (classic reservoir-sampling
  algorithm) used later to train k-means; the index is marked stale and
  rebuilt lazily on the next `query()` if rows changed since the last build.
- `build()` runs Lloyd's algorithm via `createKMeansTrainer` (GPU does
  per-point nearest-centroid assignment, CPU/Worker does the mean-update —
  see below), then a final GPU assignment pass over every corpus row, and
  builds CSR-style inverted lists: `listOffset: Int32Array(nlist+1)` (prefix
  sums) + `listRows: Int32Array(rows)` (row ids grouped by cluster).
- `query()` computes the `nprobe` nearest clusters to the query on the CPU
  (`pickProbes`), gathers candidate row ids (`gatherCandidates`), buckets
  them by corpus chunk, dispatches the indexed scan kernel per chunk
  (`dispatchScan`), then top-k with a remap back to original row ids.

`src/index/ivfquant.ts` is the same structure with quantized storage: the
corpus is `wordsPerRow`-wide `u32` codes (via `QuantEncoder`, any of
1/4/8-bit) plus a separate global `scales` buffer; k-means training happens
in rotated fp32 space (via the fp32 training sample), but the final per-row
cluster assignment uses a bit-width-specific quantized assign kernel
(`assignQ8`/`assignQ4`/`assignQ1`); queries are rotated
(`encoder.rotateQuery`) before probing/scanning, and the scan kernel is also
bit-width-specific.

## HNSW (`src/index/hnswGraph.ts`, `hnsw.ts`, `hnsw.worker.ts`, `hnswGpu.ts`)

The graph core (`hnswGraph.ts`) is allocation-light by design: adjacency lives
in flat typed arrays (layer 0 is one `Int32Array` of count-prefixed `(2M+1)`-wide
blocks; upper layers get a per-node `Int32Array` only for the ~1/M of nodes that
have them), the visited set is a generation-stamped `Int32Array` (bump a counter
instead of clearing), and the two beam heaps are reused scratch (`MinHeap` over
parallel dist/id arrays; the results heap stores negated distances to act as a
max-heap). Neighbor selection uses the paper's diversity heuristic — keep a
candidate only if it's closer to the query than to every already-kept neighbor —
which is what keeps the graph navigable *between* clusters. Distances share the
house unrolled-×4 loops; internally smaller = closer (negated dot, or squared
L2), so `score = -dist` matches every metric's public convention.

The seam (`hnsw.ts`) mirrors `kmeansTrainer.ts`, with one difference: the Worker
**owns** the graph and a packed corpus copy. Appends transfer vectors in;
searches send back only the top-k arrays — so both build and CPU search run
off the main thread. Builds are seeded-deterministic on either path.

The GPU path (`graphSearch.ts` + `hnswGpu.ts`) is CAGRA-style: no hierarchy on
the GPU — the kernel walks the flat layer-0 graph (fixed `2M` slots per row,
`0xFFFFFFFF`-padded) seeded from the entry node plus evenly-spread rows. One
workgroup per query keeps the whole beam state in ~12 KB of workgroup shared
memory: a 256-slot candidate beam (dist/id/expanded), a 2048-slot open-addressed
atomic visited hash (id+1, 0 = empty; sheds under pressure rather than stalling —
the CPU dedups the readback), and reduction scratch. Loop-control reads go
through `workgroupUniformLoad` so barriers stay in uniform control flow.
Termination is exhaustion: expansion clears flags and inserts only add
unexpanded slots when they beat the current worst, so a converged beam ends up
fully expanded (the classic "best unexpanded > worst kept" check is a no-op in
a fused beam — the best unexpanded *is* a kept entry). The kernel emits the raw
beam; the CPU dedups, sorts, and takes k. Batching is free: `queryBatch` packs
queries row-major and dispatches `nQ` workgroups in one submit.

Persistence (M7c): `serializeGraph()`/`loadGraph()` snapshot and restore the
structure (see [persistence format](#persistence-format-srcpersistformatts));
`loadGraph` fast-forwards the level RNG one draw per restored row so post-load
inserts draw the same levels a never-saved store would — save/load doesn't fork
determinism.

## Worker-offload seam (`src/quant/encoder.ts`, `src/index/kmeansTrainer.ts`)

Both follow the identical pattern: wrap a synchronous in-thread
implementation, lazily boot a Vite-inlined Worker
(`?worker&inline`) on first use, and fall back to the in-thread
implementation — byte-identical output either way — if `typeof Worker ===
'undefined'` or Worker construction throws. `mode()` reports which path is
active (`'worker' | 'main-thread' | 'pending'`), surfaced to callers via
`Stats.ingest`/`Stats.train`.

- `QuantEncoder`: `encode(data, count, wantRotated)`, plus `rotateQuery(q)`
  which *always* runs in-thread (it's on the query's critical path, not
  worth a postMessage round trip).
- `KMeansTrainer`: `init()` (random centroid init), `update(assign, iter)`
  (one Lloyd mean-update).
- Data crossing the Worker boundary is `.slice()`d to a tight owned copy and
  transferred zero-copy via `postMessage(..., [buffer])` — avoids
  structured-cloning large arrays and leaves the caller's data untouched.

## CPU fallback (`src/fallback/cpu.ts`, `src/fallback/simd.ts`)

`CpuIndex` is a brute-force scan that mirrors `FlatIndex` exactly: dot
product for cosine/dot (vectors are pre-normalized on insert, same as the
GPU path) and negative squared-L2 for `l2` ("higher = closer" convention
preserved). Two execution tiers:

1. **WASM-SIMD** (`simd.ts`): a hand-assembled 526-byte WASM module
   (base64-embedded; source at `src/fallback/kernel.wat`) exporting
   `dot`/`l2` functions that run `f32x4`/`v128` SIMD over the module's own
   linear memory — 4 lanes per iteration with a scalar tail for `dim % 4`.
   The corpus lives *inside* WASM linear memory, so queries score in place
   with no per-query copy. Memory grows geometrically in 64 KiB pages
   (`mem.grow`); the buffer view is re-taken after growth since growing
   detaches the previous view. `simdAvailable()` detects support via
   `WebAssembly.validate()` (the module uses `v128`, which is invalid on
   non-SIMD engines) and caches the result module-wide.
2. **Scalar JS** fallback when SIMD is unavailable: an unrolled (×4) plain
   `Float32Array` loop — identical results, ~7× slower.

`quantBits`/`ann` are rejected under the CPU fallback (`buildCpuIndex` in
`src/index.ts`) rather than silently degraded — they're GPU-throughput
optimizations that don't help an exact CPU scan.
