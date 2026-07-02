# Development guide

How to set up, build, and extend BrowserVec.

## Setup

```bash
git clone <repo>
cd browservec
npm install
```

## Dev workflow

```bash
npm run dev
```

Opens Vite dev server with the demo at `/demo/`. The demo exercises every
index type and config combination with an interactive report — this is the
primary development verification tool.

Source changes are hot-reloaded (Vite serves TS directly).

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Start dev server, open demo |
| `npm run build` | Build the library dist + declaration files |
| `npm run typecheck` | Run TypeScript type-checking (`tsc --noEmit`) |
| `npm run preview` | Preview the production build locally |

Before committing, run:

```bash
npm run typecheck
npm run build
```

## Project layout

```
src/
├── index.ts              # Public entry: BrowserVec class, create/query/delete/save
├── types.ts              # All public types (BrowserVecConfig, Stats, QueryResult, etc.)
├── store/
│   └── store.ts          # CPU-side id↔row map, raw vectors (persistence source of truth)
├── index/                # VectorIndex implementations (the GPU compute layer)
│   ├── flat.ts           #   FlatIndex — fp32 brute-force GPU scan
│   ├── quant.ts          #   QuantIndex — quantized (int8/int4/1-bit) flat scan
│   ├── ivf.ts            #   IVFIndex — fp32 clustered (approximate)
│   ├── ivfquant.ts       #   IVFQuantIndex — clustered + quantized (the 1M path)
│   ├── gpuTopk.ts        #   On-GPU top-k reduction
│   ├── kmeans.ts         #   CPU helpers for k-means clustering
│   ├── kmeansTrainer.ts  #   K-means trainer (Worker offload seam)
│   └── kmeans.worker.ts  #   Inlined Worker for mean-updates
├── engine/               # WebGPU device management & WGSL kernel templates
│   ├── device.ts         #   Device acquisition, adapter probing, limits
│   ├── buffers.ts        #   ChunkedCorpus — multi-buffer corpus spanning
│   ├── profile.ts        #   GPU query timing (queryTrace)
│   └── wgsl/             #   WGSL shader source generators
│       ├── distance.ts   #     fp32 distance kernel (cosine/dot/l2)
│       ├── distanceQ8.ts #     int8 quantized distance kernel
│       ├── distanceQ4.ts #     int4 quantized distance kernel
│       ├── distanceQ1.ts #     1-bit binary distance kernel
│       ├── topk.ts       #     On-GPU top-k reduction kernel
│       ├── assign.ts     #     fp32 centroid-assignment (IVF build)
│       ├── assignQ8.ts   #     int8 centroid-assignment
│       ├── assignQ4.ts   #     int4 centroid-assignment
│       └── assignQ1.ts   #     1-bit centroid-assignment
├── quant/                # Quantization (TurboQuant)
│   ├── rotator.ts        #   Randomized Hadamard rotation
│   ├── codec.ts          #   int8/int4/1-bit pack/unpack codec
│   ├── encoder.ts        #   QuantEncoder (Worker offload seam)
│   ├── encode.ts         #   BatchEncoder — shared impl (worker + main-thread)
│   └── quantize.worker.ts#   Inlined Worker for rotate+quantize
├── persist/              # Persistence backend
│   ├── format.ts         #   Binary snapshot format (BVEC magic, header, vectors)
│   ├── crypto.ts         #   AES-256-GCM + PBKDF2 encryption envelope
│   ├── backend.ts        #   Backend selection (OPFS / IndexedDB)
│   ├── opfs.ts           #   OPFS storage backend
│   └── indexeddb.ts      #   IndexedDB storage backend
├── fallback/             # CPU fallback (no WebGPU)
│   ├── cpu.ts            #   CpuIndex — exact flat scan
│   ├── simd.ts           #   WASM-SIMD kernel (526-byte module, base64-embedded)
│   └── kernel.wat        #   WAT source for the WASM module
├── embed/                # On-device text embedders
│   ├── hashing.ts        #   Zero-dep feature-hashing embedder
│   └── transformers.ts   #   Transformers.js adapter (lazy-loaded)
└── env.d.ts              # Ambient declarations for Vite worker inlining
```

## Architecture overview

The `VectorIndex` interface (`src/index/flat.ts`) is the central abstraction:

```ts
interface VectorIndex {
  readonly size: number;
  append(data: Float32Array, count: number): void | Promise<void>;
  query(queryVec: Float32Array, k: number): Promise<FlatHit[]>;
  destroy(): void;
}
```

`BrowserVec` (orchestration layer) delegates all GPU work to a
`VectorIndex`. Which concrete type is used depends on the config — see
[architecture.md](../architecture.md#startup-which-index-gets-built) for the
selection table.

## How to add a new index type

1. Create `src/index/myindex.ts` implementing the `VectorIndex` interface.
2. If it needs a custom WGSL kernel, add it to `src/engine/wgsl/`.
3. Wire it into the factory in `src/index.ts` (`buildIndex()` function).
4. Add any new config fields to `BrowserVecConfig` in `src/types.ts`.
5. Add the corresponding stats fields to `Stats` in `src/types.ts` and
   populate them in `BrowserVec.stats()` (`src/index.ts`).
6. Wire the new type into the `buildIndex()` decision table.
7. Update the decision table in `docs/architecture.md`.
8. Add example usage in `examples/`.
9. Verify with the demo (`npm run dev`).

## How to add a new WGSL kernel

1. Create the shader template in `src/engine/wgsl/` (plain TS function that
   returns a WGSL shader string, parameterized by dim/metric/workgroup size).
2. The kernel is compiled at runtime via `device.createComputePipeline()` with
   `constants` for the dimension-specific unrolling.
3. Import and use the shader in the appropriate index implementation.
4. Verify with the demo's correctness check (GPU vs CPU reference).

## Worker offload seam

Two operations use Worker offload to keep the main thread responsive:

- **Quantized ingest** (`src/quant/encoder.ts` + `src/quant/quantize.worker.ts`)
- **IVF k-means mean-updates** (`src/index/kmeansTrainer.ts` + `src/index/kmeans.worker.ts`)

Both follow the same pattern:
- A seam wrapper lazily boots a Vite-inlined Worker (`?worker&inline`).
- If Workers are unavailable, the operation runs in-thread transparently.
- Results are byte-identical regardless of path.
- `Stats.ingest` / `Stats.train` report which path ran.

The Workers are base64-inlined by Vite into the single-file dist — there are
no separate assets to host.

## Testing

There is no automated test suite yet. Verification is done through:

- **The demo** (`npm run dev` → open demo) — exercises every index type,
  checks GPU results against a CPU brute-force reference, and reports recall.
- **Examples** in `examples/` — each is a self-contained HTML page testing a
  specific feature.
- **Node parity checks** in the dev workflow — the project currently relies
  on manual verification via the demo's M6 device report tool.

When adding or modifying kernels, always verify:
1. GPU results match CPU reference (demo recall check).
2. Quantized results match fp32 on identical data (demo quantization check).
3. The CPU fallback returns bit-identical results to the GPU path.

## Code conventions

- **No JSDoc comments** except for public API surface (`src/index.ts`,
  `src/types.ts`). Internal code uses minimal inline comments.
- **Metric convention:** higher score = closer for all metrics (cosine/dot
  are raw dot product, l2 is negative squared distance).
- **Imports:** always with `.js` extension (TypeScript ESM convention).
- **Formatting:** no formatter dependency — follow the style of surrounding code.

## Requirements mapping

The codebase references `REQUIREMENTS.md` sections in comments (e.g.
`§NFR-8`, `§9 M4`). The authoritative file↔spec-section index is in
[docs/architecture.md](../architecture.md#how-it-maps-to-the-design).
