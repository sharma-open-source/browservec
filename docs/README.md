# BrowserVec developer docs

This folder is for people modifying or extending BrowserVec itself — for
usage as a library, start with the top-level [README.md](../README.md).

- [architecture.md](./architecture.md) — module map, data flow, GPU vs. CPU
  paths, how a query/ingest actually executes end to end.
- [api-reference.md](./api-reference.md) — full public API surface
  (`BrowserVec` class, config types, return types), pulled from
  [src/index.ts](../src/index.ts) and [src/types.ts](../src/types.ts).
- [internals.md](./internals.md) — per-subsystem notes: quantization codec,
  IVF/k-means, persistence format, Worker offload, CPU/WASM fallback.
- [../REQUIREMENTS.md](../REQUIREMENTS.md) — the original design spec
  (functional/non-functional requirements, milestone definitions). Code
  comments reference sections of this file (e.g. `§NFR-8`, `§9 M4`).
- [../CHANGELOG.md](../CHANGELOG.md) — release history.

The [README's "How it maps to the design" table](../README.md#how-it-maps-to-the-design)
is the fastest way to jump from a requirement to the file that implements it.
