# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/) (pre-1.0:
the public API may still shift between minor versions).

## [Unreleased]

### Added
- `stats()` now reports a per-query time breakdown: `lastQueryGpuMs` (measured
  wait on submitted GPU work + readback) and `lastQueryCpuMs` (JS-side prep,
  candidate gather, re-rank, top-k merge). 0/total respectively on the CPU
  fallback.
- Benchmark dashboard example: CPU / GPU / transfer columns and stacked
  time-breakdown chart, GPU-busy chart, scan-corpus memory chart; quantized
  BrowserVec rows (int8, 1-bit); an HNSW `efSearch` control (previously ran at
  hnswlib's default ~10, cratering recall below `k`); and two new engines —
  voy-search (Rust→WASM kd-tree) and Orama (pure-JS vector search). usearch and
  faiss are documented as unavailable (native Node addons, no browser builds).

### Changed
- Performance pass across the query and ingest hot paths:
  - fp32 distance and k-means assign kernels now use `vec4<f32>` storage
    bindings when the dimension is a multiple of 4 (one 16-byte load per
    iteration instead of four scalar loads); quant kernels (int8/int4/1-bit)
    stage the query as vec4s and hoist the per-row scale out of the inner loop.
  - CPU top-k and GPU top-k partial merges now select directly off the mapped
    readback buffer instead of copying all scores to the JS heap first
    (removes an O(N) copy + allocation per query).
  - Exact fp32 re-rank scores candidates straight from the store's packed
    buffer (no per-candidate vector copy) with a ×4-unrolled dot product.
  - IVF: single-chunk corpora (the common case) skip candidate bucketing
    entirely; probed-list gather uses bulk `set` copies; centroid probe
    scoring is ×4-unrolled.
  - Hadamard rotation folds the 1/√P orthonormal scale into the precomputed
    sign vectors — one pass per round instead of two on quantized ingest
    (output is bit-identical).
  - CPU (no-WebGPU) fallback: scalar L2 loop unrolled ×4; per-query uniform
    upload no longer allocates.

## [0.1.0] - 2026-07-02

Initial public release. Implements milestones M1–M5 and most of M6 — see
[README.md](./README.md) for the full feature breakdown and
[REQUIREMENTS.md](./REQUIREMENTS.md) for the design spec.

### Added
- WebGPU flat brute-force index (cosine/dot/L2, configurable dimension, GPU
  top-k past 4k rows with CPU top-k below).
- TurboQuant quantization: int8, int4, and 1-bit binary, each via randomized
  Hadamard rotation + asymmetric dequant kernels + exact fp32 re-rank.
- IVF approximate index and IVF × quant combos (int8/int4/1-bit), with
  GPU-assisted k-means clustering and probe-limited scans.
- Corpus chunking across multiple GPU buffers so indexes can exceed a single
  buffer's device limit (tested up to ~1M × 768 on default-limit devices).
- Persistence to OPFS with IndexedDB fallback: versioned binary snapshot
  format, export/import, and compaction of tombstoned rows.
- Snapshot encryption at rest (AES-256-GCM + PBKDF2 passphrase).
- On-device text embedding: zero-dependency feature hashing, plus an
  optional transformers.js adapter for real semantic embeddings.
- Web Worker offload for quantized ingest (rotate+quantize) and IVF k-means
  centroid updates, with a transparent in-thread fallback.
- CPU fallback path for devices without WebGPU: WASM-SIMD kernel (~7× faster
  than scalar JS) with an automatic scalar fallback, both bit-exact with the
  GPU path.
- Tombstone-based deletes and in-place `update`/`compact`, consistent across
  every index type.

### Known limitations
- 1/2-bit packing refinements and further WASM fallback coverage are still
  open (tracked under M6).
- Cross-device recall/latency tuning (M6) is in progress; see the demo's
  device report tool for gathering per-device matrices.
