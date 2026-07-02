# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/) (pre-1.0:
the public API may still shift between minor versions).

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
