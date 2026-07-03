# BrowserVec developer docs

Start here, then follow the path that matches what you're doing.

## I'm new to BrowserVec

- [Quick start](../README.md#quick-start) — get the demo running in 2 commands
- [Features](../README.md#features) — overview of every capability
- [Core API](../README.md#core-api) — minimal code example to see the shape
- [Benchmarks](../README.md#benchmarks) — what performance to expect

## I'm integrating BrowserVec into my app

- [Integration guide](./guide/integration.md) — install, bundle, CSP, browser
  support, error handling, device reuse, lifecycle
- [Configuration guide](./guide/configuration.md) — decision tree, parameter
  tuning, per-query overrides
- [API reference](./api-reference.md) — every method, config field, and type
- [Feature deep-dives](./features.md) — code examples for each feature

## I'm choosing an index configuration

- **[Decision tree →](./guide/configuration.md#decision-tree)** — flat fp32,
  quantized, IVF, or IVF×quant combo
- [Parameter tuning](./guide/configuration.md#parameter-tuning) — dimension,
  metric, quantBits, rerankFactor, nprobe, nlist

## I want to understand how it works

- [Architecture](./architecture.md) — module map, data flow, index selection,
  ingest/query/delete paths, file↔spec mapping table
- [Internals](./internals.md) — device acquisition, store internals,
  persistence format, encryption envelope, quantization codec, IVF/k-means
  details, Worker offload seam, CPU/WASM fallback details

## I want to run the examples

Open any via `npm run dev`:

| Example | Feature |
|---|---|
| [`examples/01-basic-flat.html`](../examples/01-basic-flat.html) | Basic flat search — create, addBatch, query |
| [`examples/02-quantization.html`](../examples/02-quantization.html) | int8 / int4 quantization comparison |
| [`examples/03-ivf-index.html`](../examples/03-ivf-index.html) | IVF approximate search with nprobe sweep |
| [`examples/04-ivf-quant-combo.html`](../examples/04-ivf-quant-combo.html) | IVF × int8/int4 combo — the 1M path |
| [`examples/18-hnsw-index.html`](../examples/18-hnsw-index.html) | HNSW graph index — efSearch sweep, recall vs speed, works without WebGPU |
| [`examples/19-hnsw-gpu.html`](../examples/19-hnsw-gpu.html) | GPU graph search — one-dispatch beam kernel, queryBatch, CPU vs GPU crossover |
| [`examples/05-text-retrieval.html`](../examples/05-text-retrieval.html) | End-to-end text → search |
| [`examples/06-persistence.html`](../examples/06-persistence.html) | Save/load to OPFS/IndexedDB |
| [`examples/07-encryption.html`](../examples/07-encryption.html) | AES-256-GCM encrypted snapshots |
| [`examples/08-deletion-compaction.html`](../examples/08-deletion-compaction.html) | Tombstone deletes, update, compact |
| [`examples/09-worker-ingest.html`](../examples/09-worker-ingest.html) | Worker-offloaded quantized ingest |
| [`examples/10-chunking.html`](../examples/10-chunking.html) | Multi-buffer corpus chunking |
| [`examples/11-custom-embedder.html`](../examples/11-custom-embedder.html) | Custom embedder implementation |
| [`examples/12-transformers-embedder.html`](../examples/12-transformers-embedder.html) | Transformers.js semantic embeddings |
| [`examples/13-all-metrics.html`](../examples/13-all-metrics.html) | Cosine vs dot vs L2 comparison |
| [`examples/14-advanced-api.html`](../examples/14-advanced-api.html) | isSupported, stats, destroy, device reuse |
| [`examples/15-browser-extension.html`](../examples/15-browser-extension.html) | Browser extension pattern — capture pages, persist, semantic search, offline |
| [`examples/16-react-hooks.html`](../examples/16-react-hooks.html) | React hooks: useVectorStore, useSimilaritySearch, useEmbedding, useRetriever |
| [`examples/17-vue-composables.html`](../examples/17-vue-composables.html) | Vue composables: useVectorStore, useSimilaritySearch, useEmbedding, useRetriever |
| [`examples/perf-benchmark.html`](../examples/perf-benchmark.html) | Large-scale performance benchmark |

## I want to contribute to BrowserVec

- [Development guide](./guide/development.md) — setup, build, project layout,
  how to add an index type / WGSL kernel, code conventions
- [Architecture](./architecture.md) — understand the module structure and
  data flow before making changes

## Reference

- [API reference](./api-reference.md) — complete public API surface
- [Architecture → spec mapping](./architecture.md#how-it-maps-to-the-design)
  — which file implements which REQUIREMENTS.md section
- [REQUIREMENTS.md](../REQUIREMENTS.md) — original design spec
- [CHANGELOG.md](../CHANGELOG.md) — release history
