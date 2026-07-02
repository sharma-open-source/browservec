# Integration guide

How to consume `browservec` in your web application — bundler setup, browser
support, CSP, error handling, and patterns for production use.

## Install

```bash
npm install browservec
```

```ts
import { BrowserVec } from 'browservec';
```

The library ships as a single ES module (`dist/browservec.js`) with all GPU
WGSL kernels, the WASM-SIMD fallback module, and the Web Worker for ingest
offload **base64-inlined** — there are no separate assets to co-host.

## Bundle size

The ESM dist is a single file. Tree-shaking removes unused exports at the JS
level, but because the WGSL kernels and the WASM blob are inlined as strings
in the bundle, they are always present regardless of the config you actually
use at runtime. At query/build time, only the paths matching your config are
compiled or instantiated — the unused code is dead data, not dead code.

If bundle size is critical, the biggest savings come from:

- **Dropping quantization** — remove `quantBits`, `quant`, and the WASM
  fallback isn't affected either way.
- **Dropping the embedder helper** — `hashingEmbedder` and
  `transformersEmbedder` are separate exports you can simply not import.
  `@xenova/transformers` is **never loaded** unless you call
  `transformersEmbedder()` — it's in `devDependencies` for the examples only.

## Browser support

| Capability | Where it works | Notes |
|---|---|---|
| WebGPU (GPU path) | Chrome 113+, Edge 113+, Opera, Firefox Nightly | Not on iOS (third-party browsers use WebKit). Android Chrome has WebGPU. |
| CPU fallback (`fallback: 'wasm'`) | Any browser with WASM-SIMD | `v128` instructions required. Scalar JS fallback on older engines — same results, ~7× slower. |
| OPFS persistence | Chromium browsers, Firefox (flag) | Not on iOS/Safari. Falls back to IndexedDB. |
| IndexedDB persistence | All modern browsers | Slower, smaller quota than OPFS. Selected automatically when OPFS is unavailable. |
| Web Worker ingest | All modern browsers | Transparent fallback to main-thread ingest when Workers are unavailable. |

### Quick detection

```ts
const info = BrowserVec.isSupported();
// → { webgpu: boolean, opfs: boolean, wasm: boolean }

if (info.webgpu) {
  // GPU-accelerated path, all configs available
} else if (info.wasm) {
  // CPU fallback — fp32 flat only, no quant/IVF
} else {
  // Neither WebGPU nor WASM — the library won't work
}
```

## Content Security Policy (CSP)

`browservec` uses three features that may interact with CSP:

1. **Web Workers** (quantized ingest, IVF k-means) — inlined via
   `?worker&inline` at build time, so no `worker-src` exception is needed.
   The worker blob is created from a base64 data URL.
2. **WASM module** (CPU fallback) — requires `'wasm-unsafe-eval'` in
   `script-src` when the WASM module is compiled from a base64 string.
   Without it, the WASM-SIMD kernel won't compile and the scalar JS fallback
   is used instead (slower but functional).
3. **WebGPU** — no CSP-specific requirement beyond what the browser already
   allows for WebGPU.

Minimal CSP header for full functionality:

```
Content-Security-Policy: script-src 'wasm-unsafe-eval'; worker-src blob:
```

Without `'wasm-unsafe-eval'`, the CPU fallback silently degrades to the
scalar JS loop — the library still works.

## Import strategies

### ESM (recommended)

```ts
import { BrowserVec, hashingEmbedder } from 'browservec';
```

### CDN (via esm.sh or unpkg)

```html
<script type="module">
import { BrowserVec } from 'https://esm.sh/browservec';
// ...
</script>
```

### Dynamic import (lazy-load the library)

```ts
const { BrowserVec } = await import('browservec');
```

## Error handling

All errors thrown by `browservec` are native `Error` instances. Key errors to
handle:

| Error | When | How to handle |
|---|---|---|
| `WebGPUUnavailableError` | `create()` when WebGPU is absent and `fallback: 'error'` (default) | Fall back to CPU or show a message. Pass `fallback: 'wasm'` to auto-degrade. |
| `"CPU fallback supports fp32 flat only"` | `create()` with `fallback: 'wasm'` + `quantBits` or `ann` | Drop quantization/IVF when targeting CPU-only environments. |
| `"dimension must be a positive integer"` | `create()` with invalid `dimension` | Validate input. |
| `"duplicate id: …"` | `add()`/`addBatch()` with an id already in the store | Check `db.get(id)` first, or use `update()` for upsert. |
| `"query dim X != store dim Y"` | `query()` with wrong-dimension vector | Ensure query dimension matches store dimension. |
| `"text methods require an embedder"` | `addText()`/`queryText()` without an `embedder` in config | Pass an embedder or use `add()`/`query()` directly. |
| `"snapshot dimension/metric mismatch"` | `create()` with `persist.autoLoad` and the snapshot doesn't match config | Match the config to the snapshot, or load into a fresh store. |
| `"snapshot is encrypted; pass encryption"` | Creating/importing an encrypted snapshot without a passphrase | Provide the correct passphrase. |
| "decryption failed" | Wrong passphrase or tampered encrypted blob | Show a message — the cause is intentionally vague (no oracle). |

## Reusing a WebGPU device

If your app already has a `GPUDevice`, pass it to avoid requesting a second
one:

```ts
const device = await navigator.gpu.requestAdapter().then(a => a!.requestDevice());
const db = await BrowserVec.create({ dimension: 768, device });
```

When you call `db.destroy()`, the injected device is **not** destroyed — only
the browservec-internal references are cleaned up.

## CPU fallback strategy

For apps targeting the widest browser compatibility:

```ts
const info = BrowserVec.isSupported();
const fallback = info.webgpu ? 'error' : 'wasm';
const db = await BrowserVec.create({
  dimension: 768,
  metric: 'cosine',
  fallback,                              // GPU if available, CPU otherwise
  quantBits: info.webgpu ? 8 : 0,        // only quantize on GPU
}).catch(() => null);                    // handle the no-WASM case
```

## Lifecycle management

```ts
// Create
const db = await BrowserVec.create({ dimension: 768, persist: { name: 'my-store' } });

// Use
await db.addBatch(records);
const hits = await db.query(queryVec);

// Persist
await db.save();

// Clean up when done — frees GPU resources
db.destroy();
```

Without calling `destroy()`, the GPU device persists until the page is
navigated away or the tab closes. For single-page apps that create and
discard stores, always call `destroy()`.

## Framework integration

### React

Reusable hooks are demonstrated in
[`examples/16-react-hooks.html`](../examples/16-react-hooks.html) — four hooks
you can copy into your own project:

| Hook | Purpose |
|---|---|
| `useVectorStore(config)` | Creates a `BrowserVec` instance; auto-destroys on unmount or config change. Returns `{ store, ready, error, stats, count }`. |
| `useSimilaritySearch(store, vector, opts)` | Runs a query reactively when the vector reference changes. Returns `{ results, loading, error, search }`. |
| `useEmbedding(embedder)` | Embeds text using the configured embedder. Returns `{ embed, embedding, loading }`. |
| `useRetriever(store, embedder)` | RAG-style: embed a query text and search the store in one call. Returns `{ retrieve, context, loading, error }`. |

The hooks load BrowserVec and the embedder directly — no wrapper library
needed. They handle lifecycle (destroy on unmount), reactive re-query, and
loading/error states.

### Vue

Equivalent composables in
[`examples/17-vue-composables.html`](../examples/17-vue-composables.html) — same
API surface, Vue Composition API style:

| Composable | Purpose |
|---|---|
| `useVectorStore(config)` | Creates a BrowserVec instance; auto-destroys on unmount. Returns reactive `{ store, ready, error, stats, count }`. |
| `useSimilaritySearch(store, vector, opts)` | Watches the vector ref and runs queries reactively. |
| `useEmbedding(embedder)` | Reactive text embedding via `embed(text)`. |
| `useRetriever(store, embedder)` | Text-in, context-out retrieval composable. |

Both the React hooks and Vue composables follow the same patterns and
produce identical results — choose whichever matches your framework.

## Examples

Each example in [`examples/`](../examples/) is a self-contained HTML page you
can open via `npm run dev`:

| Example | What it covers |
|---|---|
| [`examples/01-basic-flat.html`](../examples/01-basic-flat.html) | Create, addBatch, query, GPU-vs-CPU correctness check |
| [`examples/14-advanced-api.html`](../examples/14-advanced-api.html) | isSupported, stats, destroy, device reuse, chunkRows, autoLoad |
| [`examples/06-persistence.html`](../examples/06-persistence.html) | Save/load, export/import, cold-start reload |
| [`examples/07-encryption.html`](../examples/07-encryption.html) | Encrypted persistence, wrong-passphrase handling |
| [`examples/15-browser-extension.html`](../examples/15-browser-extension.html) | Browser extension pattern — popup UI, page capture, local persistence, offline search |
| [`examples/16-react-hooks.html`](../examples/16-react-hooks.html) | React hooks: useVectorStore, useSimilaritySearch, useEmbedding, useRetriever |
| [`examples/17-vue-composables.html`](../examples/17-vue-composables.html) | Vue composables: useVectorStore, useSimilaritySearch, useEmbedding, useRetriever |

For a complete browser-environment reference, the
[demo](../demo/index.html) exercises every config combination with an
interactive report.
