// Shared vector-search engine adapters + benchmark runner + chart rendering.
//
// Used by both benchmark-dashboard.html (synthetic vectors) and
// benchmark-real-embeddings.html (real sentence embeddings) — the engines,
// timing harness, and rendering are identical; only where the vectors come
// from differs between the two pages. Keeping this in one module means the
// two benchmarks can never drift apart on what "GPU time" or "recall" means.
//
// Every engine exposes the same shape so the runner doesn't special-case any
// of them: build(vectors, ids) -> {ms}, query(vec, k) -> {ids, ms, cpuMs?, gpuMs?},
// memoryBytes(), destroy(); optionally transferEstimateMs() (GPU engines only).

import { BrowserVec } from '../../src/index.ts';
import { loadHnswlib } from 'hnswlib-wasm';
import { array as vArray, dot as vDot } from 'vectorious';
import { create as oramaCreate, insertMultiple as oramaInsertMultiple, search as oramaSearch } from '@orama/orama';
import * as voyBg from 'voy-search/voy_search_bg.js';
import voyWasmUrl from 'voy-search/voy_search_bg.wasm?url';

// ---- BrowserVec (WebGPU) ----
//
// The caller owns the GPU device for BrowserVec engines (BrowserVec accepts
// an injected device) so it can measure a raw readback round-trip on the SAME
// device — that's the "Transfer" column. Mirrors the library's own limit requests.
export async function createBenchDevice() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const want = {
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
  };
  const requiredLimits = {};
  for (const [key, v] of Object.entries(want)) {
    const cap = adapter.limits[key];
    if (typeof cap === 'number') requiredLimits[key] = Math.min(v, cap);
  }
  return adapter.requestDevice({ requiredLimits });
}

// Average copy+map round-trip on an idle queue: the fixed synchronization cost
// every GPU query pays to get results back to JS. Score readbacks are small
// (top-k partials), so latency, not bandwidth, dominates transfer.
export async function measureTransferMs(device, bytes = 4096, iters = 12) {
  const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let total = 0;
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    rb.unmap();
    if (i >= 2) total += performance.now() - t0; // first two are warm-up
  }
  src.destroy();
  rb.destroy();
  return total / (iters - 2);
}

// Shared query wrapper: wall time from the adapter, CPU/GPU split from
// BrowserVec's own stats (see src/engine/profile.ts).
function browserVecQuery(db) {
  return async (q, k) => {
    const t0 = performance.now();
    const hits = await db.query(q, { k });
    const ms = performance.now() - t0;
    const s = db.stats();
    return { ids: hits.map((h) => h.id), ms, cpuMs: s.lastQueryCpuMs ?? ms, gpuMs: s.lastQueryGpuMs ?? 0 };
  };
}

export function makeBrowserVecEngine(dim) {
  let db;
  let device = null;
  let n = 0;
  return {
    name: 'BrowserVec (WebGPU flat)',
    async build(vectors, ids) {
      device = await createBenchDevice();
      db = await BrowserVec.create({
        dimension: dim, metric: 'cosine', normalize: false, fallback: 'wasm',
        ...(device ? { device } : {}),
      });
      this.query = browserVecQuery(db);
      n = vectors.length;
      const t0 = performance.now();
      await db.addBatch(vectors.map((v, i) => ({ id: ids[i], vector: v })));
      return { ms: performance.now() - t0 };
    },
    query: null, // installed in build()
    async transferEstimateMs() {
      return device ? measureTransferMs(device) : 0;
    },
    memoryBytes() {
      return n * dim * 4; // raw fp32 corpus; GPU buffer(s) hold an equivalent copy
    },
    destroy() { db?.destroy(); }, // also destroys the injected device
  };
}

// Quantized BrowserVec: the corpus the GPU scans is rotated + packed codes
// (int8 = ~4× smaller than fp32, 1-bit = ~32×), and an exact fp32 re-rank on
// the over-fetched candidates recovers recall. Memory reports the scan-corpus
// bytes (codes + per-row scale) — the same convention as the flat row, which
// reports its fp32 scan corpus. Requires WebGPU (no CPU fallback for quant).
export function makeBrowserVecQuantEngine(dim, bits) {
  let db;
  let device = null;
  let n = 0;
  const paddedDim = Math.max(4, 2 ** Math.ceil(Math.log2(dim))); // Hadamard pad
  return {
    name: `BrowserVec (WebGPU ${bits === 1 ? '1-bit' : `int${bits}`} + rerank)`,
    async build(vectors, ids) {
      device = await createBenchDevice();
      if (!device) throw new Error('quantized mode requires WebGPU');
      db = await BrowserVec.create({ dimension: dim, metric: 'cosine', normalize: false, quantBits: bits, device });
      this.query = browserVecQuery(db);
      n = vectors.length;
      const t0 = performance.now();
      await db.addBatch(vectors.map((v, i) => ({ id: ids[i], vector: v })));
      return { ms: performance.now() - t0 };
    },
    query: null, // installed in build()
    async transferEstimateMs() {
      return device ? measureTransferMs(device) : 0;
    },
    memoryBytes() {
      const codeBytes = (paddedDim * bits) / 8;
      return n * (codeBytes + 4); // packed codes + fp32 scale per row
    },
    destroy() { db?.destroy(); }, // also destroys the injected device
  };
}

// ---- voy-search (Rust→WASM kd-tree) ----
//
// The wasm-bindgen module is initialized manually (fetch + instantiate) so no
// extra Vite plugin is needed.
let voyInit = null;
async function initVoy() {
  voyInit ??= (async () => {
    const imports = { './voy_search_bg.js': voyBg };
    let instance;
    try {
      ({ instance } = await WebAssembly.instantiateStreaming(fetch(voyWasmUrl), imports));
    } catch {
      // Server didn't send application/wasm — fall back to ArrayBuffer instantiation.
      const buf = await (await fetch(voyWasmUrl)).arrayBuffer();
      ({ instance } = await WebAssembly.instantiate(buf, imports));
    }
    voyBg.__wbg_set_wasm(instance.exports);
    return voyBg;
  })();
  return voyInit;
}

// voy ingests JSON resources (number[] per vector) — fine at demo scale, but
// large corpus x dim means many boxed numbers through serde; callers should
// skip voy above this cell budget.
export const VOY_MAX_CELLS = 8_000_000;

export function makeVoyEngine(dim) {
  let voy;
  let n = 0;
  return {
    name: 'voy-search (WASM kd-tree)',
    async build(vectors, ids) {
      const bg = await initVoy();
      n = vectors.length;
      const t0 = performance.now();
      const resource = {
        embeddings: vectors.map((v, i) => ({ id: ids[i], title: '', url: '', embeddings: Array.from(v) })),
      };
      voy = new bg.Voy(resource);
      return { ms: performance.now() - t0 };
    },
    async query(q, k) {
      const t0 = performance.now();
      const res = voy.search(q, k);
      const ms = performance.now() - t0;
      return { ids: res.neighbors.map((nb) => nb.id), ms, cpuMs: ms, gpuMs: 0 };
    },
    memoryBytes() {
      return n * dim * 4; // vector data; kd-tree node overhead not estimated
    },
    destroy() { voy?.free?.(); },
  };
}

// ---- Orama (pure-JS vector search) ----
export function makeOramaEngine(dim) {
  let db;
  let n = 0;
  return {
    name: 'Orama (pure JS vector)',
    async build(vectors, ids) {
      n = vectors.length;
      const t0 = performance.now();
      db = oramaCreate({ schema: { embedding: `vector[${dim}]` } });
      await oramaInsertMultiple(db, vectors.map((v, i) => ({ id: ids[i], embedding: Array.from(v) })));
      return { ms: performance.now() - t0 };
    },
    async query(q, k) {
      const t0 = performance.now();
      const res = await oramaSearch(db, {
        mode: 'vector',
        vector: { value: Array.from(q), property: 'embedding' },
        similarity: 0, // don't threshold-filter; we want a plain top-k
        limit: k,
      });
      const ms = performance.now() - t0;
      return { ids: res.hits.map((h) => h.id), ms, cpuMs: ms, gpuMs: 0 };
    },
    memoryBytes() {
      return n * dim * 4; // vector data; JS object/index overhead not estimated
    },
    destroy() { db = null; },
  };
}

// ---- hnswlib-wasm (HNSW) ----
export function makeHnswEngine(dim, efConstruction, m, efSearch) {
  let index;
  let labelToId = [];
  let n = 0;
  return {
    name: `hnswlib-wasm (HNSW, efC=${efConstruction} M=${m} efS=${efSearch})`,
    async build(vectors, ids) {
      const mod = await loadHnswlib();
      index = new mod.HierarchicalNSW('cosine', dim, '');
      index.initIndex(vectors.length, m, efConstruction, 100);
      // efSearch defaults to ~10 in hnswlib — below k it can't even hold k
      // candidates and recall collapses. Always set it explicitly (≥ k, done
      // by the caller) so the recall/latency trade-off is the real one.
      index.setEfSearch(efSearch);
      labelToId = ids;
      n = vectors.length;
      const t0 = performance.now();
      const items = vectors.map((v) => Array.from(v));
      const labels = ids.map((_, i) => i);
      index.addPoints(items, labels, false);
      return { ms: performance.now() - t0 };
    },
    async query(q, k) {
      const t0 = performance.now();
      const res = index.searchKnn(Array.from(q), k, undefined);
      const ids = res.neighbors.map((label) => labelToId[label]);
      return { ids, ms: performance.now() - t0 };
    },
    memoryBytes() {
      // Data + a rough per-node graph-link overhead (M bidirectional links,
      // ~2 layers on average, 4 bytes/link + label/metadata slop). This is a
      // back-of-envelope estimate, not a measured allocation.
      const dataBytes = n * dim * 4;
      const graphBytes = n * m * 2 * 4 * 2;
      return dataBytes + graphBytes;
    },
    destroy() { index?.delete?.(); },
  };
}

// ---- Vectorious (BLAS brute force) ----
export function makeVectoriousEngine(dim) {
  let rows = [];
  let ids = [];
  return {
    name: 'Vectorious (brute force, BLAS)',
    async build(vectors, idList) {
      const t0 = performance.now();
      rows = vectors.map((v) => vArray(Array.from(v)));
      ids = idList;
      return { ms: performance.now() - t0 };
    },
    async query(q, k) {
      const t0 = performance.now();
      const qa = vArray(Array.from(q));
      const scored = new Array(rows.length);
      for (let i = 0; i < rows.length; i++) scored[i] = { id: ids[i], score: vDot(qa, rows[i]) };
      scored.sort((a, b) => b.score - a.score);
      const top = scored.slice(0, k).map((s) => s.id);
      return { ids: top, ms: performance.now() - t0 };
    },
    memoryBytes() { return rows.length * dim * 4; },
    destroy() {},
  };
}

// ---- Naive JS brute force (ground truth) ----
export function makeNaiveBruteForceEngine(dim) {
  let data = [];
  let ids = [];
  return {
    name: 'Brute force (naive JS)',
    async build(vectors, idList) {
      const t0 = performance.now();
      data = vectors;
      ids = idList;
      return { ms: performance.now() - t0 };
    },
    async query(q, k) {
      const t0 = performance.now();
      const n = data.length;
      const scores = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const row = data[i];
        let s = 0;
        for (let d = 0; d < dim; d++) s += row[d] * q[d];
        scores[i] = s;
      }
      const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, k);
      return { ids: order.map((i) => ids[i]), ms: performance.now() - t0 };
    },
    memoryBytes() { return data.length * dim * 4; },
    destroy() {},
  };
}

// ---- Benchmark runner ----
//
// Runs one engine end to end and returns both its timing/memory metrics and
// the raw per-query result-id lists — recall is computed by the caller
// against a shared ground truth, so the exact-reference engine only builds
// and queries once (its own results double as the ground truth).
export async function runEngine(engine, corpus, queries, k, log = () => {}) {
  log(`--- ${engine.name} ---`);

  const buildResult = await engine.build(corpus.vectors, corpus.ids);
  const ingestRate = corpus.vectors.length / (buildResult.ms / 1000);
  log(`  ingest: ${buildResult.ms.toFixed(0)} ms (${ingestRate.toFixed(0)} rows/s)`);

  // Warm-up query (compiles shaders / JITs hot loops).
  await engine.query(queries[0], k);

  let totalMs = 0;
  let totalCpuMs = 0;
  let totalGpuMs = 0;
  const perQueryIds = [];
  for (let i = 0; i < queries.length; i++) {
    const r = await engine.query(queries[i], k);
    totalMs += r.ms;
    totalCpuMs += r.cpuMs ?? r.ms; // WASM/JS engines: everything is CPU
    totalGpuMs += r.gpuMs ?? 0;
    perQueryIds.push(r.ids);
  }
  const nq = queries.length;
  const avgMs = totalMs / nq;
  const qps = 1000 / avgMs;

  // Transfer: measured raw readback round-trip on the engine's own device
  // (0 / n.a. for CPU-only engines). Reported GPU time is the wait minus that
  // fixed sync cost, so "GPU" ≈ kernel execution + queue scheduling.
  const transferMs = engine.transferEstimateMs ? await engine.transferEstimateMs() : 0;
  const avgGpuWaitMs = totalGpuMs / nq;
  const avgTransferMs = Math.min(transferMs, avgGpuWaitMs);
  const avgGpuMs = avgGpuWaitMs - avgTransferMs;
  const avgCpuMs = totalCpuMs / nq;
  const gpuBusy = avgMs > 0 ? avgGpuWaitMs / avgMs : 0;

  log(`  query: ${avgMs.toFixed(3)} ms/q avg, ${qps.toFixed(0)} q/s` +
    (avgGpuWaitMs > 0 ? ` (cpu ${avgCpuMs.toFixed(3)} + gpu ${avgGpuMs.toFixed(3)} + xfer ${avgTransferMs.toFixed(3)} ms)` : ''));

  const memBytes = engine.memoryBytes();
  engine.destroy();

  return {
    name: engine.name,
    ingestMs: buildResult.ms,
    ingestRate,
    avgQueryMs: avgMs,
    qps,
    avgCpuMs,
    avgGpuMs,
    avgTransferMs,
    gpuBusy,
    isGpu: avgGpuWaitMs > 0,
    perQueryIds,
    memBytes,
  };
}

export function recallAgainst(perQueryIds, groundTruth, k) {
  let hitCount = 0;
  for (let i = 0; i < perQueryIds.length; i++) {
    const truth = groundTruth[i];
    hitCount += perQueryIds[i].filter((id) => truth.has(id)).length;
  }
  return hitCount / (perQueryIds.length * k);
}

/**
 * Build the standard engine list (BrowserVec flat/int8/1-bit, HNSW, Vectorious,
 * Orama, and voy-search unless the corpus exceeds its comfort zone) for a given
 * dimension and HNSW config.
 */
export function standardEngines(dim, n, { efConstruction, m, efSearch }, log = () => {}) {
  const engines = [
    makeBrowserVecEngine(dim),
    makeBrowserVecQuantEngine(dim, 8),
    makeBrowserVecQuantEngine(dim, 1),
    makeHnswEngine(dim, efConstruction, m, efSearch),
    makeVectoriousEngine(dim),
    makeOramaEngine(dim),
  ];
  if (n * dim <= VOY_MAX_CELLS) {
    engines.push(makeVoyEngine(dim));
  } else {
    log(`voy-search skipped: ${n.toLocaleString()}×${dim} exceeds its JSON-ingest comfort zone (${(VOY_MAX_CELLS / 1e6).toFixed(0)}M cells).`, 'warn');
  }
  return engines;
}

/** BrowserVec-only engine list (flat + int8 + 1-bit) — no third-party comparisons. */
export function browserVecEngines(dim) {
  return [
    makeBrowserVecEngine(dim),
    makeBrowserVecQuantEngine(dim, 8),
    makeBrowserVecQuantEngine(dim, 1),
  ];
}

// ---- Rendering ----

export function fmtBytes(b) {
  if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + ' MiB';
  if (b >= 1024) return (b / 1024).toFixed(1) + ' KiB';
  return b.toFixed(0) + ' B';
}

/** Fill the standard 9-column results table (see either benchmark page's <thead>). */
export function renderResultsTable(tbody, results) {
  tbody.innerHTML = '';
  for (const r of results) {
    const tr = tbody.insertRow();
    const recallPct = r.recall * 100;
    const recallCls = recallPct >= 95 ? 'good' : recallPct >= 80 ? 'warn' : 'bad';
    tr.innerHTML = `
      <td>${r.name}</td>
      <td>${r.ingestRate.toFixed(0)} rows/s</td>
      <td>${r.avgQueryMs.toFixed(3)} ms</td>
      <td>${r.qps.toFixed(0)}</td>
      <td>${r.avgCpuMs.toFixed(3)} ms</td>
      <td>${r.isGpu ? r.avgGpuMs.toFixed(3) + ' ms' : '—'}</td>
      <td>${r.isGpu ? r.avgTransferMs.toFixed(3) + ' ms' : '—'}</td>
      <td class="${recallCls}">${recallPct.toFixed(1)}%</td>
      <td>${fmtBytes(r.memBytes)}</td>
    `;
  }
}

/** Render the standard set of charts (latency, breakdown, GPU busy, memory, recall) into given container ids. */
export function renderStandardCharts(ids, results) {
  renderBarChart(ids.latency, 'Avg query latency (lower is better)', results, (r) => r.avgQueryMs, (v) => v.toFixed(3) + ' ms', (v, max) => (v <= max * 0.34 ? '#2e7d32' : v <= max * 0.67 ? '#e65100' : '#d32f2f'));
  renderStackedChart(ids.breakdown, 'Query time breakdown — CPU / GPU / transfer (ms)', results, [
    { label: 'CPU', color: '#1a73e8', valueFn: (r) => r.avgCpuMs },
    { label: 'GPU', color: '#2e7d32', valueFn: (r) => r.avgGpuMs },
    { label: 'Transfer', color: '#e65100', valueFn: (r) => r.avgTransferMs },
  ]);
  renderBarChart(ids.gpuBusy, 'GPU busy — fraction of query wall time waiting on GPU work (approx)', results, (r) => r.gpuBusy * 100, (v) => v.toFixed(0) + '%', () => '#2e7d32');
  renderBarChart(ids.memory, 'Scan-corpus memory (lower is better)', results, (r) => r.memBytes, (v) => fmtBytes(v), (v, max) => (v <= max * 0.34 ? '#2e7d32' : v <= max * 0.67 ? '#e65100' : '#d32f2f'));
  renderBarChart(ids.recall, 'Recall@k (higher is better)', results, (r) => r.recall * 100, (v) => v.toFixed(1) + '%', (v) => (v >= 95 ? '#2e7d32' : v >= 80 ? '#e65100' : '#d32f2f'));
}

// Horizontal stacked bars: one row per engine, segments per component. All rows
// share one scale (the largest total), so lengths compare across engines.
export function renderStackedChart(containerId, title, results, segments) {
  const container = document.getElementById(containerId);
  container.innerHTML = `<h3 style="font-size:.9rem;margin:.5rem 0;">${title}</h3>`;

  const legend = document.createElement('div');
  legend.style.cssText = 'font-size:.72rem;color:#555;margin:.2rem 0 .4rem;';
  legend.innerHTML = segments
    .map((s) => `<span style="display:inline-block;width:10px;height:10px;background:${s.color};border-radius:2px;margin:0 .3rem 0 .8rem;"></span>${s.label}`)
    .join('');
  container.appendChild(legend);

  const max = Math.max(...results.map((r) => segments.reduce((a, s) => a + s.valueFn(r), 0)), 1e-9);
  for (const r of results) {
    const total = segments.reduce((a, s) => a + s.valueFn(r), 0);
    const row = document.createElement('div');
    row.className = 'row';
    const bars = segments
      .filter((s) => s.valueFn(r) > 0)
      .map((s) => `<span class="bar" style="display:inline-block;width:${Math.max((s.valueFn(r) / max) * 100, 0.4)}%;background:${s.color};height:12px;border-radius:0;" title="${s.label}: ${s.valueFn(r).toFixed(3)} ms"></span>`)
      .join('');
    row.innerHTML = `
      <span class="lbl">${r.name}</span>
      <span class="bar-wrap" style="white-space:nowrap;">${bars} ${total.toFixed(3)} ms</span>
    `;
    container.appendChild(row);
  }
}

export function renderBarChart(containerId, title, results, valueFn, fmtFn, colorFn) {
  const container = document.getElementById(containerId);
  container.innerHTML = `<h3 style="font-size:.9rem;margin:.5rem 0;">${title}</h3>`;
  const max = Math.max(...results.map(valueFn), 1e-9);

  for (const r of results) {
    const val = valueFn(r);
    const pct = (val / max) * 100;
    const color = colorFn(val, max);
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `
      <span class="lbl">${r.name}</span>
      <span class="bar-wrap">
        <span class="bar" style="display:inline-block;width:${Math.max(pct, 1.5)}%;background:${color};height:12px;border-radius:3px;"></span>
        ${fmtFn(val)}
      </span>
    `;
    container.appendChild(row);
  }
}
