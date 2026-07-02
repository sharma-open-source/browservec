// HNSW Web Worker (§NFR-8). Graph construction is the expensive half of HNSW —
// O(log N) beam searches per insert, all CPU — and would jank the main thread
// during ingest exactly like the k-means mean-update did for IVF. This worker
// *owns* the graph and the packed corpus copy: appends stream vectors in
// (transferred, zero-copy) and searches send back only the tiny top-k arrays, so
// both build and query CPU time stay off the main thread. It owns no GPU state.
// Bundled and inlined by Vite via the `?worker&inline` import in hnsw.ts, so it
// ships inside the single-file dist with no separate asset.

import type { Metric } from '../types.js';
import { HNSWGraph, type HNSWGraphParams, type HNSWGraphState } from './hnswGraph.js';

// In a worker `self` is the DedicatedWorkerGlobalScope (postMessage takes a
// transfer list), but the DOM lib types it as Window. Alias to the minimal
// surface we use rather than pull in the conflicting WebWorker lib.
interface WorkerScope {
  onmessage: ((e: MessageEvent<InMessage>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}
const ctx = self as unknown as WorkerScope;

type InitMessage = { type: 'init'; dim: number; metric: Metric; params: HNSWGraphParams };
type AppendMessage = { type: 'append'; id: number; vectors: Float32Array; count: number };
type SearchMessage = { type: 'search'; id: number; query: Float32Array; k: number; ef: number };
type ExportMessage = { type: 'export'; id: number };
type SerializeMessage = { type: 'serialize'; id: number };
type LoadMessage = { type: 'load'; id: number; vectors: Float32Array; count: number; state: HNSWGraphState };
type InMessage = InitMessage | AppendMessage | SearchMessage | ExportMessage | SerializeMessage | LoadMessage;

let graph: HNSWGraph | null = null;

ctx.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    graph = new HNSWGraph(msg.dim, msg.metric, msg.params);
    return;
  }
  if (!graph) throw new Error('hnsw worker got work before init');
  if (msg.type === 'append') {
    graph.append(msg.vectors, msg.count);
    ctx.postMessage({ type: 'appended', id: msg.id, maxLevel: graph.maxLevel });
    return;
  }
  if (msg.type === 'export') {
    // Layer-0 snapshot for the GPU search path (M7b) — transferred, not cloned.
    const links = graph.exportLinks();
    ctx.postMessage({ type: 'graph', id: msg.id, links, entry: graph.entryNode }, [links.buffer]);
    return;
  }
  if (msg.type === 'serialize') {
    // Full-structure snapshot for persistence (M7c) — transferred, not cloned.
    const state = graph.serializeGraph();
    ctx.postMessage({ type: 'state', id: msg.id, state }, [
      state.levels.buffer,
      state.links0.buffer,
      state.upper.buffer,
    ]);
    return;
  }
  if (msg.type === 'load') {
    graph.loadGraph(msg.vectors, msg.count, msg.state);
    ctx.postMessage({ type: 'loaded', id: msg.id, maxLevel: graph.maxLevel });
    return;
  }
  // search
  const hits = graph.search(msg.query, msg.k, msg.ef);
  const rows = new Int32Array(hits.length);
  const scores = new Float32Array(hits.length);
  for (let i = 0; i < hits.length; i++) {
    rows[i] = hits[i]!.row;
    scores[i] = hits[i]!.score;
  }
  ctx.postMessage({ type: 'hits', id: msg.id, rows, scores }, [rows.buffer, scores.buffer]);
};
