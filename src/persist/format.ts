// Versioned binary snapshot format ( persist/format.ts).
//
// Layout (little-endian):
//   [0..4)   magic   "BVEC"
//   [4..8)   u32     format version (1, or 2 when a graph section is present)
//   [8..12)  u32     dimension
//   [12..16) u32     metric code (0 cosine, 1 dot, 2 l2)
//   [16..20) u32     flags (bit0 = normalize)
//   [20..24) u32     count (rows)
//   [24..28) u32     metadata JSON byte length
//   [28..32) u32     v1: reserved (0). v2: byte offset of the HNSW graph section (0 = none)
//   [32 .. 32+metaLen)            metadata JSON (UTF-8): Array<{id, metadata?}> in row order
//   [pad to 4-byte alignment]
//   [.. + count*dim*4)           Float32 vectors, row-major (post-normalization)
//
// v2 graph section (M7c — persisted HNSW graph, so loads skip the O(N·efC)
// rebuild). Starts 4-aligned right after the vectors; all words little-endian:
//   8×u32 header: magic "HNSW", graph version (=1), M, entry, top(maxLevel),
//                 upperLen, reserved, reserved
//   i32 × count            levels (per-node top layer)
//   i32 × count*(2M+1)     layer-0 adjacency, count-prefixed blocks
//   i32 × upperLen         upper-layer blocks, concatenated in row order
// Snapshots WITHOUT a graph keep writing v1, so older builds can still read
// everything they could before — only the new feature pays the version bump.

import type { Metric } from '../types.js';
import type { RowEntry } from '../store/store.js';
import type { HNSWGraphState } from '../index/hnswGraph.js';

const MAGIC = 0x43455642; // "BVEC" little-endian
const GRAPH_MAGIC = 0x57534e48; // "HNSW" little-endian
export const FORMAT_VERSION = 2;
const HEADER_BYTES = 32;
const GRAPH_HEADER_WORDS = 8;
const GRAPH_VERSION = 1;

const METRIC_TO_CODE: Record<Metric, number> = { cosine: 0, dot: 1, l2: 2 };
const CODE_TO_METRIC: Metric[] = ['cosine', 'dot', 'l2'];

export interface Snapshot {
  dimension: number;
  metric: Metric;
  normalize: boolean;
  count: number;
  entries: Array<{ id: string; metadata?: Metadata }>;
  vectors: Float32Array; // count * dim
  /** Persisted HNSW graph (v2, M7c) — absent on v1 snapshots and non-HNSW stores. */
  graph?: HNSWGraphState;
}

type Metadata = RowEntry['metadata'];

export interface SerializeInput {
  dimension: number;
  metric: Metric;
  normalize: boolean;
  count: number;
  /** Row-ordered entries (ids + metadata). */
  entries: Array<{ id: string; metadata?: Metadata }>;
  /** Packed vectors, count * dim floats. */
  vectors: Float32Array;
  /** HNSW graph to persist alongside (M7c). Bumps the snapshot to v2. */
  graph?: HNSWGraphState;
}

export function serialize(input: SerializeInput): ArrayBuffer {
  const { dimension, metric, normalize, count, entries, vectors, graph } = input;
  if (vectors.length !== count * dimension) {
    throw new Error(`vectors length ${vectors.length} != count*dim ${count * dimension}`);
  }

  const metaJson = JSON.stringify(entries.map((e) => (e.metadata !== undefined ? { id: e.id, metadata: e.metadata } : { id: e.id })));
  const metaBytes = new TextEncoder().encode(metaJson);
  const metaPadded = (metaBytes.length + 3) & ~3; // align vectors to 4 bytes

  const graphOffset = graph ? HEADER_BYTES + metaPadded + count * dimension * 4 : 0;
  const graphWords = graph
    ? GRAPH_HEADER_WORDS + graph.levels.length + graph.links0.length + graph.upper.length
    : 0;
  const total = HEADER_BYTES + metaPadded + count * dimension * 4 + graphWords * 4;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);

  dv.setUint32(0, MAGIC, true);
  // Graph-less snapshots stay v1 so older builds keep reading them.
  dv.setUint32(4, graph ? FORMAT_VERSION : 1, true);
  dv.setUint32(8, dimension, true);
  dv.setUint32(12, METRIC_TO_CODE[metric], true);
  dv.setUint32(16, normalize ? 1 : 0, true);
  dv.setUint32(20, count, true);
  dv.setUint32(24, metaBytes.length, true);
  dv.setUint32(28, graphOffset, true);

  new Uint8Array(buf, HEADER_BYTES, metaBytes.length).set(metaBytes);

  // Vector region is 4-byte aligned, so a Float32Array view is safe.
  new Float32Array(buf, HEADER_BYTES + metaPadded, count * dimension).set(vectors);

  if (graph) {
    if (graph.levels.length !== count || graph.links0.length !== count * (2 * graph.M + 1)) {
      throw new Error('graph state does not match snapshot count');
    }
    const head = new Uint32Array(buf, graphOffset, GRAPH_HEADER_WORDS);
    head[0] = GRAPH_MAGIC;
    head[1] = GRAPH_VERSION;
    head[2] = graph.M;
    head[3] = graph.entry >>> 0;
    head[4] = graph.top >>> 0;
    head[5] = graph.upper.length;
    let off = graphOffset + GRAPH_HEADER_WORDS * 4;
    new Int32Array(buf, off, graph.levels.length).set(graph.levels);
    off += graph.levels.length * 4;
    new Int32Array(buf, off, graph.links0.length).set(graph.links0);
    off += graph.links0.length * 4;
    new Int32Array(buf, off, graph.upper.length).set(graph.upper);
  }

  return buf;
}

export function deserialize(buf: ArrayBuffer): Snapshot {
  if (buf.byteLength < HEADER_BYTES) throw new Error('snapshot too small / corrupt');
  const dv = new DataView(buf);

  if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad magic: not a BrowserVec snapshot');
  const version = dv.getUint32(4, true);
  if (version < 1 || version > FORMAT_VERSION) {
    throw new Error(`unsupported snapshot version ${version} (this build reads 1..${FORMAT_VERSION})`);
  }

  const dimension = dv.getUint32(8, true);
  const metric = CODE_TO_METRIC[dv.getUint32(12, true)];
  if (!metric) throw new Error('unknown metric code in snapshot');
  const normalize = (dv.getUint32(16, true) & 1) === 1;
  const count = dv.getUint32(20, true);
  const metaLen = dv.getUint32(24, true);
  const metaPadded = (metaLen + 3) & ~3;

  const metaJson = new TextDecoder().decode(new Uint8Array(buf, HEADER_BYTES, metaLen));
  const entries = JSON.parse(metaJson) as Array<{ id: string; metadata?: Metadata }>;
  if (entries.length !== count) throw new Error(`entry count ${entries.length} != header count ${count}`);

  const vecOffset = HEADER_BYTES + metaPadded;
  const expected = count * dimension * 4;
  if (buf.byteLength < vecOffset + expected) throw new Error('snapshot truncated (vector region)');
  // Copy out so the returned array doesn't pin the whole snapshot buffer.
  const vectors = new Float32Array(buf, vecOffset, count * dimension).slice();

  const snap: Snapshot = { dimension, metric, normalize, count, entries, vectors };

  // v2: optional trailing HNSW graph section (offset stored in the v1-reserved word).
  const graphOffset = version >= 2 ? dv.getUint32(28, true) : 0;
  if (graphOffset !== 0) {
    if (buf.byteLength < graphOffset + GRAPH_HEADER_WORDS * 4) throw new Error('snapshot truncated (graph header)');
    const head = new Uint32Array(buf, graphOffset, GRAPH_HEADER_WORDS);
    if (head[0] !== GRAPH_MAGIC) throw new Error('bad graph section magic');
    if (head[1] !== GRAPH_VERSION) throw new Error(`unsupported graph section version ${head[1]}`);
    const M = head[2]!;
    const entry = head[3]! | 0;
    const top = head[4]! | 0;
    const upperLen = head[5]!;
    const links0Len = count * (2 * M + 1);
    let off = graphOffset + GRAPH_HEADER_WORDS * 4;
    if (buf.byteLength < off + (count + links0Len + upperLen) * 4) {
      throw new Error('snapshot truncated (graph region)');
    }
    const levels = new Int32Array(buf, off, count).slice();
    off += count * 4;
    const links0 = new Int32Array(buf, off, links0Len).slice();
    off += links0Len * 4;
    const upper = new Int32Array(buf, off, upperLen).slice();
    snap.graph = { M, entry, top, levels, links0, upper };
  }

  return snap;
}
