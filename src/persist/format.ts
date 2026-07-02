// Versioned binary snapshot format ( persist/format.ts).
//
// Layout (little-endian):
//   [0..4)   magic   "BVEC"
//   [4..8)   u32     format version (= 1)
//   [8..12)  u32     dimension
//   [12..16) u32     metric code (0 cosine, 1 dot, 2 l2)
//   [16..20) u32     flags (bit0 = normalize)
//   [20..24) u32     count (rows)
//   [24..28) u32     metadata JSON byte length
//   [28..32) u32     reserved (0)
//   [32 .. 32+metaLen)            metadata JSON (UTF-8): Array<{id, metadata?}> in row order
//   [pad to 4-byte alignment]
//   [.. + count*dim*4)           Float32 vectors, row-major (post-normalization)

import type { Metric } from '../types.js';
import type { RowEntry } from '../store/store.js';

const MAGIC = 0x43455642; // "BVEC" little-endian
export const FORMAT_VERSION = 1;
const HEADER_BYTES = 32;

const METRIC_TO_CODE: Record<Metric, number> = { cosine: 0, dot: 1, l2: 2 };
const CODE_TO_METRIC: Metric[] = ['cosine', 'dot', 'l2'];

export interface Snapshot {
  dimension: number;
  metric: Metric;
  normalize: boolean;
  count: number;
  entries: Array<{ id: string; metadata?: Metadata }>;
  vectors: Float32Array; // count * dim
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
}

export function serialize(input: SerializeInput): ArrayBuffer {
  const { dimension, metric, normalize, count, entries, vectors } = input;
  if (vectors.length !== count * dimension) {
    throw new Error(`vectors length ${vectors.length} != count*dim ${count * dimension}`);
  }

  const metaJson = JSON.stringify(entries.map((e) => (e.metadata !== undefined ? { id: e.id, metadata: e.metadata } : { id: e.id })));
  const metaBytes = new TextEncoder().encode(metaJson);
  const metaPadded = (metaBytes.length + 3) & ~3; // align vectors to 4 bytes

  const total = HEADER_BYTES + metaPadded + count * dimension * 4;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);

  dv.setUint32(0, MAGIC, true);
  dv.setUint32(4, FORMAT_VERSION, true);
  dv.setUint32(8, dimension, true);
  dv.setUint32(12, METRIC_TO_CODE[metric], true);
  dv.setUint32(16, normalize ? 1 : 0, true);
  dv.setUint32(20, count, true);
  dv.setUint32(24, metaBytes.length, true);
  dv.setUint32(28, 0, true);

  new Uint8Array(buf, HEADER_BYTES, metaBytes.length).set(metaBytes);

  // Vector region is 4-byte aligned, so a Float32Array view is safe.
  new Float32Array(buf, HEADER_BYTES + metaPadded, count * dimension).set(vectors);

  return buf;
}

export function deserialize(buf: ArrayBuffer): Snapshot {
  if (buf.byteLength < HEADER_BYTES) throw new Error('snapshot too small / corrupt');
  const dv = new DataView(buf);

  if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad magic: not a BrowserVec snapshot');
  const version = dv.getUint32(4, true);
  if (version !== FORMAT_VERSION) {
    throw new Error(`unsupported snapshot version ${version} (this build reads ${FORMAT_VERSION})`);
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

  return { dimension, metric, normalize, count, entries, vectors };
}
