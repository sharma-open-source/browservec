// Templated distance kernel (lever 4: dimension-specialized shaders).
//
// `DIM` and `WG` are injected as compile-time constants so the inner loop fully
// unrolls and the accumulator vectorizes to vec4<f32>. The metric reduction is
// baked into the template, so there is no per-row branch. One invocation scores
// one corpus row; the full score array is written to a storage buffer and the CPU
// does top-k  (GPU top-k, lever 3).

import type { Metric } from '../../types.js';

export interface DistanceShaderKey {
  dim: number;
  metric: Metric;
  workgroupSize: number;
  /**
   * Indexed mode (IVF, §9 M4): instead of scoring row == gid, score the row id
   * found at `candidates[gid]`. Adds a binding(4) candidate-id buffer; params.x
   * becomes the candidate count. Lets one dispatch score only the probed lists.
   */
  indexed?: boolean;
}

/** Per-metric: how to fold a vec4 chunk into the accumulator, and finalize. */
function metricSnippets(metric: Metric): { acc: string; finalize: string; init: string } {
  switch (metric) {
    case 'dot':
    case 'cosine':
      // For cosine, vectors are normalized on insert + query normalized in JS,
      // so dot product IS cosine similarity. Higher = closer.
      return {
        init: 'var acc: f32 = 0.0;',
        acc: 'acc = dot(c, q) + acc;',
        finalize: 'return acc;',
      };
    case 'l2':
      // Negative squared L2 so that "higher = closer" holds uniformly,
      // matching cosine/dot ordering for a single top-k path.
      return {
        init: 'var acc: f32 = 0.0;',
        acc: 'let d = c - q; acc = dot(d, d) + acc;',
        finalize: 'return -acc;',
      };
  }
}

export function buildDistanceShader(key: DistanceShaderKey): string {
  const { dim, metric, workgroupSize, indexed = false } = key;
  const m = metricSnippets(metric);

  // Number of full vec4 chunks and the scalar tail (dim not divisible by 4).
  const vecChunks = Math.floor(dim / 4);
  const tail = dim % 4;

  // In indexed (IVF) mode the row to score comes from a candidate-id buffer.
  const candidateBinding = indexed
    ? '@group(0) @binding(4) var<storage, read> candidates: array<u32>;'
    : '';
  const rowExpr = indexed ? 'candidates[gid.x]' : 'gid.x';
  // Chunked corpus (§NFR-10): the bound `corpus` holds one chunk starting at
  // local row 0. Flat mode writes the *global* score slot params.y (chunk base
  // row) + gid.x. Indexed (IVF) mode buckets candidates by chunk and scores each
  // bucket with candidate ids *local* to its chunk, writing into a dense output
  // region at params.z (this bucket's offset in the scores buffer). Single-chunk
  // stores pass params.y = params.z = 0, so both are a no-op there.
  const scoreSlot = indexed ? 'params.z + gid.x' : 'params.y + gid.x';

  // Cooperative load of the query into workgroup shared memory (lever 5):
  // loaded once per workgroup, reused across every row the workgroup scores.
  return /* wgsl */ `
// AUTO-GENERATED for dim=${dim}, metric=${metric}, wg=${workgroupSize}
override DIM: u32 = ${dim}u;
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>        corpus: array<f32>;
@group(0) @binding(1) var<storage, read>        query:  array<f32>;
@group(0) @binding(2) var<storage, read_write>  scores: array<f32>;
@group(0) @binding(3) var<uniform>              params: vec4<u32>; // x = count (rows this dispatch), y = score base row (chunk offset)
${candidateBinding}

var<workgroup> q_shared: array<f32, ${dim}>;

fn score_row(base: u32) -> f32 {
  ${m.init}
  var i: u32 = 0u;
  // ${vecChunks} vec4 chunks — fully unrollable because DIM is constant.
  loop {
    if (i >= ${vecChunks}u) { break; }
    let o = i * 4u;
    let c = vec4<f32>(corpus[base + o], corpus[base + o + 1u], corpus[base + o + 2u], corpus[base + o + 3u]);
    let q = vec4<f32>(q_shared[o], q_shared[o + 1u], q_shared[o + 2u], q_shared[o + 3u]);
    ${m.acc}
    i = i + 1u;
  }
${tail > 0 ? scalarTail(vecChunks * 4, tail, metric) : ''}
  ${m.finalize}
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  // 1) cooperatively stage the query vector into shared memory.
  var k: u32 = lid.x;
  loop {
    if (k >= DIM) { break; }
    q_shared[k] = query[k];
    k = k + WG;
  }
  workgroupBarrier();

  // 2) one row per invocation. In indexed mode the slot is dense (gid.x) but the
  //    scored row is looked up from the candidate list.
  if (gid.x >= params.x) { return; }
  let row = ${rowExpr};
  scores[${scoreSlot}] = score_row(row * DIM);
}
`;
}

/** Scalar remainder when dim % 4 != 0, keeping the metric math consistent. */
function scalarTail(start: number, count: number, metric: Metric): string {
  const lines: string[] = [];
  for (let j = 0; j < count; j++) {
    const idx = start + j;
    if (metric === 'l2') {
      lines.push(`  { let dd = corpus[base + ${idx}u] - q_shared[${idx}u]; acc = dd * dd + acc; }`);
    } else {
      lines.push(`  acc = corpus[base + ${idx}u] * q_shared[${idx}u] + acc;`);
    }
  }
  return lines.join('\n');
}
