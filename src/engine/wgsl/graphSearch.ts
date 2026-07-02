// GPU graph-search kernel (M7b) — CAGRA-style best-first beam search.
//
// HNSW-on-CPU walks the graph hop by hop; a naive GPU port would need one
// dispatch + readback per hop (~ms each in a browser), which is why graph ANN
// is usually written off for WebGPU. This kernel instead runs the ENTIRE beam
// search inside one dispatch: a single workgroup owns one query and keeps the
// whole search state in workgroup shared memory — the candidate beam (dist/id/
// expanded flags), a hashed visited set, and reduction scratch. Each iteration
// it (1) argmin-reduces the beam for the best unexpanded node, (2) fans the
// node's K neighbors across lanes to compute distances in parallel, and
// (3) folds improving neighbors back into the beam via argmax reductions.
// No hierarchy: like CAGRA, the search runs on the flat degree-K bottom layer
// seeded from a few well-spread entry points — the upper HNSW layers exist to
// find a good start, and E entries do that job well enough on a dense beam.
//
// One workgroup per query means batches are free: dispatchWorkgroups(nQueries)
// searches every query concurrently — where the GPU actually beats the CPU walk
// (a single query is latency-bound by dispatch+readback overhead; see hnswGpu).
//
// The kernel emits the raw beam (ef dist/id pairs, unsorted); the CPU dedups
// (the visited hash sheds entries under pressure rather than stalling, so rare
// duplicates are possible) and selects the final top-k.
//
// Uniformity notes: loop-control decisions read from shared memory go through
// workgroupUniformLoad (implicit barrier, provably uniform), and every
// reduction has a fixed trip count so barriers stay in uniform control flow.

import type { Metric } from '../../types.js';

// Beam capacity (shared-array size). Runtime ef ≤ EF via uniform; 256 keeps the
// whole state near ~12 KB, inside the 16 KB baseline workgroup-storage limit.
export const GRAPH_EF_CAP = 256;
// Visited hash slots (u32, id+1, 0 = empty). ~8 KB. Linear probing, bounded.
const HASH_SIZE = 2048;
const HASH_PROBES = 16;
// Fibonacci hashing constant (Knuth) — spreads sequential row ids well.
const HASH_MULT = '2654435761u';

const INF = '3.4e38';

export interface GraphSearchShaderKey {
  dim: number;
  /** Fixed out-degree of the graph (slots per row, empty = 0xFFFFFFFF). ≤ workgroupSize. */
  k: number;
  metric: Metric;
  workgroupSize: number;
}

export function buildGraphSearchShader(key: GraphSearchShaderKey): string {
  const { dim, k, metric, workgroupSize } = key;
  if ((workgroupSize & (workgroupSize - 1)) !== 0) {
    throw new Error(`graph-search workgroupSize must be a power of two, got ${workgroupSize}`);
  }
  if (k > workgroupSize) {
    throw new Error(`graph degree ${k} exceeds workgroupSize ${workgroupSize}`);
  }

  // Internal distance: smaller = closer (negated dot for cosine/dot, squared L2).
  const distExpr =
    metric === 'l2'
      ? `let d = corpus[base + i] - queries[qBase + i]; acc = acc + d * d;`
      : `acc = acc + corpus[base + i] * queries[qBase + i];`;
  const distReturn = metric === 'l2' ? 'return acc;' : 'return -acc;';

  return /* wgsl */ `
// AUTO-GENERATED graph beam search, dim=${dim} K=${k} metric=${metric} wg=${workgroupSize}
override WG: u32 = ${workgroupSize}u;
const DIM: u32 = ${dim}u;
const K: u32 = ${k}u;
const EF: u32 = ${GRAPH_EF_CAP}u;
const SLOTS_PER_LANE: u32 = ${Math.ceil(GRAPH_EF_CAP / workgroupSize)}u;
const HASH_MASK: u32 = ${HASH_SIZE - 1}u;
const EMPTY: u32 = 0xffffffffu;
const INF: f32 = ${INF};

@group(0) @binding(0) var<storage, read>       corpus:  array<f32>; // rows*DIM
@group(0) @binding(1) var<storage, read>       graph:   array<u32>; // rows*K, EMPTY-padded
@group(0) @binding(2) var<storage, read>       queries: array<f32>; // nQ*DIM
@group(0) @binding(3) var<storage, read>       entries: array<u32>; // entryCount seed rows
@group(0) @binding(4) var<storage, read_write> outDist: array<f32>; // nQ*EF
@group(0) @binding(5) var<storage, read_write> outId:   array<u32>; // nQ*EF
// x = ef (beam width, ≤ EF), y = iterCap (max expansions), z = entryCount
@group(0) @binding(6) var<uniform>             params:  vec4<u32>;

// The beam: candidate distances/ids and an "already expanded" flag per slot.
// Slots start at INF/expanded so they can never win the argmin until filled.
var<workgroup> candDist: array<f32, EF>;
var<workgroup> candId:   array<u32, EF>;
var<workgroup> candExp:  array<u32, EF>;
// Visited set: open-addressed hash of id+1 (0 = empty). Sheds on overflow —
// a shed id may re-enter the beam; the CPU dedups the readback.
var<workgroup> visited: array<atomic<u32>, ${HASH_SIZE}>;
// Per-expansion staging: lane j's neighbor distance/id (INF = skip).
var<workgroup> stgDist: array<f32, ${workgroupSize}>;
var<workgroup> stgId:   array<u32, ${workgroupSize}>;
// Argmin/argmax reduction scratch.
var<workgroup> redVal: array<f32, ${workgroupSize}>;
var<workgroup> redIdx: array<u32, ${workgroupSize}>;

fn distanceTo(row: u32, qBase: u32) -> f32 {
  let base = row * DIM;
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < DIM; i = i + 1u) {
    ${distExpr}
  }
  ${distReturn}
}

// Mark \`id\` visited; returns true if it was already seen. Lock-free linear
// probe with a bounded budget: under table pressure we give up and report
// "not seen" (costing a duplicate visit) rather than spinning.
fn markVisited(id: u32) -> bool {
  var h = (id * ${HASH_MULT}) & HASH_MASK;
  var probes: u32 = 0u;
  loop {
    if (probes >= ${HASH_PROBES}u) { return false; }
    let cur = atomicLoad(&visited[h]);
    if (cur == id + 1u) { return true; }
    if (cur == 0u) {
      let r = atomicCompareExchangeWeak(&visited[h], 0u, id + 1u);
      if (r.exchanged) { return false; }
      if (r.old_value == id + 1u) { return true; }
      if (r.old_value == 0u) { continue; } // spurious failure — retry this slot
    }
    h = (h + 1u) & HASH_MASK;
    probes = probes + 1u;
  }
}

// Fold the staged (dist,id) pairs 0..count into the beam: for each, argmax-find
// the worst active slot and replace it if the staged entry is better. Runs with
// all lanes (the reductions need them); lane 0 applies the swap.
fn foldStaged(count: u32, t: u32, efA: u32) {
  for (var j: u32 = 0u; j < count; j = j + 1u) {
    // Parallel argmax over the active beam.
    var worst: f32 = -INF;
    var worstIdx: u32 = 0u;
    for (var s = t; s < efA; s = s + WG) {
      if (candDist[s] > worst) { worst = candDist[s]; worstIdx = s; }
    }
    redVal[t] = worst;
    redIdx[t] = worstIdx;
    workgroupBarrier();
    var stride: u32 = WG >> 1u;
    loop {
      if (stride == 0u) { break; }
      if (t < stride) {
        if (redVal[t + stride] > redVal[t]) {
          redVal[t] = redVal[t + stride];
          redIdx[t] = redIdx[t + stride];
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    if (t == 0u) {
      if (stgDist[j] < redVal[0]) {
        let slot = redIdx[0];
        candDist[slot] = stgDist[j];
        candId[slot] = stgId[j];
        candExp[slot] = 0u;
      }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let efA = min(params.x, EF);
  let iterCap = params.y;
  let entryCount = min(params.z, WG);
  let t = lid.x;
  let qBase = wid.x * DIM;

  // Init the beam: INF + expanded, so empty slots never win the argmin.
  for (var s = t; s < EF; s = s + WG) {
    candDist[s] = INF;
    candId[s] = EMPTY;
    candExp[s] = 1u;
  }
  // (visited[] is zero-initialized per dispatch by WebGPU.)

  // Seed: stage the entry rows exactly like a neighbor batch, then fold.
  if (t < entryCount) {
    let e = entries[t];
    stgDist[t] = INF;
    if (!markVisited(e)) {
      stgDist[t] = distanceTo(e, qBase);
      stgId[t] = e;
    }
  }
  workgroupBarrier();
  foldStaged(entryCount, t, efA);

  // Beam loop: expand the best unvisited candidate until the beam is exhausted.
  var iter: u32 = 0u;
  loop {
    if (iter >= iterCap) { break; }

    // Parallel argmin over unexpanded beam slots.
    var best: f32 = INF;
    var bestIdx: u32 = 0u;
    for (var s = t; s < efA; s = s + WG) {
      if (candExp[s] == 0u && candDist[s] < best) { best = candDist[s]; bestIdx = s; }
    }
    redVal[t] = best;
    redIdx[t] = bestIdx;
    workgroupBarrier();
    var stride: u32 = WG >> 1u;
    loop {
      if (stride == 0u) { break; }
      if (t < stride) {
        if (redVal[t + stride] < redVal[t]) {
          redVal[t] = redVal[t + stride];
          redIdx[t] = redIdx[t + stride];
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    let bestDist = workgroupUniformLoad(&redVal[0]);
    if (bestDist >= INF) { break; } // nothing left to expand
    let bestSlot = workgroupUniformLoad(&redIdx[0]);
    // Termination is exhaustion: expansion clears flags and inserts only add
    // unexpanded slots when they beat the current worst, so a converged beam
    // ends up fully expanded and the INF argmin above breaks the loop. (The
    // classic "best unexpanded > worst kept" check is a no-op in a fused beam —
    // the best unexpanded IS a kept entry, so it never exceeds the beam max.)
    if (t == 0u) { candExp[bestSlot] = 1u; }
    let node = workgroupUniformLoad(&candId[bestSlot]);

    // Fan the node's K neighbor slots across lanes: visited-check + distance.
    if (t < K) {
      stgDist[t] = INF;
      let nb = graph[node * K + t];
      if (nb != EMPTY) {
        if (!markVisited(nb)) {
          stgDist[t] = distanceTo(nb, qBase);
          stgId[t] = nb;
        }
      }
    }
    workgroupBarrier();
    foldStaged(K, t, efA);

    iter = iter + 1u;
  }

  // Emit the raw beam; the CPU dedups and picks the top-k.
  let outBase = wid.x * EF;
  for (var s = t; s < EF; s = s + WG) {
    outDist[outBase + s] = candDist[s];
    outId[outBase + s] = candId[s];
  }
}
`;
}
