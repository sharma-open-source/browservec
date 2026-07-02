// GPU top-k reduction kernel (lever 3: on-GPU top-k).
//
// The distance kernels leave one f32 score per corpus row in a dense storage
// buffer. In the whole array was copied back and sorted on the CPU — an O(N)
// transfer + O(N·k) sort per query that dominates latency past ~10⁵ rows. This
// kernel reduces that on the GPU: one workgroup owns a contiguous segment of WG
// scores and extracts its local top-k by `k` rounds of parallel argmax in shared
// memory (each round finds the segment max, records it, then invalidates that
// slot). Each workgroup emits k (score,row) pairs, so the readback shrinks from
// N floats to ceil(N/WG)·k pairs — the CPU only merges that short list.
//
// WG must be a power of two (the reduction halves a stride each step). `k` is a
// uniform, not a baked constant, so a single compiled pipeline serves every k.

// A large negative sentinel that sorts below any real score. Kept safely under
// FLT_MAX (3.40282347e38) because a literal at/over it fails f32 parsing in WGSL.
const NEG_MAX = '-3.4e38';

export interface TopKShaderKey {
  /** Workgroup size == segment size == shared-array length. Power of two. */
  workgroupSize: number;
}

export function buildTopKShader(key: TopKShaderKey): string {
  const { workgroupSize } = key;
  if ((workgroupSize & (workgroupSize - 1)) !== 0) {
    throw new Error(`topk workgroupSize must be a power of two, got ${workgroupSize}`);
  }

  return /* wgsl */ `
// AUTO-GENERATED GPU top-k reduction, wg=${workgroupSize}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       scores:       array<f32>;
@group(0) @binding(1) var<storage, read_write> partialScore: array<f32>;
@group(0) @binding(2) var<storage, read_write> partialRow:   array<u32>;
@group(0) @binding(3) var<uniform>             params:       vec4<u32>; // x = n, y = k

// The segment's live scores (winners get overwritten with -FLT_MAX each round).
var<workgroup> seg: array<f32, ${workgroupSize}>;
// Scratch for the argmax tree: value + originating lane, so we can invalidate the
// winner at its *original* lane after the reduction collapses it to lane 0.
var<workgroup> redVal:  array<f32, ${workgroupSize}>;
var<workgroup> redLane: array<u32, ${workgroupSize}>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = params.x;
  let k = params.y;
  let g = wid.x;
  let base = g * WG;
  let t = lid.x;
  let gi = base + t;

  seg[t] = select(f32(${NEG_MAX}), scores[gi], gi < n);
  workgroupBarrier();

  for (var kk: u32 = 0u; kk < k; kk = kk + 1u) {
    redVal[t] = seg[t];
    redLane[t] = t;
    workgroupBarrier();

    var stride: u32 = WG >> 1u;
    loop {
      if (stride == 0u) { break; }
      if (t < stride) {
        if (redVal[t + stride] > redVal[t]) {
          redVal[t] = redVal[t + stride];
          redLane[t] = redLane[t + stride];
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }

    if (t == 0u) {
      let lane = redLane[0];
      let out = g * k + kk;
      partialScore[out] = seg[lane];
      partialRow[out] = base + lane;
      seg[lane] = f32(${NEG_MAX});
    }
    workgroupBarrier();
  }
}
`;
}
