// Score-mask kernel (FR-7 in-index filtering, flat-family paths).
//
// The distance kernels leave one f32 score per corpus row in a dense storage
// buffer; this pass overwrites the score of every row whose mask bit is clear
// with the same -FLT_MAX sentinel the top-k reduction already treats as "not a
// result". Running it between the distance dispatch and the top-k turns the
// whole flat pipeline into a pre-filtered search: masked rows can never win a
// top-k slot, so no over-fetch is needed and GPU top-k stays on its small-k
// fast path. The mask is one bit per global row (LSB-first within each u32),
// built CPU-side from the metadata predicate.

// Must sort below any real score and match the top-k merge's filler comparison
// (see src/index/gpuTopk.ts NEG_MAX / src/engine/wgsl/topk.ts).
const NEG_MAX = '-3.4e38';

export interface MaskShaderKey {
  workgroupSize: number;
}

export function buildMaskShader(key: MaskShaderKey): string {
  return /* wgsl */ `
// AUTO-GENERATED score-mask pass, wg=${key.workgroupSize}
@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read>       mask:   array<u32>; // 1 bit/row, LSB-first
@group(0) @binding(2) var<uniform>             params: vec4<u32>;  // x = n

@compute @workgroup_size(${key.workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.x) { return; }
  if (((mask[i >> 5u] >> (i & 31u)) & 1u) == 0u) {
    scores[i] = f32(${NEG_MAX});
  }
}
`;
}
