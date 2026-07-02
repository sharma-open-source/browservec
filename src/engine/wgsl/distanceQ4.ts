// Quantized 4-bit distance kernel.
//
// Like distanceQ8 but each coordinate is a signed 4-bit value, so 8 coords pack
// into one u32 — half the bandwidth of int8. There is no hardware unpack for
// 4-bit, so we extract nibbles by hand and sign-extend them, then dot against the
// fp32 (rotated) query staged in workgroup shared memory. Asymmetric (4-bit corpus
// × fp32 query); the caller does the exact fp32 re-rank, which matters more here
// because raw 4-bit recall is lower than int8.

export interface Quant4ShaderKey {
  paddedDim: number;
  workgroupSize: number;
  indexed?: boolean;
}

export function buildQuant4Shader(key: Quant4ShaderKey): string {
  const { paddedDim, workgroupSize, indexed = false } = key;
  if (paddedDim % 8 !== 0) throw new Error('paddedDim must be a multiple of 8 for 4-bit');
  const words = paddedDim / 8;

  const candidateBinding = indexed
    ? '@group(0) @binding(5) var<storage, read> candidates: array<u32>;'
    : '';
  const rowExpr = indexed ? 'candidates[gid.x]' : 'gid.x';
  // Chunked corpus (§NFR-10): codes local (`row`); scales at params.y + row (the
  // global row); score slot = params.y + gid.x (flat) or params.z + gid.x (IVF,
  // this chunk's slice of the candidate output). Single-buffer → params.y=z=0.
  const scoreSlot = indexed ? 'params.z + gid.x' : 'params.y + gid.x';

  return /* wgsl */ `
// AUTO-GENERATED 4-bit for paddedDim=${paddedDim}, words=${words}, wg=${workgroupSize}, indexed=${indexed}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       corpus: array<u32>;  // ${words} words/row (8 nibbles each)
@group(0) @binding(1) var<storage, read>       query:  array<vec4<f32>>;  // rotated, ${paddedDim / 4} vec4s
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read>       scales: array<f32>;  // per row
@group(0) @binding(4) var<uniform>             params: vec4<u32>;   // x = count, y = codes/scale base row, z = score output offset (indexed)
${candidateBinding}

var<workgroup> q_shared: array<vec4<f32>, ${paddedDim / 4}>;

// Extract nibble at \`shift\`, sign-extend [-8,7] (we only encode [-7,7]).
fn nib(w: u32, shift: u32) -> f32 {
  let n = (w >> shift) & 0xFu;
  return f32(i32(n) - 16 * i32(n >> 3u));
}

fn score_row(wordBase: u32, scale: f32) -> f32 {
  var acc: f32 = 0.0;
  var g: u32 = 0u;
  loop {
    if (g >= ${words}u) { break; }
    let w = corpus[wordBase + g];
    let o = g * 2u;
    let v0 = vec4<f32>(nib(w, 0u), nib(w, 4u), nib(w, 8u), nib(w, 12u));
    let v1 = vec4<f32>(nib(w, 16u), nib(w, 20u), nib(w, 24u), nib(w, 28u));
    acc = dot(v0, q_shared[o]) + dot(v1, q_shared[o + 1u]) + acc;
    g = g + 1u;
  }
  return acc * (scale * (1.0 / 7.0)); // per-row scale factored out of the loop
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  var k: u32 = lid.x;
  loop {
    if (k >= ${paddedDim / 4}u) { break; }
    q_shared[k] = query[k];
    k = k + WG;
  }
  workgroupBarrier();

  if (gid.x >= params.x) { return; }
  let row = ${rowExpr}; // local row within the bound codes chunk
  scores[${scoreSlot}] = score_row(row * ${words}u, scales[params.y + row]);
}
`;
}
