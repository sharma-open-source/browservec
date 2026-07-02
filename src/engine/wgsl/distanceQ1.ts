// Quantized 1-bit (binary) distance kernel.
//
// The extreme of the sub-byte ladder: each rotated coordinate is stored as a
// single sign bit, so 32 coords pack into one u32 (~32× less than fp32). A row
// reconstructs as `sign_i * scale` with `scale = mean|coord|`, so scoring is
// asymmetric (±1 corpus × fp32 query): we accumulate Σ sign_i·q_i against the
// fp32 (rotated) query staged in workgroup shared memory, then multiply by the
// row scale once. Binary is the coarsest code, so the caller's exact fp32 re-rank
// (with a wide over-fetch) does the heavy lifting for recall.

export interface Quant1ShaderKey {
  paddedDim: number;
  workgroupSize: number;
  indexed?: boolean;
}

export function buildQuant1Shader(key: Quant1ShaderKey): string {
  const { paddedDim, workgroupSize, indexed = false } = key;
  if (paddedDim % 32 !== 0) throw new Error('paddedDim must be a multiple of 32 for 1-bit');
  const words = paddedDim / 32;

  const candidateBinding = indexed
    ? '@group(0) @binding(5) var<storage, read> candidates: array<u32>;'
    : '';
  const rowExpr = indexed ? 'candidates[gid.x]' : 'gid.x';
  // Chunked corpus (§NFR-10): codes local (`row`); scales at params.y + row (the
  // global row); score slot = params.y + gid.x (flat) or params.z + gid.x (IVF).
  const scoreSlot = indexed ? 'params.z + gid.x' : 'params.y + gid.x';

  return /* wgsl */ `
// AUTO-GENERATED 1-bit for paddedDim=${paddedDim}, words=${words}, wg=${workgroupSize}, indexed=${indexed}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       corpus: array<u32>;  // ${words} words/row (32 sign bits each)
@group(0) @binding(1) var<storage, read>       query:  array<vec4<f32>>;  // rotated, ${paddedDim / 4} vec4s
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read>       scales: array<f32>;  // per row (mean|coord|)
@group(0) @binding(4) var<uniform>             params: vec4<u32>;   // x = count, y = codes/scale base row, z = score output offset (indexed)
${candidateBinding}

var<workgroup> q_shared: array<vec4<f32>, ${paddedDim / 4}>;

// Sign of bit \`shift\` in \`w\`: +1.0 when set, -1.0 when clear.
fn sgn(w: u32, shift: u32) -> f32 {
  return select(-1.0, 1.0, ((w >> shift) & 1u) == 1u);
}

fn score_row(wordBase: u32, scale: f32) -> f32 {
  var acc: f32 = 0.0;
  var g: u32 = 0u;
  loop {
    if (g >= ${words}u) { break; }
    let w = corpus[wordBase + g];
    let o = g * 8u;
    // 32 sign bits → eight vec4 dot products against the staged fp32 query.
    var s: u32 = 0u;
    loop {
      if (s >= 8u) { break; }
      let sh = s * 4u;
      let v = vec4<f32>(sgn(w, sh), sgn(w, sh + 1u), sgn(w, sh + 2u), sgn(w, sh + 3u));
      acc = dot(v, q_shared[o + s]) + acc;
      s = s + 1u;
    }
    g = g + 1u;
  }
  return acc * scale;
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
