// Quantized (int8) distance kernel — (dequant in-shader)
// and §14.2 lever 1 (quantized data = less bandwidth).
//
// Asymmetric: the corpus is stored as packed int8 snorm codes (4 per u32) and a
// per-row scale; the query stays fp32 (rotated) in workgroup shared memory. Each
// `unpack4x8snorm` dequantizes 4 coordinates in one hardware instruction. Only
// dot/cosine are supported in quantized mode (the inner-product case TurboQuant
// targets); fp32 flat handles l2.
//
// PDIM (padded dim) and WORDS (=PDIM/4) are baked as literals so the inner loop
// fully unrolls and the shared-memory array has a constant size.

export interface QuantShaderKey {
  paddedDim: number;
  workgroupSize: number;
  /**
   * Indexed mode (IVF×int8, §9 M4): score the row at `candidates[gid]` instead of
   * row == gid. Adds binding(5); params.x becomes the candidate count. Lets one
   * dispatch score only the probed cluster lists.
   */
  indexed?: boolean;
}

export function buildQuantShader(key: QuantShaderKey): string {
  const { paddedDim, workgroupSize, indexed = false } = key;
  if (paddedDim % 4 !== 0) throw new Error('paddedDim must be a multiple of 4');
  const words = paddedDim / 4;

  const candidateBinding = indexed
    ? '@group(0) @binding(5) var<storage, read> candidates: array<u32>;'
    : '';
  const rowExpr = indexed ? 'candidates[gid.x]' : 'gid.x';
  // Chunked corpus (§NFR-10). `corpus` holds one chunk of codes addressed by the
  // *local* row (`row`), while `scores`/`scales` are single global buffers:
  //   scale index = params.y (chunk base) + row  →  the global row
  //   score slot  = flat: params.y + gid.x (contiguous rows)
  //                 IVF:  params.z + gid.x (this chunk's slice of the candidate
  //                       output; candidates carry *local* ids, params.z = offset)
  // Single-buffer stores pass params.y = params.z = 0, so both reduce to today's
  // behaviour (flat: scores[gid.x]; IVF: candidates are global, base 0).
  const scoreSlot = indexed ? 'params.z + gid.x' : 'params.y + gid.x';

  return /* wgsl */ `
// AUTO-GENERATED for paddedDim=${paddedDim}, words=${words}, wg=${workgroupSize}, indexed=${indexed}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       corpus: array<u32>;  // ${words} words/row
@group(0) @binding(1) var<storage, read>       query:  array<f32>;  // rotated, ${paddedDim} long
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read>       scales: array<f32>;  // per row
@group(0) @binding(4) var<uniform>             params: vec4<u32>;   // x = count, y = codes/scale base row, z = score output offset (indexed)
${candidateBinding}

var<workgroup> q_shared: array<f32, ${paddedDim}>;

fn score_row(wordBase: u32, scale: f32) -> f32 {
  var acc: f32 = 0.0;
  var g: u32 = 0u;
  loop {
    if (g >= ${words}u) { break; }
    let v = unpack4x8snorm(corpus[wordBase + g]) * scale; // 4 coords dequantized
    let o = g * 4u;
    let q = vec4<f32>(q_shared[o], q_shared[o + 1u], q_shared[o + 2u], q_shared[o + 3u]);
    acc = dot(v, q) + acc;
    g = g + 1u;
  }
  return acc;
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  var k: u32 = lid.x;
  loop {
    if (k >= ${paddedDim}u) { break; }
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
