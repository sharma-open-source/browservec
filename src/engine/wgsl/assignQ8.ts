// Quantized centroid-assignment kernel for the IVF×int8 combo .
//
// Like assign.ts but the corpus is stored as int8 codes (rotated space): each row
// is dequantized with `unpack4x8snorm` (4 coords/instruction) and dotted against
// every fp32 centroid to find its nearest cluster. Assigning rows by their
// *quantized* representation matches how they are scored at query time. Centroids
// live in the rotated space and are `paddedDim` long.

export interface QuantAssignShaderKey {
  paddedDim: number;
  workgroupSize: number;
}

export function buildQuantAssignShader(key: QuantAssignShaderKey): string {
  const { paddedDim, workgroupSize } = key;
  if (paddedDim % 4 !== 0) throw new Error('paddedDim must be a multiple of 4');
  const words = paddedDim / 4;

  return /* wgsl */ `
// AUTO-GENERATED quant assignment for paddedDim=${paddedDim}, words=${words}, wg=${workgroupSize}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       corpus:    array<u32>;  // ${words} words/row
@group(0) @binding(1) var<storage, read>       centroids: array<f32>;  // nlist * ${paddedDim}
@group(0) @binding(2) var<storage, read_write> cluster:   array<u32>;
@group(0) @binding(3) var<storage, read>       scales:    array<f32>;  // per row
@group(0) @binding(4) var<uniform>             params:    vec4<u32>;   // x = rows (this chunk), y = nlist, z = base row (chunk offset)

fn dot_centroid(wordBase: u32, scale: f32, cenBase: u32) -> f32 {
  var acc: f32 = 0.0;
  var g: u32 = 0u;
  loop {
    if (g >= ${words}u) { break; }
    let v = unpack4x8snorm(corpus[wordBase + g]) * scale;
    let o = g * 4u;
    let m = vec4<f32>(centroids[cenBase + o], centroids[cenBase + o + 1u], centroids[cenBase + o + 2u], centroids[cenBase + o + 3u]);
    acc = dot(v, m) + acc;
    g = g + 1u;
  }
  return acc;
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Codes are addressed locally within the bound chunk (gid.x); cluster + scales
  // are single global buffers indexed by the global row params.z + gid.x. Single-
  // buffer stores pass params.z = 0, so this is a no-op there (§NFR-10).
  if (gid.x >= params.x) { return; }
  let wordBase = gid.x * ${words}u;
  let global = params.z + gid.x;
  let scale = scales[global];

  var best_c: u32 = 0u;
  var best_s: f32 = -3.4e38;
  var c: u32 = 0u;
  loop {
    if (c >= params.y) { break; }
    let s = dot_centroid(wordBase, scale, c * ${paddedDim}u);
    if (s > best_s) { best_s = s; best_c = c; }
    c = c + 1u;
  }
  cluster[global] = best_c;
}
`;
}
