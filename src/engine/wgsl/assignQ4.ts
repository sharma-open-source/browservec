// 4-bit quantized centroid-assignment kernel for the int4 × IVF combo (M3b + M4).
//
// Like assignQ8 but the corpus is 4-bit: 8 signed nibbles per u32, extracted and
// sign-extended by hand (no hardware unpack), dotted against each fp32 centroid to
// find the nearest cluster. Rows are assigned by their int4 representation — the
// same one they're scored with at query time.

export interface Quant4AssignShaderKey {
  paddedDim: number;
  workgroupSize: number;
}

export function buildQuant4AssignShader(key: Quant4AssignShaderKey): string {
  const { paddedDim, workgroupSize } = key;
  if (paddedDim % 8 !== 0) throw new Error('paddedDim must be a multiple of 8 for 4-bit');
  const words = paddedDim / 8;

  return /* wgsl */ `
// AUTO-GENERATED 4-bit assignment for paddedDim=${paddedDim}, words=${words}, wg=${workgroupSize}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       corpus:    array<u32>;  // ${words} words/row
@group(0) @binding(1) var<storage, read>       centroids: array<f32>;  // nlist * ${paddedDim}
@group(0) @binding(2) var<storage, read_write> cluster:   array<u32>;
@group(0) @binding(3) var<storage, read>       scales:    array<f32>;  // per row
@group(0) @binding(4) var<uniform>             params:    vec4<u32>;   // x = rows (this chunk), y = nlist, z = base row (chunk offset)

fn nib(w: u32, shift: u32) -> f32 {
  let n = (w >> shift) & 0xFu;
  return f32(i32(n) - 16 * i32(n >> 3u));
}

fn dot_centroid(wordBase: u32, scale: f32, cenBase: u32) -> f32 {
  var acc: f32 = 0.0;
  let inv7 = scale * (1.0 / 7.0);
  var g: u32 = 0u;
  loop {
    if (g >= ${words}u) { break; }
    let w = corpus[wordBase + g];
    let o = g * 8u;
    let v0 = vec4<f32>(nib(w, 0u), nib(w, 4u), nib(w, 8u), nib(w, 12u)) * inv7;
    let v1 = vec4<f32>(nib(w, 16u), nib(w, 20u), nib(w, 24u), nib(w, 28u)) * inv7;
    let m0 = vec4<f32>(centroids[cenBase + o], centroids[cenBase + o + 1u], centroids[cenBase + o + 2u], centroids[cenBase + o + 3u]);
    let m1 = vec4<f32>(centroids[cenBase + o + 4u], centroids[cenBase + o + 5u], centroids[cenBase + o + 6u], centroids[cenBase + o + 7u]);
    acc = dot(v0, m0) + dot(v1, m1) + acc;
    g = g + 1u;
  }
  return acc;
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Codes local to the bound chunk (gid.x); cluster + scales global at
  // params.z + gid.x. params.z = 0 for single-buffer stores (§NFR-10).
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
