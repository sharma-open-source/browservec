// 1-bit (binary) quantized centroid-assignment kernel for the 1-bit × IVF combo
// (M3b + M4).
//
// Like assignQ4 but the corpus is binary: 32 sign bits per u32, expanded to ±1 and
// scaled by the per-row `mean|coord|`, then dotted against each fp32 centroid to
// find the nearest cluster. Rows are assigned by their binary representation — the
// same one they're scored with at query time — so assignment and scan agree.

export interface Quant1AssignShaderKey {
  paddedDim: number;
  workgroupSize: number;
}

export function buildQuant1AssignShader(key: Quant1AssignShaderKey): string {
  const { paddedDim, workgroupSize } = key;
  if (paddedDim % 32 !== 0) throw new Error('paddedDim must be a multiple of 32 for 1-bit');
  const words = paddedDim / 32;

  return /* wgsl */ `
// AUTO-GENERATED 1-bit assignment for paddedDim=${paddedDim}, words=${words}, wg=${workgroupSize}
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>       corpus:    array<u32>;  // ${words} words/row (32 sign bits each)
@group(0) @binding(1) var<storage, read>       centroids: array<f32>;  // nlist * ${paddedDim}
@group(0) @binding(2) var<storage, read_write> cluster:   array<u32>;
@group(0) @binding(3) var<storage, read>       scales:    array<f32>;  // per row (mean|coord|)
@group(0) @binding(4) var<uniform>             params:    vec4<u32>;   // x = rows (this chunk), y = nlist, z = base row (chunk offset)

fn sgn(w: u32, shift: u32) -> f32 {
  return select(-1.0, 1.0, ((w >> shift) & 1u) == 1u);
}

fn dot_centroid(wordBase: u32, scale: f32, cenBase: u32) -> f32 {
  var acc: f32 = 0.0;
  var g: u32 = 0u;
  loop {
    if (g >= ${words}u) { break; }
    let w = corpus[wordBase + g];
    let o = g * 32u;
    var s: u32 = 0u;
    loop {
      if (s >= 8u) { break; }
      let sh = s * 4u;
      let base = o + sh;
      let v = vec4<f32>(sgn(w, sh), sgn(w, sh + 1u), sgn(w, sh + 2u), sgn(w, sh + 3u));
      let m = vec4<f32>(centroids[cenBase + base], centroids[cenBase + base + 1u], centroids[cenBase + base + 2u], centroids[cenBase + base + 3u]);
      acc = dot(v, m) + acc;
      s = s + 1u;
    }
    g = g + 1u;
  }
  return acc * scale;
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
