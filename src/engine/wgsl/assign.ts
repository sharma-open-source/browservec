// Centroid-assignment kernel for IVF build.
//
// One invocation per corpus row: scan all `nlist` centroids and emit the id of
// the nearest one. For cosine/dot (the only IVF metrics) "nearest" = argmax dot,
// matching the search metric, so assignment partitions the corpus consistently.
// This runs once at build time over the whole corpus; the per-query scan only
// touches the few probed lists.

export interface AssignShaderKey {
  dim: number;
  workgroupSize: number;
}

export function buildAssignShader(key: AssignShaderKey): string {
  const { dim, workgroupSize } = key;
  const vecChunks = Math.floor(dim / 4);
  const tail = dim % 4;

  return /* wgsl */ `
// AUTO-GENERATED assignment kernel for dim=${dim}, wg=${workgroupSize}
override DIM: u32 = ${dim}u;
override WG: u32 = ${workgroupSize}u;

@group(0) @binding(0) var<storage, read>        corpus:    array<f32>;
@group(0) @binding(1) var<storage, read>        centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write>  cluster:   array<u32>;
@group(0) @binding(3) var<uniform>              params:    vec4<u32>; // x = rows, y = nlist

fn dot_centroid(row_base: u32, cen_base: u32) -> f32 {
  var acc: f32 = 0.0;
  var i: u32 = 0u;
  loop {
    if (i >= ${vecChunks}u) { break; }
    let o = i * 4u;
    let c = vec4<f32>(corpus[row_base + o], corpus[row_base + o + 1u], corpus[row_base + o + 2u], corpus[row_base + o + 3u]);
    let m = vec4<f32>(centroids[cen_base + o], centroids[cen_base + o + 1u], centroids[cen_base + o + 2u], centroids[cen_base + o + 3u]);
    acc = dot(c, m) + acc;
    i = i + 1u;
  }
${tail > 0 ? scalarTail(vecChunks * 4, tail) : ''}
  return acc;
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.x) { return; }
  let row_base = row * DIM;

  var best_c: u32 = 0u;
  var best_s: f32 = -3.4e38;
  var c: u32 = 0u;
  loop {
    if (c >= params.y) { break; }
    let s = dot_centroid(row_base, c * DIM);
    if (s > best_s) { best_s = s; best_c = c; }
    c = c + 1u;
  }
  cluster[row] = best_c;
}
`;
}

function scalarTail(start: number, count: number): string {
  const lines: string[] = [];
  for (let j = 0; j < count; j++) {
    const idx = start + j;
    lines.push(`  acc = corpus[row_base + ${idx}u] * centroids[cen_base + ${idx}u] + acc;`);
  }
  return lines.join('\n');
}
