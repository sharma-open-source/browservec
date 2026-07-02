// GPU top-k driver (REQUIREMENTS.md §14.2 lever 3).
//
// Wraps the reduction kernel in src/engine/wgsl/topk.ts: given a populated dense
// scores buffer, it dispatches one workgroup per WG-score segment, reads back the
// short partials list (ceil(n/WG)·k pairs instead of n floats), and finishes the
// merge on the CPU. Buffers are pooled and grown geometrically across queries.
//
// Below GPU_TOPK_MIN_ROWS the full-readback CPU path is cheaper (kernel dispatch +
// two-buffer round-trip isn't worth it), so FlatIndex only reaches for this on
// large corpora — exactly where the O(N) readback hurts.

import type { DeviceContext } from '../engine/device.js';
import { buildTopKShader } from '../engine/wgsl/topk.js';
import type { FlatHit } from './flat.js';

/** WG must be a power of two (reduction tree) and match the shader's segment size. */
export const TOPK_WG = 256;

/**
 * Corpora smaller than this stay on the CPU full-readback path. The GPU reduction
 * only pays off once the score readback + CPU sort dominate, which is well past a
 * few thousand rows.
 */
export const GPU_TOPK_MIN_ROWS = 4096;

// Must match the shader's filler sentinel (src/engine/wgsl/topk.ts). Under-full
// workgroups emit this for empty slots; the merge drops anything <= it.
const NEG_MAX = -3.4e38;

export class GpuTopK {
  private readonly pipeline: GPUComputePipeline;
  private readonly paramsBuf: GPUBuffer;
  private partialScore: GPUBuffer | null = null;
  private partialRow: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private capacityPairs = 0;

  constructor(private readonly ctx: DeviceContext) {
    const { device } = ctx;
    const module = device.createShaderModule({
      label: 'browservec:topk',
      code: buildTopKShader({ workgroupSize: TOPK_WG }),
    });
    this.pipeline = device.createComputePipeline({
      label: 'browservec:topk',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.paramsBuf = device.createBuffer({
      label: 'browservec:topk-params',
      size: 16, // vec4<u32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Whether GPU top-k is worth it for (n rows, k neighbors). True only once the
   * corpus is large enough (GPU_TOPK_MIN_ROWS) *and* the partials readback
   * (ceil(n/WG)·k pairs, two floats each) is actually smaller than reading all n
   * scores back. The second test matters on the quantized re-rank path, where k is
   * over-fetched (k·rerankFactor); a large enough k makes the partials list exceed
   * the full-score list, so we keep the plain readback there.
   */
  static beneficial(n: number, k: number): boolean {
    if (n < GPU_TOPK_MIN_ROWS) return false;
    const groups = Math.ceil(n / TOPK_WG);
    return groups * k * 2 < n;
  }

  private ensureCapacity(pairs: number): void {
    if (this.partialScore && this.capacityPairs >= pairs) return;
    this.partialScore?.destroy();
    this.partialRow?.destroy();
    this.readback?.destroy();
    const { device } = this.ctx;
    // score (f32) + row (u32), plus one readback buffer covering both packed
    // regions: [scores | rows], each `pairs` * 4 bytes.
    const region = pairs * 4;
    this.partialScore = device.createBuffer({
      label: 'browservec:topk-partial-score',
      size: region,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.partialRow = device.createBuffer({
      label: 'browservec:topk-partial-row',
      size: region,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.readback = device.createBuffer({
      label: 'browservec:topk-readback',
      size: region * 2,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.capacityPairs = pairs;
  }

  /**
   * Reduce a dense `scores` buffer (n live rows) to the top-k on the GPU.
   * `scores` must have GPUBufferUsage.STORAGE; the caller keeps ownership.
   */
  async query(scores: GPUBuffer, n: number, k: number): Promise<FlatHit[]> {
    const { device } = this.ctx;
    const groups = Math.ceil(n / TOPK_WG);
    const pairs = groups * k;
    this.ensureCapacity(pairs);
    const partialScore = this.partialScore!;
    const partialRow = this.partialRow!;
    const readback = this.readback!;

    device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([n, k, 0, 0]));
    const bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: scores } },
        { binding: 1, resource: { buffer: partialScore } },
        { binding: 2, resource: { buffer: partialRow } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    const region = pairs * 4;
    enc.copyBufferToBuffer(partialScore, 0, readback, 0, region);
    enc.copyBufferToBuffer(partialRow, 0, readback, region, region);
    device.queue.submit([enc.finish()]);

    await readback.mapAsync(GPUMapMode.READ, 0, region * 2);
    const scoreView = new Float32Array(readback.getMappedRange(0, region).slice(0));
    const rowView = new Uint32Array(readback.getMappedRange(region, region).slice(0));
    readback.unmap();

    return mergePartials(scoreView, rowView, pairs, n, k);
  }

  destroy(): void {
    this.paramsBuf.destroy();
    this.partialScore?.destroy();
    this.partialRow?.destroy();
    this.readback?.destroy();
  }
}

/**
 * Final CPU merge over the GPU partials — a short list (ceil(n/WG)·k) versus the
 * full n scores. Same higher-is-closer selection as {@link topK}; invalid slots
 * (a workgroup with fewer than k live rows emits -FLT_MAX fillers) are skipped.
 */
export function mergePartials(
  scores: Float32Array,
  rows: Uint32Array,
  count: number,
  n: number,
  k: number,
): FlatHit[] {
  const limit = Math.min(k, n);
  const heap: FlatHit[] = [];
  for (let i = 0; i < count; i++) {
    const score = scores[i]!;
    if (score <= NEG_MAX) continue; // filler from an under-full workgroup
    const row = rows[i]!;
    if (row >= n) continue;
    if (heap.length < limit) {
      heap.push({ row, score });
      if (heap.length === limit) heap.sort((a, b) => a.score - b.score); // ascending; [0] = min
    } else if (score > heap[0]!.score) {
      heap[0] = { row, score };
      let j = 0;
      while (j + 1 < heap.length && heap[j]!.score > heap[j + 1]!.score) {
        const t = heap[j]!;
        heap[j] = heap[j + 1]!;
        heap[j + 1] = t;
        j++;
      }
    }
  }
  return heap.sort((a, b) => b.score - a.score); // descending for output
}
