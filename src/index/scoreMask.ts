// Score-mask driver (FR-7 in-index filtering).
//
// Owns the compiled mask pipeline and a pooled GPU-side bitset buffer. Between
// a distance dispatch and the top-k, `encode()` uploads the per-row match
// bitset and stamps the -FLT_MAX sentinel over every masked-out row's score,
// so the unchanged top-k machinery (GPU reduction or CPU select) only ever
// surfaces rows that satisfy the metadata filter. Upload cost is n/8 bytes per
// filtered query (~125 KB at 1M rows).

import type { DeviceContext } from '../engine/device.js';
import { buildMaskShader } from '../engine/wgsl/mask.js';

const WORKGROUP_SIZE = 256;

/**
 * Any readback score at or below this is a masked-out row, not a real result
 * (real scores — dot products or negative squared distances of finite vectors —
 * are nowhere near -1e38). Kept slightly above the shader's -3.4e38 sentinel so
 * f32 rounding can't leak a masked row through the comparison.
 */
export const MASKED_SCORE = -3.39e38;

export class ScoreMask {
  private readonly pipeline: GPUComputePipeline;
  private readonly paramsBuf: GPUBuffer;
  private maskBuf: GPUBuffer | null = null;
  private maskCapWords = 0;
  private readonly paramsScratch = new Uint32Array(4);

  constructor(private readonly ctx: DeviceContext) {
    const { device } = ctx;
    const module = device.createShaderModule({
      label: 'browservec:mask',
      code: buildMaskShader({ workgroupSize: WORKGROUP_SIZE }),
    });
    this.pipeline = device.createComputePipeline({
      label: 'browservec:mask',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.paramsBuf = device.createBuffer({
      label: 'browservec:mask-params',
      size: 16, // vec4<u32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private ensureCapacity(words: number): void {
    if (this.maskBuf && this.maskCapWords >= words) return;
    this.maskBuf?.destroy();
    const cap = Math.max(words, this.maskCapWords * 2, 256);
    this.maskBuf = this.ctx.device.createBuffer({
      label: 'browservec:mask-bits',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.maskCapWords = cap;
  }

  /**
   * Upload `maskWords` (ceil(n/32) u32s, bit set = row passes the filter) and
   * stamp the sentinel over all masked-out slots of the dense `scores` buffer.
   * Enqueued on the device queue — ordered after the distance dispatches and
   * before the top-k submission, no extra sync needed.
   */
  apply(scores: GPUBuffer, n: number, maskWords: Uint32Array): void {
    const { device } = this.ctx;
    const words = Math.ceil(n / 32);
    if (maskWords.length < words) {
      throw new Error(`mask has ${maskWords.length} words, need ${words} for ${n} rows`);
    }
    this.ensureCapacity(words);
    device.queue.writeBuffer(this.maskBuf!, 0, maskWords, 0, words);
    this.paramsScratch[0] = n;
    device.queue.writeBuffer(this.paramsBuf, 0, this.paramsScratch);

    const bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: scores } },
        { binding: 1, resource: { buffer: this.maskBuf! } },
        { binding: 2, resource: { buffer: this.paramsBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.paramsBuf.destroy();
    this.maskBuf?.destroy();
  }
}
