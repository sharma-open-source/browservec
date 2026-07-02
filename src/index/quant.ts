// Quantized flat GPU index — TurboQuant int8 path.
//
// Stores the corpus as rotated + int8-quantized codes (4× less memory than fp32)
// plus a per-row scale. Queries are rotated the same way and run asymmetrically
// (fp32 query × int8 corpus) via the unpack4x8snorm kernel. Returns approximate
// top-(k·overfetch); the caller does an exact fp32 re-rank to recover recall.

import type { DeviceContext } from '../engine/device.js';
import { ChunkedCorpus } from '../engine/buffers.js';
import { buildQuantShader } from '../engine/wgsl/distanceQ8.js';
import { buildQuant4Shader } from '../engine/wgsl/distanceQ4.js';
import { buildQuant1Shader } from '../engine/wgsl/distanceQ1.js';
import { createQuantEncoder, type QuantEncoder, type IngestMode } from '../quant/encoder.js';
import { topK, type FlatHit } from './flat.js';
import { GpuTopK } from './gpuTopk.js';

const WORKGROUP_SIZE = 64;

export class QuantIndex {
  private readonly encoder: QuantEncoder;
  private readonly paddedDim: number;
  private readonly wordsPerRow: number;

  // Codes are chunked across GPU buffers (§NFR-10); scales stay in one small
  // global buffer (4 B/row — never near the binding limit) indexed by global row.
  private readonly codes: ChunkedCorpus;
  private scales: GPUBuffer | null = null;
  private scalesCap = 0;
  private rows = 0;

  private readonly queryBuf: GPUBuffer;
  private readonly paramsBuf: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private scoreBuf: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private scoreCap = 0;
  private gpuTopk: GpuTopK | null = null;

  constructor(
    private readonly ctx: DeviceContext,
    readonly dim: number,
    seed: number,
    rounds: number,
    readonly bits: 1 | 4 | 8 = 8,
    forcedRowsPerChunk?: number,
  ) {
    // Rotate + quantize runs through the encoder — off-thread in a Worker when
    // available (§NFR-8), else in-thread. int8 4 coords/word, int4 8, 1-bit 32.
    this.encoder = createQuantEncoder({ dim, seed, rounds, bits });
    this.paddedDim = this.encoder.paddedDim;
    this.wordsPerRow = this.encoder.wordsPerRow;

    const { device } = ctx;
    this.codes = new ChunkedCorpus(
      device,
      this.wordsPerRow,
      4,
      ctx.limits.maxStorageBufferBindingSize,
      forcedRowsPerChunk,
    );
    this.queryBuf = device.createBuffer({
      label: 'browservec:q-query',
      size: this.paddedDim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.paramsBuf = device.createBuffer({
      label: 'browservec:q-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const code =
      bits === 8
        ? buildQuantShader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE })
        : bits === 4
          ? buildQuant4Shader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE })
          : buildQuant1Shader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE });
    const module = device.createShaderModule({ label: `browservec:distanceQ${bits}`, code });
    this.pipeline = device.createComputePipeline({
      label: `browservec:quant${bits}`,
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  get size(): number {
    return this.rows;
  }

  /** Grow the single global scales buffer (4 B/row) geometrically, copying on growth. */
  private ensureScalesCapacity(rows: number): void {
    if (rows <= this.scalesCap) return;
    const next = Math.max(rows, this.scalesCap * 2, 4096);
    const { device } = this.ctx;
    const newScales = device.createBuffer({
      label: 'browservec:q-scales',
      size: next * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    if (this.scales && this.rows > 0) {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(this.scales, 0, newScales, 0, this.rows * 4);
      device.queue.submit([enc.finish()]);
    }
    this.scales?.destroy();
    this.scales = newScales;
    this.scalesCap = next;
  }

  /** Which path ingest used: 'worker' (off-thread), 'main-thread' (fallback), or 'pending'. */
  get ingestMode(): IngestMode {
    return this.encoder.mode();
  }

  /** Number of GPU buffers the codes span (§NFR-10). 1 until they overflow one buffer. */
  get chunkCount(): number {
    return this.codes.chunkCount;
  }

  /**
   * Rotate + quantize `count` rows packed in `data` (count*dim floats) and upload.
   * The rotate+quantize runs through the encoder (off-thread when a Worker is
   * available), so `append` is async — callers must await appends in order.
   */
  async append(data: Float32Array, count: number): Promise<void> {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    const base = this.rows;
    this.ensureScalesCapacity(base + count);

    const { words, scales } = await this.encoder.encode(data, count, false);

    this.codes.append(words, count); // splits across chunk buffers as needed
    this.ctx.device.queue.writeBuffer(this.scales!, base * 4, scales);
    this.rows = base + count;
  }

  private ensureScores(): { score: GPUBuffer; readback: GPUBuffer } {
    if (this.scoreBuf && this.readback && this.scoreCap >= this.rows) {
      return { score: this.scoreBuf, readback: this.readback };
    }
    this.scoreBuf?.destroy();
    this.readback?.destroy();
    const size = Math.max(this.rows, 1) * 4;
    this.scoreBuf = this.ctx.device.createBuffer({
      label: 'browservec:q-scores',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.readback = this.ctx.device.createBuffer({
      label: 'browservec:q-readback',
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.scoreCap = this.rows;
    return { score: this.scoreBuf, readback: this.readback };
  }

  /** Approximate top-k by quantized score (rotates the query internally). */
  async query(queryVec: Float32Array, k: number): Promise<FlatHit[]> {
    if (this.ctx.lost) throw new Error('GPU device lost; re-create the store');
    if (this.rows === 0) return [];
    const { device } = this.ctx;

    const rq = this.encoder.rotateQuery(queryVec); // fp32 rotated query (asymmetric)
    device.queue.writeBuffer(this.queryBuf, 0, rq);

    const sp = this.ensureScores();
    // One dispatch per codes chunk: codes are addressed locally, scores/scales
    // globally at params.y (chunk base) + gid.x. params is rewritten per chunk,
    // so each chunk is its own submit (the queue preserves write/submit order).
    this.codes.eachChunk((buffer, baseRow, rowCount) => {
      device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([rowCount, baseRow, 0, 0]));
      const bind = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.queryBuf } },
          { binding: 2, resource: { buffer: sp.score } },
          { binding: 3, resource: { buffer: this.scales! } },
          { binding: 4, resource: { buffer: this.paramsBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(rowCount / WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([enc.finish()]);
    });

    // On-GPU top-k once the corpus is large enough that the N-score readback
    // dominates (§14.2 lever 3). Scores are already dense-global (chunk offsets
    // applied in-shader), so the reduction is chunk-oblivious. Skipped when the
    // over-fetched k would make the partials list larger than the full readback.
    if (GpuTopK.beneficial(this.rows, k)) {
      this.gpuTopk ??= new GpuTopK(this.ctx);
      return this.gpuTopk.query(sp.score, this.rows, k);
    }

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sp.score, 0, sp.readback, 0, this.rows * 4);
    device.queue.submit([enc.finish()]);

    await sp.readback.mapAsync(GPUMapMode.READ, 0, this.rows * 4);
    const scores = new Float32Array(sp.readback.getMappedRange(0, this.rows * 4).slice(0));
    sp.readback.unmap();
    return topK(scores, this.rows, k);
  }

  destroy(): void {
    this.encoder.dispose();
    this.codes.destroy();
    this.scales?.destroy();
    this.queryBuf.destroy();
    this.paramsBuf.destroy();
    this.scoreBuf?.destroy();
    this.readback?.destroy();
    this.gpuTopk?.destroy();
  }
}
