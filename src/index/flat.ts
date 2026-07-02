// Exact flat (brute-force) GPU index —  baseline & ground truth.
//
// Owns the corpus buffer, the compiled (dim,metric)-specialized pipeline, and the
// query path: write query -> dispatch distance kernel -> reduce to top-k. Past
// GPU_TOPK_MIN_ROWS the reduction runs on the GPU (src/index/gpuTopk.ts), so only
// a short candidate list is read back instead of all N scores; small corpora keep
// the simpler full-readback + CPU topK path.

import type { Metric } from '../types.js';
import type { DeviceContext } from '../engine/device.js';
import {
  ChunkedCorpus,
  createQueryBuffer,
  createScoreBuffers,
  type ScorePair,
} from '../engine/buffers.js';
import { buildDistanceShader } from '../engine/wgsl/distance.js';
import { tracedGpuWait } from '../engine/profile.js';
import { GpuTopK } from './gpuTopk.js';

const WORKGROUP_SIZE = 64;

export interface FlatHit {
  row: number;
  score: number;
}

/** Common shape for an on-GPU index (flat fp32 or quantized). */
export interface VectorIndex {
  readonly size: number;
  /**
   * Append `count` rows. Quantized indexes offload rotate+quantize to a Worker,
   * so this may be async; exact-fp32 indexes stay synchronous. Callers await it
   * either way, and appends must be awaited in order.
   */
  append(data: Float32Array, count: number): void | Promise<void>;
  query(queryVec: Float32Array, k: number): Promise<FlatHit[]>;
  destroy(): void;
}

export class FlatIndex {
  private readonly corpus: ChunkedCorpus;
  private readonly queryBuf: GPUBuffer;
  private readonly paramsBuf: GPUBuffer;
  private pipeline: GPUComputePipeline;
  private scores: ScorePair | null = null;
  private scoreCapacity = 0;
  private rows = 0;
  private gpuTopk: GpuTopK | null = null;
  private readonly paramsScratch = new Uint32Array(4);

  constructor(
    private readonly ctx: DeviceContext,
    private readonly dim: number,
    private readonly metric: Metric,
    forcedRowsPerChunk?: number,
  ) {
    const { device, limits } = ctx;
    this.corpus = new ChunkedCorpus(device, dim, 4, limits.maxStorageBufferBindingSize, forcedRowsPerChunk);
    this.queryBuf = createQueryBuffer(device, dim);
    this.paramsBuf = device.createBuffer({
      label: 'browservec:params',
      size: 16, // vec4<u32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.pipeline = this.buildPipeline();
  }

  get size(): number {
    return this.rows;
  }

  /** Number of GPU buffers the corpus is spread across (§NFR-10). 1 until it overflows. */
  get chunkCount(): number {
    return this.corpus.chunkCount;
  }

  private buildPipeline(): GPUComputePipeline {
    const code = buildDistanceShader({
      dim: this.dim,
      metric: this.metric,
      workgroupSize: WORKGROUP_SIZE,
    });
    const module = this.ctx.device.createShaderModule({ label: 'browservec:distance', code });
    return this.ctx.device.createComputePipeline({
      label: 'browservec:flat',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  /** Append a contiguous block of `count` rows packed in `data` (count*dim floats). */
  append(data: Float32Array, count: number): void {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    this.corpus.append(data, count);
    this.rows += count;
  }

  /** Overwrite a single existing row (used by update). */
  writeRow(row: number, vector: Float32Array): void {
    this.corpus.writeRow(row, vector);
  }

  private ensureScores(): ScorePair {
    if (this.scores && this.scoreCapacity >= this.rows) return this.scores;
    this.scores?.scores.destroy();
    this.scores?.readback.destroy();
    this.scores = createScoreBuffers(this.ctx.device, this.rows);
    this.scoreCapacity = this.rows;
    return this.scores;
  }

  /** Run the kernel over all rows and return the top-k by score (higher = closer). */
  async query(queryVec: Float32Array, k: number): Promise<FlatHit[]> {
    if (this.ctx.lost) throw new Error('GPU device lost; re-create the store');
    if (this.rows === 0) return [];

    const { device } = this.ctx;
    const sp = this.ensureScores();

    device.queue.writeBuffer(this.queryBuf, 0, queryVec);

    // One dispatch per corpus chunk, each writing its scores at global slot
    // params.y (chunk base) + gid.x into the shared scores buffer. params is
    // rewritten per chunk, so each chunk is its own submit to keep the uniform
    // stable for its dispatch (the queue preserves write/submit order).
    this.corpus.eachChunk((buffer, baseRow, rowCount) => {
      this.paramsScratch[0] = rowCount;
      this.paramsScratch[1] = baseRow;
      device.queue.writeBuffer(this.paramsBuf, 0, this.paramsScratch);
      const bind = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.queryBuf } },
          { binding: 2, resource: { buffer: sp.scores } },
          { binding: 3, resource: { buffer: this.paramsBuf } },
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

    // Reduce on the GPU once the O(N) score readback dominates; below the
    // threshold (or when a very large k would make the partials list exceed the
    // full readback) the copy-back + CPU sort is cheaper than a second dispatch.
    if (GpuTopK.beneficial(this.rows, k)) {
      this.gpuTopk ??= new GpuTopK(this.ctx);
      return this.gpuTopk.query(sp.scores, this.rows, k);
    }

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sp.scores, 0, sp.readback, 0, this.rows * 4);
    device.queue.submit([enc.finish()]);

    await tracedGpuWait(sp.readback.mapAsync(GPUMapMode.READ, 0, this.rows * 4));
    // Select straight off the mapped GPU memory — no O(N) copy. topK's output
    // holds no references into the mapped range, so unmapping after is safe.
    const scores = new Float32Array(sp.readback.getMappedRange(0, this.rows * 4));
    const hits = topK(scores, this.rows, k);
    sp.readback.unmap();
    return hits;
  }

  destroy(): void {
    this.corpus.destroy();
    this.queryBuf.destroy();
    this.paramsBuf.destroy();
    this.scores?.scores.destroy();
    this.scores?.readback.destroy();
    this.gpuTopk?.destroy();
  }
}

/**
 * CPU partial selection — O(n·k). Used for small corpora (below GPU_TOPK_MIN_ROWS)
 * and as the exact reference the GPU reduction is validated against.
 */
export function topK(scores: Float32Array, n: number, k: number): FlatHit[] {
  const limit = Math.min(k, n);
  const heap: FlatHit[] = [];
  for (let row = 0; row < n; row++) {
    const score = scores[row]!;
    if (heap.length < limit) {
      heap.push({ row, score });
      if (heap.length === limit) heap.sort((a, b) => a.score - b.score); // ascending; [0] is min
    } else if (score > heap[0]!.score) {
      // replace current minimum, keep sorted-ascending invariant
      heap[0] = { row, score };
      let i = 0;
      while (i + 1 < heap.length && heap[i]!.score > heap[i + 1]!.score) {
        const t = heap[i]!;
        heap[i] = heap[i + 1]!;
        heap[i + 1] = t;
        i++;
      }
    }
  }
  return heap.sort((a, b) => b.score - a.score); // descending for output
}
