// IVF × int8 combo index — the M4 target path.
//
// Combines TurboQuant int8 storage (so ~1M×768 fits in one ~1 GB buffer) with IVF
// clustering (so a query scans only the nprobe nearest cells). The corpus is
// rotated + quantized exactly as QuantIndex; on top, k-means partitions the
// rotated space into nlist cells and queries scan just the probed lists through
// the *indexed* quantized kernel. Caller does the usual exact fp32 re-rank.
//
// Build keeps the heavy work on the GPU: k-means assignment over the training
// sample uses the fp32 assign kernel, and the final all-rows assignment uses the
// quantized assign kernel (rows are assigned by their int8 representation, the
// same one they're scored with). Only the cheap centroid-mean update is on CPU.

import type { DeviceContext } from '../engine/device.js';
import { ChunkedCorpus, createScoreBuffers, type ScorePair } from '../engine/buffers.js';
import { buildQuantShader } from '../engine/wgsl/distanceQ8.js';
import { buildQuant4Shader } from '../engine/wgsl/distanceQ4.js';
import { buildQuant1Shader } from '../engine/wgsl/distanceQ1.js';
import { buildQuantAssignShader } from '../engine/wgsl/assignQ8.js';
import { buildQuant4AssignShader } from '../engine/wgsl/assignQ4.js';
import { buildQuant1AssignShader } from '../engine/wgsl/assignQ1.js';
import { buildAssignShader } from '../engine/wgsl/assign.js';
import { createQuantEncoder, type QuantEncoder, type IngestMode } from '../quant/encoder.js';
import { createKMeansTrainer, type TrainMode } from './kmeansTrainer.js';
import { topK, type FlatHit, type VectorIndex } from './flat.js';
import { GpuTopK } from './gpuTopk.js';
import type { IVFParams } from './ivf.js';

const WORKGROUP_SIZE = 64;

export class IVFQuantIndex implements VectorIndex {
  private readonly encoder: QuantEncoder;
  private readonly paddedDim: number;
  private readonly wordsPerRow: number;

  // Corpus storage: int8/int4 codes chunked across GPU buffers (§NFR-10); the
  // per-row scale stays in one small global buffer indexed by global row.
  private readonly codes: ChunkedCorpus;
  private scales: GPUBuffer | null = null;
  private scalesCap = 0;
  private rows = 0;

  private readonly queryBuf: GPUBuffer;
  private readonly paramsBuf: GPUBuffer;
  private readonly fp32AssignPipeline: GPUComputePipeline;
  private readonly quantAssignPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;

  // Training reservoir (rotated fp32, paddedDim each).
  private readonly sampleCap: number;
  private readonly sample: Float32Array;
  private sampleFilled = 0;
  private seen = 0;
  private readonly sampleRng: () => number;

  // Built artefacts.
  private centroids: Float32Array | null = null;
  private centroidsBuf: GPUBuffer | null = null;
  private nlistActual = 0;
  private trainModeActual: TrainMode = 'pending'; // where the k-means mean-update ran
  private listRows: Int32Array | null = null;
  private listOffset: Int32Array | null = null;
  private builtRows = -1;
  private nprobeOverride: number | undefined;

  // Per-query scratch.
  private candidatesBuf: GPUBuffer | null = null;
  private candidateCap = 0;
  private scoreScratch: ScorePair | null = null;
  private scoreCap = 0;
  private gpuTopk: GpuTopK | null = null;

  constructor(
    private readonly ctx: DeviceContext,
    readonly dim: number,
    seed: number,
    rounds: number,
    private readonly params: IVFParams = {},
    readonly bits: 1 | 4 | 8 = 8,
    forcedRowsPerChunk?: number,
  ) {
    // Ingest rotate+quantize runs through the encoder — off-thread in a Worker
    // when available (§NFR-8), else in-thread. It also returns the rotated fp32
    // rows so k-means can train in the rotated space.
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
      label: 'browservec:iq-query',
      size: this.paddedDim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.paramsBuf = device.createBuffer({
      label: 'browservec:iq-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampleCap = Math.max(1024, params.sampleSize ?? 50_000);
    this.sample = new Float32Array(this.sampleCap * this.paddedDim);
    this.sampleRng = (() => {
      // local deterministic PRNG seeded off the rotation seed
      let a = ((params.seed ?? seed) ^ 0x85ebca6b) >>> 0;
      return () => {
        a |= 0;
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    })();

    this.fp32AssignPipeline = this.makePipeline(
      buildAssignShader({ dim: this.paddedDim, workgroupSize: WORKGROUP_SIZE }),
      'iq-assign-f32',
    );
    this.quantAssignPipeline = this.makePipeline(
      bits === 8
        ? buildQuantAssignShader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE })
        : bits === 4
          ? buildQuant4AssignShader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE })
          : buildQuant1AssignShader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE }),
      `iq-assign-q${bits}`,
    );
    this.scanPipeline = this.makePipeline(
      bits === 8
        ? buildQuantShader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE, indexed: true })
        : bits === 4
          ? buildQuant4Shader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE, indexed: true })
          : buildQuant1Shader({ paddedDim: this.paddedDim, workgroupSize: WORKGROUP_SIZE, indexed: true }),
      `iq-scan-q${bits}`,
    );
  }

  get size(): number {
    return this.rows;
  }
  get nlist(): number {
    return this.nlistActual;
  }
  setNprobe(n: number | undefined): void {
    this.nprobeOverride = n;
  }

  private makePipeline(code: string, label: string): GPUComputePipeline {
    const module = this.ctx.device.createShaderModule({ label: `browservec:${label}`, code });
    return this.ctx.device.createComputePipeline({
      label: `browservec:${label}`,
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  // ---- Ingest ---------------------------------------------------------------

  /** Grow the single global scales buffer (4 B/row) geometrically, copying on growth. */
  private ensureScalesCapacity(rows: number): void {
    if (rows <= this.scalesCap) return;
    const next = Math.max(rows, this.scalesCap * 2, 4096);
    const { device } = this.ctx;
    const newScales = device.createBuffer({
      label: 'browservec:iq-scales',
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

  /** Where the k-means mean-update ran (§NFR-8): 'worker', 'main-thread', or 'pending' pre-build. */
  get trainMode(): TrainMode {
    return this.trainModeActual;
  }

  async append(data: Float32Array, count: number): Promise<void> {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    const base = this.rows;
    this.ensureScalesCapacity(base + count);

    // Off-thread rotate+quantize; `rotated` comes back so we can train k-means on it.
    const { words, scales, rotated } = await this.encoder.encode(data, count, true);
    const pd = this.paddedDim;
    for (let i = 0; i < count; i++) {
      this.addToReservoir(rotated!.subarray(i * pd, i * pd + pd));
    }

    this.codes.append(words, count); // splits across chunk buffers as needed
    this.ctx.device.queue.writeBuffer(this.scales!, base * 4, scales);
    this.rows = base + count;
  }

  private addToReservoir(rotated: Float32Array): void {
    const pd = this.paddedDim;
    if (this.sampleFilled < this.sampleCap) {
      this.sample.set(rotated, this.sampleFilled * pd);
      this.sampleFilled++;
    } else {
      const j = Math.floor(this.sampleRng() * (this.seen + 1));
      if (j < this.sampleCap) this.sample.set(rotated, j * pd);
    }
    this.seen++;
  }

  // ---- Build ----------------------------------------------------------------

  private chooseNlist(): number {
    if (this.params.nlist) return Math.max(1, Math.min(this.params.nlist, this.rows));
    const guess = Math.round(Math.sqrt(this.rows));
    return Math.max(1, Math.min(4096, Math.max(16, guess), this.rows));
  }

  private async build(): Promise<void> {
    const { device } = this.ctx;
    const pd = this.paddedDim;
    const nlist = this.chooseNlist();
    const seed = (this.params.seed ?? 0x51ed270b) >>> 0;
    const iters = this.params.iters ?? 8;
    const trainCount = Math.min(this.sampleFilled, this.params.trainSize ?? 16_384);

    this.nlistActual = nlist;
    this.centroidsBuf?.destroy();
    this.centroidsBuf = device.createBuffer({
      label: 'browservec:iq-centroids',
      size: nlist * pd * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Train on a fp32 (rotated) subset; assignment on GPU, mean-update on CPU.
    const trainBuf = device.createBuffer({
      label: 'browservec:iq-train',
      size: trainCount * pd * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(trainBuf, 0, this.sample.subarray(0, trainCount * pd));

    // Mean-update (and random init) offloaded to a Worker when available (§NFR-8),
    // in the padded rotated space; GPU still does assignment. Identical centroids.
    const trainer = createKMeansTrainer({ sample: this.sample.subarray(0, trainCount * pd), trainCount, nlist, dim: pd, seed });
    let centroids = await trainer.init();
    for (let it = 0; it < iters; it++) {
      device.queue.writeBuffer(this.centroidsBuf, 0, centroids);
      const assign = await this.assignFp32(trainBuf, trainCount, nlist);
      centroids = await trainer.update(assign, it);
    }
    this.trainModeActual = trainer.mode();
    trainer.dispose();
    trainBuf.destroy();
    this.centroids = centroids;

    // Final assignment of every row, using the int8 codes it will be scored with.
    device.queue.writeBuffer(this.centroidsBuf, 0, centroids);
    const clusters = await this.assignQuant(nlist);

    const counts = new Int32Array(nlist);
    for (let r = 0; r < this.rows; r++) counts[clusters[r]!]!++;
    const offset = new Int32Array(nlist + 1);
    for (let c = 0; c < nlist; c++) offset[c + 1] = offset[c]! + counts[c]!;
    const listRows = new Int32Array(this.rows);
    const cursor = offset.slice(0, nlist);
    for (let r = 0; r < this.rows; r++) listRows[cursor[clusters[r]!]!++] = r;
    this.listOffset = offset;
    this.listRows = listRows;
    this.builtRows = this.rows;
  }

  /** fp32 assignment over a rotated training buffer. */
  private async assignFp32(srcBuf: GPUBuffer, rowCount: number, nlist: number): Promise<Uint32Array> {
    const { device } = this.ctx;
    const { clusterBuf, readback, bytes } = this.makeClusterBuffers(rowCount);
    device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([rowCount, nlist, 0, 0]));
    const bind = device.createBindGroup({
      layout: this.fp32AssignPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuf } },
        { binding: 1, resource: { buffer: this.centroidsBuf! } },
        { binding: 2, resource: { buffer: clusterBuf } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    });
    return this.runAssign(this.fp32AssignPipeline, bind, clusterBuf, readback, rowCount, bytes);
  }

  /**
   * Quantized assignment over the full corpus. Codes are chunked, so we dispatch
   * once per chunk (codes local, cluster + scales global at params.z = chunk base)
   * into one shared cluster buffer, then read it back once (§NFR-10).
   */
  private async assignQuant(nlist: number): Promise<Uint32Array> {
    const { device } = this.ctx;
    const bytes = this.rows * 4;
    const clusterBuf = device.createBuffer({
      label: 'browservec:iq-cluster',
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      label: 'browservec:iq-cluster-readback',
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.codes.eachChunk((buffer, baseRow, rowCount) => {
      device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([rowCount, nlist, baseRow, 0]));
      const bind = device.createBindGroup({
        layout: this.quantAssignPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.centroidsBuf! } },
          { binding: 2, resource: { buffer: clusterBuf } },
          { binding: 3, resource: { buffer: this.scales! } },
          { binding: 4, resource: { buffer: this.paramsBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.quantAssignPipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(rowCount / WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([enc.finish()]);
    });

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(clusterBuf, 0, readback, 0, bytes);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const clusters = new Uint32Array(readback.getMappedRange(0, bytes).slice(0));
    readback.unmap();
    clusterBuf.destroy();
    readback.destroy();
    return clusters;
  }

  private makeClusterBuffers(rowCount: number): { clusterBuf: GPUBuffer; readback: GPUBuffer; bytes: number } {
    const bytes = rowCount * 4;
    const { device } = this.ctx;
    return {
      bytes,
      clusterBuf: device.createBuffer({
        label: 'browservec:iq-cluster',
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      readback: device.createBuffer({
        label: 'browservec:iq-cluster-readback',
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  private async runAssign(
    pipeline: GPUComputePipeline,
    bind: GPUBindGroup,
    clusterBuf: GPUBuffer,
    readback: GPUBuffer,
    rowCount: number,
    bytes: number,
  ): Promise<Uint32Array> {
    const { device } = this.ctx;
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(rowCount / WORKGROUP_SIZE));
    pass.end();
    enc.copyBufferToBuffer(clusterBuf, 0, readback, 0, bytes);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const clusters = new Uint32Array(readback.getMappedRange(0, bytes).slice(0));
    readback.unmap();
    clusterBuf.destroy();
    readback.destroy();
    return clusters;
  }

  // ---- Query ----------------------------------------------------------------

  async query(queryVec: Float32Array, k: number): Promise<FlatHit[]> {
    if (this.ctx.lost) throw new Error('GPU device lost; re-create the store');
    if (this.rows === 0) return [];
    if (this.builtRows !== this.rows) await this.build();

    const rq = this.encoder.rotateQuery(queryVec); // rotated fp32 query (asymmetric)
    const probes = this.pickProbes(rq, this.resolveNprobe());
    const candidates = this.gatherCandidates(probes);
    const total = candidates.length;
    if (total === 0) return [];

    const { sp, order } = this.dispatchScan(rq, candidates);
    // Remap the dense scan slot back to the original corpus row.
    const remap = (h: FlatHit): FlatHit => ({ row: order[h.row]!, score: h.score });

    // Reduce on the GPU once the candidate readback dominates (§14.2 lever 3);
    // skipped when the over-fetched k would make the partials exceed the readback.
    if (GpuTopK.beneficial(total, k)) {
      this.gpuTopk ??= new GpuTopK(this.ctx);
      return (await this.gpuTopk.query(sp.scores, total, k)).map(remap);
    }

    const scores = await this.readbackScores(sp, total);
    return topK(scores, total, k).map(remap);
  }

  private resolveNprobe(): number {
    const def = Math.max(1, Math.round(this.nlistActual * 0.05));
    const n = this.nprobeOverride ?? this.params.nprobe ?? def;
    return Math.max(1, Math.min(n, this.nlistActual));
  }

  private pickProbes(rq: Float32Array, nprobe: number): number[] {
    const pd = this.paddedDim;
    const centroids = this.centroids!;
    const scores = new Float32Array(this.nlistActual);
    for (let c = 0; c < this.nlistActual; c++) {
      const base = c * pd;
      let acc = 0;
      for (let i = 0; i < pd; i++) acc += centroids[base + i]! * rq[i]!;
      scores[c] = acc;
    }
    return topK(scores, this.nlistActual, nprobe).map((h) => h.row);
  }

  private gatherCandidates(probes: number[]): Uint32Array {
    const offset = this.listOffset!;
    const listRows = this.listRows!;
    let total = 0;
    for (const c of probes) total += offset[c + 1]! - offset[c]!;
    const out = new Uint32Array(total);
    let w = 0;
    for (const c of probes) {
      for (let i = offset[c]!; i < offset[c + 1]!; i++) out[w++] = listRows[i]!;
    }
    return out;
  }

  private ensureCandidateBuf(n: number): GPUBuffer {
    if (this.candidatesBuf && this.candidateCap >= n) return this.candidatesBuf;
    this.candidatesBuf?.destroy();
    const cap = Math.max(n, Math.ceil(this.candidateCap * 1.5), 4096);
    this.candidatesBuf = this.ctx.device.createBuffer({
      label: 'browservec:iq-candidates',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.candidateCap = cap;
    return this.candidatesBuf;
  }

  private ensureScores(n: number): ScorePair {
    if (this.scoreScratch && this.scoreCap >= n) return this.scoreScratch;
    this.scoreScratch?.scores.destroy();
    this.scoreScratch?.readback.destroy();
    this.scoreScratch = createScoreBuffers(this.ctx.device, n);
    this.scoreCap = n;
    return this.scoreScratch;
  }

  /**
   * Score the candidate rows through the indexed quant kernel. Codes are chunked,
   * so candidates are bucketed by chunk and each bucket is one dispatch binding
   * that chunk's codes with *local* candidate ids (params.y = chunk base for the
   * global scales, params.z = this bucket's offset in the output). `order[p]` is
   * the global row that produced `scores[p]`, for remapping the top-k (§NFR-10).
   */
  private dispatchScan(
    rq: Float32Array,
    candidates: Uint32Array,
  ): { sp: ScorePair; order: Uint32Array } {
    const { device } = this.ctx;
    const total = candidates.length;
    const rpc = this.codes.rowsPerChunk;

    // Bucket candidates by chunk (any order; we track their global ids in `order`).
    const buckets = new Map<number, number[]>();
    for (let i = 0; i < total; i++) {
      const g = candidates[i]!;
      const ci = Math.floor(g / rpc);
      let b = buckets.get(ci);
      if (!b) buckets.set(ci, (b = []));
      b.push(g);
    }
    let maxBucket = 0;
    for (const b of buckets.values()) maxBucket = Math.max(maxBucket, b.length);

    const candBuf = this.ensureCandidateBuf(maxBucket);
    const sp = this.ensureScores(total);
    device.queue.writeBuffer(this.queryBuf, 0, rq);

    const order = new Uint32Array(total);
    const local = new Uint32Array(Math.max(maxBucket, 1));
    let outOffset = 0;
    for (const [ci, ids] of buckets) {
      const base = ci * rpc;
      const cnt = ids.length;
      for (let j = 0; j < cnt; j++) {
        local[j] = ids[j]! - base; // local row within chunk ci
        order[outOffset + j] = ids[j]!;
      }
      // Reused per bucket; the queue preserves write→submit order so each dispatch
      // sees its own candidate list + params (see QuantIndex chunked query).
      device.queue.writeBuffer(candBuf, 0, local, 0, cnt);
      device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([cnt, base, outOffset, 0]));
      const bind = device.createBindGroup({
        layout: this.scanPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.codes.bufferAt(ci) } },
          { binding: 1, resource: { buffer: this.queryBuf } },
          { binding: 2, resource: { buffer: sp.scores } },
          { binding: 3, resource: { buffer: this.scales! } },
          { binding: 4, resource: { buffer: this.paramsBuf } },
          { binding: 5, resource: { buffer: candBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.scanPipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(cnt / WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([enc.finish()]);
      outOffset += cnt;
    }

    return { sp, order };
  }

  /** Full copy-back of the candidate scores for the CPU top-k path (small sets). */
  private async readbackScores(sp: ScorePair, n: number): Promise<Float32Array> {
    const { device } = this.ctx;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sp.scores, 0, sp.readback, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await sp.readback.mapAsync(GPUMapMode.READ, 0, n * 4);
    const out = new Float32Array(sp.readback.getMappedRange(0, n * 4).slice(0));
    sp.readback.unmap();
    return out;
  }

  destroy(): void {
    this.encoder.dispose();
    this.codes.destroy();
    this.scales?.destroy();
    this.queryBuf.destroy();
    this.paramsBuf.destroy();
    this.centroidsBuf?.destroy();
    this.candidatesBuf?.destroy();
    this.scoreScratch?.scores.destroy();
    this.scoreScratch?.readback.destroy();
    this.gpuTopk?.destroy();
  }
}
