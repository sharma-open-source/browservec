// IVF (inverted-file) approximate index.
//
// Coarse-quantize the corpus into `nlist` clusters (k-means). A query scores only
// the `nprobe` nearest clusters instead of all N rows, turning an O(N) scan into
// O(nprobe/nlist · N). The probed rows are scored on the GPU through the *indexed*
// distance kernel (one dispatch over a candidate-id list — no physical reorder),
// then a small CPU top-k picks the winners. Exact scores, approximate recall: the
// true neighbour is missed only if it sits in an unprobed cluster, which `nprobe`
// trades off against latency.
//
// Build cost is kept off the critical path of ingest: k-means runs on a bounded
// reservoir SAMPLE of the corpus (not all N), and assignment of every row is done
// on the GPU. The index builds lazily on the first query after an append.

import type { Metric } from '../types.js';
import type { DeviceContext } from '../engine/device.js';
import { ChunkedCorpus, createQueryBuffer, createScoreBuffers, type ScorePair } from '../engine/buffers.js';
import { buildDistanceShader } from '../engine/wgsl/distance.js';
import { buildAssignShader } from '../engine/wgsl/assign.js';
import { createKMeansTrainer, type TrainMode } from './kmeansTrainer.js';
import { tracedGpuWait } from '../engine/profile.js';
import { topK, type FlatHit, type VectorIndex } from './flat.js';
import { GpuTopK } from './gpuTopk.js';
import { mulberry32 } from '../quant/prng.js';

const WORKGROUP_SIZE = 64;

export interface IVFParams {
  /** Number of clusters. Default ≈ sqrt(rows), clamped to [16, 4096]. */
  nlist?: number;
  /** Clusters scanned per query. Default ≈ 5% of nlist (min 1). */
  nprobe?: number;
  /** Reservoir sample size kept for training. Default 50_000. */
  sampleSize?: number;
  /** Max points actually fed to k-means per iteration (GPU-assigned). Default 16_384. */
  trainSize?: number;
  /** Lloyd iterations. Default 8. */
  iters?: number;
  /** Seed for sampling + k-means (reproducible builds). */
  seed?: number;
}

export class IVFIndex implements VectorIndex {
  private readonly corpus: ChunkedCorpus;
  private readonly queryBuf: GPUBuffer;
  private readonly paramsBuf: GPUBuffer;
  private readonly assignPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;

  private rows = 0;
  private builtRows = -1; // rows count at last build (-1 = never built)
  private nprobeOverride: number | undefined; // per-query override (see setNprobe)

  // Reservoir sample (CPU) for k-means; bounded so ingest stays memory-light.
  private readonly sampleCap: number;
  private readonly sample: Float32Array;
  private sampleFilled = 0;
  private seen = 0;
  private readonly sampleRng: () => number;

  // Built artefacts.
  private centroids: Float32Array | null = null; // nlist*dim (CPU, for probe pick)
  private centroidsBuf: GPUBuffer | null = null;
  private nlistActual = 0;
  private trainModeActual: TrainMode = 'pending'; // where the k-means mean-update ran
  private listRows: Int32Array | null = null; // row ids grouped by cluster
  private listOffset: Int32Array | null = null; // nlist+1 prefix offsets into listRows

  // Per-query scratch (grown on demand).
  private candidatesBuf: GPUBuffer | null = null;
  private candidateCap = 0;
  private scores: ScorePair | null = null;
  private gpuTopk: GpuTopK | null = null;
  private scoreCap = 0;

  constructor(
    private readonly ctx: DeviceContext,
    private readonly dim: number,
    metric: Metric,
    private readonly params: IVFParams = {},
    forcedRowsPerChunk?: number,
  ) {
    if (metric === 'l2') {
      throw new Error('IVF supports metric cosine/dot only (l2 is later work)');
    }
    const { device, limits } = ctx;
    // fp32 corpus split across GPU buffers past the per-buffer limit (§NFR-10),
    // so a pure-fp32 IVF store scales past one buffer like the other paths.
    this.corpus = new ChunkedCorpus(device, dim, 4, limits.maxStorageBufferBindingSize, forcedRowsPerChunk);
    this.queryBuf = createQueryBuffer(device, dim);
    this.paramsBuf = device.createBuffer({
      label: 'browservec:ivf-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampleCap = Math.max(1024, params.sampleSize ?? 50_000);
    this.sample = new Float32Array(this.sampleCap * dim);
    this.sampleRng = mulberry32((params.seed ?? 0x51ed270b) >>> 0);

    this.assignPipeline = this.makePipeline(buildAssignShader({ dim, workgroupSize: WORKGROUP_SIZE }), 'ivf-assign');
    this.scanPipeline = this.makePipeline(
      buildDistanceShader({ dim, metric, workgroupSize: WORKGROUP_SIZE, indexed: true }),
      'ivf-scan',
    );
  }

  get size(): number {
    return this.rows;
  }

  /** Number of clusters in the built index (0 until first build). */
  get nlist(): number {
    return this.nlistActual;
  }

  /** Number of GPU buffers the corpus spans (§NFR-10). 1 until it overflows one buffer. */
  get chunkCount(): number {
    return this.corpus.chunkCount;
  }

  /** Where the k-means mean-update ran (§NFR-8): 'worker', 'main-thread', or 'pending' pre-build. */
  get trainMode(): TrainMode {
    return this.trainModeActual;
  }

  /** Set the nprobe used by the next query() (consumed once). */
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

  append(data: Float32Array, count: number): void {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    this.corpus.append(data, count); // splits across chunk buffers as needed
    this.updateReservoir(data, count);
    this.rows += count;
    // Lists are now stale; next query rebuilds.
  }

  /** Streaming reservoir sample so k-means sees a uniform subset without a full CPU copy. */
  private updateReservoir(data: Float32Array, count: number): void {
    const dim = this.dim;
    for (let i = 0; i < count; i++) {
      const src = data.subarray(i * dim, i * dim + dim);
      if (this.sampleFilled < this.sampleCap) {
        this.sample.set(src, this.sampleFilled * dim);
        this.sampleFilled++;
      } else {
        const j = Math.floor(this.sampleRng() * (this.seen + 1));
        if (j < this.sampleCap) this.sample.set(src, j * dim);
      }
      this.seen++;
    }
  }

  // ---- Build ----------------------------------------------------------------

  private chooseNlist(): number {
    if (this.params.nlist) return Math.max(1, Math.min(this.params.nlist, this.rows));
    const guess = Math.round(Math.sqrt(this.rows));
    return Math.max(1, Math.min(4096, Math.max(16, guess), this.rows));
  }

  private async build(): Promise<void> {
    const { device } = this.ctx;
    const dim = this.dim;
    const nlist = this.chooseNlist();
    const seed = (this.params.seed ?? 0x51ed270b) >>> 0;
    const iters = this.params.iters ?? 8;
    const trainCount = Math.min(this.sampleFilled, this.params.trainSize ?? 16_384);

    this.nlistActual = nlist;
    this.centroidsBuf?.destroy();
    this.centroidsBuf = device.createBuffer({
      label: 'browservec:centroids',
      size: nlist * dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // 1) GPU-assisted Lloyd over the training subset: assignment (the heavy step)
    //    runs on the GPU; the centroid-mean update is cheap and stays on the CPU.
    const trainBuf = device.createBuffer({
      label: 'browservec:ivf-train',
      size: trainCount * dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(trainBuf, 0, this.sample.subarray(0, trainCount * dim));

    // The mean-update (and random init) run off-thread when a Worker is available
    // (§NFR-8), keeping the UI responsive during a build; the GPU still does the
    // heavy assignment step. Both paths produce identical centroids.
    const trainer = createKMeansTrainer({ sample: this.sample.subarray(0, trainCount * dim), trainCount, nlist, dim, seed });
    let centroids = await trainer.init();
    for (let it = 0; it < iters; it++) {
      device.queue.writeBuffer(this.centroidsBuf, 0, centroids);
      const assign = await this.assignBuffer(trainBuf, trainCount, nlist);
      centroids = await trainer.update(assign, it);
    }
    this.trainModeActual = trainer.mode();
    trainer.dispose();
    trainBuf.destroy();
    this.centroids = centroids;

    // 2) Final centroids → assign EVERY row → inverted lists.
    device.queue.writeBuffer(this.centroidsBuf, 0, centroids);
    const clusters = await this.assignCorpus(nlist);

    const counts = new Int32Array(nlist);
    for (let r = 0; r < this.rows; r++) counts[clusters[r]!]!++;
    const offset = new Int32Array(nlist + 1);
    for (let c = 0; c < nlist; c++) offset[c + 1] = offset[c]! + counts[c]!;
    const listRows = new Int32Array(this.rows);
    const cursor = offset.slice(0, nlist);
    for (let r = 0; r < this.rows; r++) {
      const c = clusters[r]!;
      listRows[cursor[c]!++] = r;
    }
    this.listOffset = offset;
    this.listRows = listRows;
    this.builtRows = this.rows;
  }

  /**
   * Assign every corpus row to its nearest centroid, one dispatch per chunk. The
   * assign kernel addresses rows *locally* (row_base = row·DIM, cluster[row]), so
   * each chunk's buffer scores its own local rows [0, rowCount); we place the
   * result at the chunk's global base row. Chunk-oblivious by construction.
   */
  private async assignCorpus(nlist: number): Promise<Uint32Array> {
    const clusters = new Uint32Array(this.rows);
    const chunks: { buffer: GPUBuffer; baseRow: number; rowCount: number }[] = [];
    this.corpus.eachChunk((buffer, baseRow, rowCount) => chunks.push({ buffer, baseRow, rowCount }));
    for (const { buffer, baseRow, rowCount } of chunks) {
      const part = await this.assignBuffer(buffer, rowCount, nlist);
      clusters.set(part, baseRow);
    }
    return clusters;
  }

  /** Run the assignment kernel over `rowCount` rows of `srcBuf`, return cluster ids. */
  private async assignBuffer(srcBuf: GPUBuffer, rowCount: number, nlist: number): Promise<Uint32Array> {
    const { device } = this.ctx;
    const bytes = rowCount * 4;
    const clusterBuf = device.createBuffer({
      label: 'browservec:cluster',
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      label: 'browservec:cluster-readback',
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([rowCount, nlist, 0, 0]));

    const bind = device.createBindGroup({
      layout: this.assignPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuf } },
        { binding: 1, resource: { buffer: this.centroidsBuf! } },
        { binding: 2, resource: { buffer: clusterBuf } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.assignPipeline);
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

    const nprobe = this.resolveNprobe();
    const probes = this.pickProbes(queryVec, nprobe);
    const candidates = this.gatherCandidates(probes);
    const n = candidates.length;
    if (n === 0) return [];

    const { sp, order } = this.dispatchScan(queryVec, candidates);
    // Remap the dense scan slot back to the original corpus row (candidates are
    // bucketed by chunk, so the scan output order differs from `candidates`).
    const remap = (h: FlatHit): FlatHit => ({ row: order[h.row]!, score: h.score });

    // Reduce on the GPU once the candidate set is large enough that its N-score
    // readback dominates (§14.2 lever 3); the remap is CPU-side either way.
    if (GpuTopK.beneficial(n, k)) {
      this.gpuTopk ??= new GpuTopK(this.ctx);
      return (await this.gpuTopk.query(sp.scores, n, k)).map(remap);
    }

    return (await this.readbackTopK(sp, n, k)).map(remap);
  }

  private resolveNprobe(): number {
    const def = Math.max(1, Math.round(this.nlistActual * 0.05));
    const n = this.nprobeOverride ?? this.params.nprobe ?? def;
    return Math.max(1, Math.min(n, this.nlistActual));
  }

  /** Score the query against each centroid and return the nprobe nearest cluster ids. */
  private pickProbes(query: Float32Array, nprobe: number): number[] {
    const dim = this.dim;
    const centroids = this.centroids!;
    const scores = new Float32Array(this.nlistActual);
    const vec = dim & ~3; // unrolled ×4 so the JS engine keeps accumulators in registers
    for (let c = 0; c < this.nlistActual; c++) {
      const base = c * dim;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      let i = 0;
      for (; i < vec; i += 4) {
        a0 += centroids[base + i]! * query[i]!;
        a1 += centroids[base + i + 1]! * query[i + 1]!;
        a2 += centroids[base + i + 2]! * query[i + 2]!;
        a3 += centroids[base + i + 3]! * query[i + 3]!;
      }
      let acc = a0 + a1 + a2 + a3;
      for (; i < dim; i++) acc += centroids[base + i]! * query[i]!;
      scores[c] = acc;
    }
    return topK(scores, this.nlistActual, nprobe).map((h) => h.row);
  }

  /** Concatenate the row ids of the probed clusters into one candidate list. */
  private gatherCandidates(probes: number[]): Uint32Array {
    const offset = this.listOffset!;
    const listRows = this.listRows!;
    let total = 0;
    for (const c of probes) total += offset[c + 1]! - offset[c]!;
    const out = new Uint32Array(total);
    let w = 0;
    for (const c of probes) {
      const start = offset[c]!;
      const end = offset[c + 1]!;
      out.set(listRows.subarray(start, end), w); // bulk memcpy per list
      w += end - start;
    }
    return out;
  }

  private ensureCandidateBuf(n: number): GPUBuffer {
    if (this.candidatesBuf && this.candidateCap >= n) return this.candidatesBuf;
    this.candidatesBuf?.destroy();
    const cap = Math.max(n, Math.ceil(this.candidateCap * 1.5), 4096);
    this.candidatesBuf = this.ctx.device.createBuffer({
      label: 'browservec:candidates',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.candidateCap = cap;
    return this.candidatesBuf;
  }

  private ensureScores(n: number): ScorePair {
    if (this.scores && this.scoreCap >= n) return this.scores;
    this.scores?.scores.destroy();
    this.scores?.readback.destroy();
    this.scores = createScoreBuffers(this.ctx.device, n);
    this.scoreCap = n;
    return this.scores;
  }

  /**
   * Score the candidates through the indexed kernel, leaving scores in sp.scores.
   * Candidates are global row ids spanning chunks, so we bucket them by chunk and
   * issue one dispatch per bucket binding that chunk's buffer with candidate ids
   * made *local* to it (params.z = this bucket's dense output offset). `order[p]`
   * is the global row that produced `scores[p]`, for remapping the top-k.
   */
  private dispatchScan(query: Float32Array, candidates: Uint32Array): { sp: ScorePair; order: Uint32Array } {
    const { device } = this.ctx;
    const total = candidates.length;
    const rpc = this.corpus.rowsPerChunk;

    // Fast path — single-chunk corpus (the common case): every candidate is
    // already a local row of chunk 0 and the scan output is in candidate order,
    // so skip the bucketing entirely and upload the candidate list as-is.
    if (this.corpus.chunkCount === 1) {
      const candBuf = this.ensureCandidateBuf(total);
      const sp = this.ensureScores(total);
      device.queue.writeBuffer(this.queryBuf, 0, query);
      device.queue.writeBuffer(candBuf, 0, candidates, 0, total);
      device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([total, 0, 0, 0]));
      const bind = device.createBindGroup({
        layout: this.scanPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.corpus.bufferAt(0) } },
          { binding: 1, resource: { buffer: this.queryBuf } },
          { binding: 2, resource: { buffer: sp.scores } },
          { binding: 3, resource: { buffer: this.paramsBuf } },
          { binding: 4, resource: { buffer: candBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.scanPipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(total / WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([enc.finish()]);
      return { sp, order: candidates };
    }

    // Bucket candidates by chunk (any order; their global ids are tracked in `order`).
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
    device.queue.writeBuffer(this.queryBuf, 0, query);

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
      // sees its own candidate list + params (see IVFQuantIndex chunked scan).
      device.queue.writeBuffer(candBuf, 0, local, 0, cnt);
      device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([cnt, 0, outOffset, 0]));
      const bind = device.createBindGroup({
        layout: this.scanPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.corpus.bufferAt(ci) } },
          { binding: 1, resource: { buffer: this.queryBuf } },
          { binding: 2, resource: { buffer: sp.scores } },
          { binding: 3, resource: { buffer: this.paramsBuf } },
          { binding: 4, resource: { buffer: candBuf } },
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

  /** Copy-back + CPU top-k for small candidate sets, selecting straight off the mapped range. */
  private async readbackTopK(sp: ScorePair, n: number, k: number): Promise<FlatHit[]> {
    const { device } = this.ctx;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sp.scores, 0, sp.readback, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await tracedGpuWait(sp.readback.mapAsync(GPUMapMode.READ, 0, n * 4));
    const scores = new Float32Array(sp.readback.getMappedRange(0, n * 4));
    const hits = topK(scores, n, k); // holds no references into the mapped range
    sp.readback.unmap();
    return hits;
  }

  destroy(): void {
    this.corpus.destroy();
    this.queryBuf.destroy();
    this.paramsBuf.destroy();
    this.centroidsBuf?.destroy();
    this.candidatesBuf?.destroy();
    this.scores?.scores.destroy();
    this.scores?.readback.destroy();
    this.gpuTopk?.destroy();
  }
}
