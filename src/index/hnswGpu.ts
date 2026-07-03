// GPU graph-search executor (M7b) — owns the WebGPU side of HNSW.
//
// The CPU (worker) still owns graph construction; this class holds the GPU
// mirrors — corpus rows uploaded at append time, the flat layer-0 adjacency
// uploaded per build generation — and runs the single-dispatch beam-search
// kernel (engine/wgsl/graphSearch.ts). One workgroup per query makes batches
// concurrent for free: search() takes nQ packed queries in one dispatch.
//
// Honest performance model: a single query pays fixed dispatch + mapAsync
// readback latency (~ms in browsers), which the CPU walk doesn't — the GPU
// path earns its keep on BATCHED queries and very large corpora, and the
// examples measure exactly that trade instead of hiding it.
//
// The corpus mirror is a single storage buffer for now: the beam hops to
// arbitrary rows, so it can't be split across bind groups the way the linear
// scans are (§NFR-10). Past the device's per-buffer limit the index falls back
// to CPU search (hnsw.ts decides); chunked graph search is later work.

import type { Metric } from '../types.js';
import type { DeviceContext } from '../engine/device.js';
import type { FlatHit } from './flat.js';
import { buildGraphSearchShader, GRAPH_EF_CAP } from '../engine/wgsl/graphSearch.js';
import { tracedGpuWait } from '../engine/profile.js';

const WORKGROUP_SIZE = 64;
const MAX_ENTRIES = 8;
const EMPTY = 0xffffffff;

export class HNSWGpuSearcher {
  private readonly pipeline: GPUComputePipeline;
  private readonly paramsBuf: GPUBuffer;
  private readonly entriesBuf: GPUBuffer;

  private corpusBuf: GPUBuffer | null = null;
  private corpusCapRows = 0;
  private corpusRows = 0;

  private graphBuf: GPUBuffer | null = null;
  private graphCapRows = 0;
  private graphRows = 0; // rows covered by the uploaded adjacency

  private queryBuf: GPUBuffer | null = null;
  private queryCapQ = 0;
  private outBuf: GPUBuffer | null = null; // dist then id planes, nQ*EF each
  private readbackBuf: GPUBuffer | null = null;
  private outCapQ = 0;

  constructor(
    private readonly ctx: DeviceContext,
    private readonly dim: number,
    metric: Metric,
    /** Fixed out-degree (graph rows are `degree` slots wide) = 2·M. */
    private readonly degree: number,
  ) {
    if (degree > WORKGROUP_SIZE) {
      throw new Error(`GPU graph search supports degree ≤ ${WORKGROUP_SIZE} (M ≤ ${WORKGROUP_SIZE / 2}), got ${degree}`);
    }
    const { device } = ctx;
    this.paramsBuf = device.createBuffer({
      label: 'browservec:graph-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.entriesBuf = device.createBuffer({
      label: 'browservec:graph-entries',
      size: MAX_ENTRIES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const code = buildGraphSearchShader({ dim, k: degree, metric, workgroupSize: WORKGROUP_SIZE });
    const module = device.createShaderModule({ label: 'browservec:graph-search', code });
    this.pipeline = device.createComputePipeline({
      label: 'browservec:graph-search',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  /** Rows the corpus mirror can hold before hitting the per-buffer limit. */
  get maxRows(): number {
    return Math.floor(this.ctx.limits.maxStorageBufferBindingSize / (this.dim * 4));
  }

  /** Mirror appended rows into the GPU corpus buffer. Throws past the buffer limit. */
  append(data: Float32Array, count: number): void {
    const { device } = this.ctx;
    const needed = this.corpusRows + count;
    if (needed > this.corpusCapRows) {
      const nextRows = Math.max(needed, Math.ceil(this.corpusCapRows * 1.5), 1024);
      const capped = Math.min(nextRows, this.maxRows);
      if (needed > capped) {
        throw new Error(`GPU graph corpus exceeds one storage buffer (${this.maxRows} rows at dim=${this.dim})`);
      }
      const next = device.createBuffer({
        label: 'browservec:graph-corpus',
        size: capped * this.dim * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      if (this.corpusBuf) {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(this.corpusBuf, 0, next, 0, this.corpusRows * this.dim * 4);
        device.queue.submit([enc.finish()]);
        this.corpusBuf.destroy();
      }
      this.corpusBuf = next;
      this.corpusCapRows = capped;
    }
    device.queue.writeBuffer(this.corpusBuf!, this.corpusRows * this.dim * 4, data, 0, count * this.dim);
    this.corpusRows = needed;
  }

  /** Upload a fresh layer-0 adjacency snapshot (rows × degree, EMPTY-padded). */
  uploadGraph(links: Uint32Array, rows: number): void {
    const { device } = this.ctx;
    if (rows > this.graphCapRows) {
      this.graphBuf?.destroy();
      const capRows = Math.max(rows, Math.ceil(this.graphCapRows * 1.5), 1024);
      this.graphBuf = device.createBuffer({
        label: 'browservec:graph-links',
        size: capRows * this.degree * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.graphCapRows = capRows;
    }
    device.queue.writeBuffer(this.graphBuf!, 0, links, 0, rows * this.degree);
    this.graphRows = rows;
  }

  /** Adjacency generation currently on the GPU (rows covered). */
  get uploadedRows(): number {
    return this.graphRows;
  }

  /**
   * Beam-search `nQ` packed queries in ONE dispatch (one workgroup each).
   * Returns per-query hits, deduped and sorted, at most k each.
   */
  async search(queries: Float32Array, nQ: number, k: number, ef: number, entry: number): Promise<FlatHit[][]> {
    const { device } = this.ctx;
    const efA = Math.min(Math.max(ef, k), GRAPH_EF_CAP);
    this.ensureQueryCapacity(nQ);

    // Seed rows: the HNSW entry plus rows spread evenly across the corpus —
    // the flat-graph stand-in for descending the (CPU-side) hierarchy.
    const entries = new Uint32Array(MAX_ENTRIES).fill(EMPTY);
    let e = 0;
    entries[e++] = entry >>> 0;
    const extra = Math.min(MAX_ENTRIES - 1, this.graphRows);
    for (let i = 0; i < extra; i++) {
      entries[e++] = Math.floor(((i + 1) * this.graphRows) / (extra + 1)) % this.graphRows;
    }
    device.queue.writeBuffer(this.entriesBuf, 0, entries, 0, e);
    device.queue.writeBuffer(this.queryBuf!, 0, queries, 0, nQ * this.dim);
    // iterCap: ef expansions matches the CPU beam's work bound; entries count e.
    device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([efA, efA, e, 0]));

    const bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.corpusBuf! } },
        { binding: 1, resource: { buffer: this.graphBuf! } },
        { binding: 2, resource: { buffer: this.queryBuf! } },
        { binding: 3, resource: { buffer: this.entriesBuf } },
        { binding: 4, resource: { buffer: this.outBuf!, offset: 0, size: this.outCapQ * GRAPH_EF_CAP * 4 } },
        { binding: 5, resource: { buffer: this.outBuf!, offset: this.outCapQ * GRAPH_EF_CAP * 4 } },
        { binding: 6, resource: { buffer: this.paramsBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(nQ);
    pass.end();
    // Copy both planes (dist, id) for the live queries into the readback.
    const distBytes = nQ * GRAPH_EF_CAP * 4;
    enc.copyBufferToBuffer(this.outBuf!, 0, this.readbackBuf!, 0, distBytes);
    enc.copyBufferToBuffer(this.outBuf!, this.outCapQ * GRAPH_EF_CAP * 4, this.readbackBuf!, distBytes, distBytes);
    device.queue.submit([enc.finish()]);

    await tracedGpuWait(this.readbackBuf!.mapAsync(GPUMapMode.READ, 0, distBytes * 2));
    const mapped = this.readbackBuf!.getMappedRange(0, distBytes * 2);
    const dists = new Float32Array(mapped, 0, nQ * GRAPH_EF_CAP);
    const ids = new Uint32Array(mapped, distBytes, nQ * GRAPH_EF_CAP);

    const results: FlatHit[][] = [];
    for (let q = 0; q < nQ; q++) {
      const base = q * GRAPH_EF_CAP;
      // Collect live beam entries, dedup (hash-shed can duplicate), sort, take k.
      const seen = new Set<number>();
      const hits: FlatHit[] = [];
      for (let s = 0; s < efA; s++) {
        const id = ids[base + s]!;
        if (id === EMPTY || seen.has(id)) continue;
        seen.add(id);
        hits.push({ row: id, score: -dists[base + s]! });
      }
      hits.sort((a, b) => b.score - a.score);
      results.push(hits.slice(0, k));
    }
    this.readbackBuf!.unmap();
    return results;
  }

  private ensureQueryCapacity(nQ: number): void {
    const { device } = this.ctx;
    if (nQ > this.queryCapQ) {
      this.queryBuf?.destroy();
      const cap = Math.max(nQ, 16);
      this.queryBuf = device.createBuffer({
        label: 'browservec:graph-queries',
        size: cap * this.dim * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.queryCapQ = cap;
    }
    if (nQ > this.outCapQ) {
      this.outBuf?.destroy();
      this.readbackBuf?.destroy();
      const cap = Math.max(nQ, 16);
      this.outBuf = device.createBuffer({
        label: 'browservec:graph-out',
        size: cap * GRAPH_EF_CAP * 8, // dist plane + id plane
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.readbackBuf = device.createBuffer({
        label: 'browservec:graph-out-readback',
        size: cap * GRAPH_EF_CAP * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.outCapQ = cap;
    }
  }

  destroy(): void {
    this.paramsBuf.destroy();
    this.entriesBuf.destroy();
    this.corpusBuf?.destroy();
    this.graphBuf?.destroy();
    this.queryBuf?.destroy();
    this.outBuf?.destroy();
    this.readbackBuf?.destroy();
  }
}
