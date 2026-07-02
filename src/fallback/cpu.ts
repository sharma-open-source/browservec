// CPU fallback index (REQUIREMENTS.md §NFR-7, §M6 — "WASM/CPU parity path").
//
// When WebGPU is absent (older/locked-down browsers, some mobile) the library must
// still work with *identical results*, just slower. This is the exact brute-force
// flat scan that the GPU FlatIndex mirrors, done on the CPU: same metric semantics
// (dot for cosine/dot with vectors normalized on insert; negative squared-L2 so
// "higher = closer" holds), same top-k selection. It's the correctness reference
// the GPU path is measured against, so parity is by construction.
//
// The hot loop runs on a hand-assembled WASM-SIMD kernel (see ./simd.ts) ~7× faster
// than scalar JS, matched to f32 precision. The corpus lives *inside* the kernel's
// linear memory so a query needs no copy — we just point the kernel at it. Engines
// without WASM/SIMD (rare) transparently fall back to the unrolled scalar loop below;
// results are identical either way. The throughput ceiling is why the GPU path
// exists — this is the "works everywhere" safety net, not the fast path.

import type { Metric } from '../types.js';
import { topK, type FlatHit, type VectorIndex } from '../index/flat.js';
import { createSimd, type SimdInstance } from './simd.js';

const PAGE = 65536;

export class CpuIndex implements VectorIndex {
  private rows = 0;

  // SIMD path: the corpus lives packed row-major from byte 0 of the kernel's
  // linear memory; per query we stage the query vector and score output just
  // above the live corpus. `null` on engines without WASM-SIMD.
  private simd: SimdInstance | null;

  // Scalar fallback path: same packed layout, but in a plain JS Float32Array.
  private data = new Float32Array(0);

  constructor(
    private readonly dim: number,
    private readonly metric: Metric,
  ) {
    this.simd = createSimd();
  }

  get size(): number {
    return this.rows;
  }

  append(data: Float32Array, count: number): void {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    const newRows = this.rows + count;
    if (this.simd) {
      this.ensureFloats(newRows * this.dim);
      new Float32Array(this.simd.mem.buffer).set(data, this.rows * this.dim);
    } else {
      this.ensureCapacity(newRows);
      this.data.set(data, this.rows * this.dim);
    }
    this.rows = newRows;
  }

  /** Overwrite a single existing row (update path parity with FlatIndex). */
  writeRow(row: number, vector: Float32Array): void {
    if (this.simd) new Float32Array(this.simd.mem.buffer).set(vector, row * this.dim);
    else this.data.set(vector, row * this.dim);
  }

  async query(queryVec: Float32Array, k: number): Promise<FlatHit[]> {
    if (this.rows === 0) return [];
    const { dim, rows } = this;

    if (this.simd) {
      const corpusFloats = rows * dim;
      const qOff = corpusFloats;
      const outOff = qOff + dim;
      this.ensureFloats(outOff + rows);
      // Re-view after any grow (grow detaches the previous buffer).
      const view = new Float32Array(this.simd.mem.buffer);
      view.set(queryVec, qOff);
      const kernel = this.metric === 'l2' ? this.simd.l2 : this.simd.dot;
      kernel(0, qOff * 4, outOff * 4, rows, dim);
      const scores = view.subarray(outOff, outOff + rows);
      return topK(scores, rows, k);
    }

    const scores = new Float32Array(rows);
    if (this.metric === 'l2') this.scoreL2(queryVec, scores);
    else this.scoreDot(queryVec, scores);
    return topK(scores, rows, k);
  }

  destroy(): void {
    this.simd = null;
    this.data = new Float32Array(0);
    this.rows = 0;
  }

  // --- SIMD-path memory growth (grows the WASM linear memory geometrically) ---
  private ensureFloats(needFloats: number): void {
    const mem = this.simd!.mem;
    const needBytes = needFloats * 4;
    const curBytes = mem.buffer.byteLength;
    if (needBytes <= curBytes) return;
    const target = Math.max(needBytes, Math.floor(curBytes * 1.5));
    mem.grow(Math.ceil((target - curBytes) / PAGE));
  }

  // --- scalar fallback path (only when WASM-SIMD is unavailable) ---

  // dot(v, q) — cosine (vectors pre-normalized) and dot share this. Unrolled ×4
  // so the JS engine can keep the accumulators in registers.
  private scoreDot(q: Float32Array, out: Float32Array): void {
    const { dim, data, rows } = this;
    const tail = dim & ~3;
    for (let r = 0; r < rows; r++) {
      const base = r * dim;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      let i = 0;
      for (; i < tail; i += 4) {
        a0 += data[base + i]! * q[i]!;
        a1 += data[base + i + 1]! * q[i + 1]!;
        a2 += data[base + i + 2]! * q[i + 2]!;
        a3 += data[base + i + 3]! * q[i + 3]!;
      }
      let acc = a0 + a1 + a2 + a3;
      for (; i < dim; i++) acc += data[base + i]! * q[i]!;
      out[r] = acc;
    }
  }

  // Negative squared L2, matching the GPU kernel's "higher = closer" convention.
  // Unrolled ×4 like scoreDot so the accumulators stay in registers.
  private scoreL2(q: Float32Array, out: Float32Array): void {
    const { dim, data, rows } = this;
    const tail = dim & ~3;
    for (let r = 0; r < rows; r++) {
      const base = r * dim;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      let i = 0;
      for (; i < tail; i += 4) {
        const d0 = data[base + i]! - q[i]!;
        const d1 = data[base + i + 1]! - q[i + 1]!;
        const d2 = data[base + i + 2]! - q[i + 2]!;
        const d3 = data[base + i + 3]! - q[i + 3]!;
        a0 += d0 * d0;
        a1 += d1 * d1;
        a2 += d2 * d2;
        a3 += d3 * d3;
      }
      let acc = a0 + a1 + a2 + a3;
      for (; i < dim; i++) {
        const d = data[base + i]! - q[i]!;
        acc += d * d;
      }
      out[r] = -acc;
    }
  }

  private ensureCapacity(rows: number): void {
    const needed = rows * this.dim;
    if (needed <= this.data.length) return;
    const nextRows = Math.max(rows, Math.ceil((this.data.length / this.dim) * 1.5), 1024);
    const next = new Float32Array(nextRows * this.dim);
    next.set(this.data.subarray(0, this.rows * this.dim));
    this.data = next;
  }
}
