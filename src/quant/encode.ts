// Shared batch encoder — the CPU-heavy half of quantized ingest 
//  Rotating (randomized Hadamard / FWHT) then int8/int4-quantizing
// every row is the dominant cost of a large ingest and it's pure CPU work. This
// module is the single implementation of that transform, imported by BOTH the
// main-thread sync path and the Web Worker — so the offloaded path can never
// drift from the in-thread one.

import { Rotator } from './rotator.js';
import { quantizeRow, quantizeRow4, quantizeRow1, type QuantizedRow } from './codec.js';

/** Everything the encoder needs to reproduce the rotation + quantization. */
export interface EncodeConfig {
  dim: number;
  seed: number;
  rounds: number;
  bits: 1 | 4 | 8;
}

export interface EncodedBatch {
  /** Packed codes, `count * wordsPerRow` u32. */
  words: Uint32Array;
  /** Per-row dequant scale, `count` floats. */
  scales: Float32Array;
  /**
   * Rotated fp32 rows (`count * paddedDim`), only when requested — IVF needs them
   * to train k-means in the rotated space. Omitted for the flat path.
   */
  rotated?: Float32Array;
}

export class BatchEncoder {
  readonly rotator: Rotator;
  readonly paddedDim: number;
  readonly wordsPerRow: number;
  private readonly dim: number;
  private readonly quantize: (rotated: Float32Array) => QuantizedRow;
  private readonly scratch: Float32Array; // one rotated row when `rotated` isn't kept

  constructor(cfg: EncodeConfig) {
    this.dim = cfg.dim;
    this.rotator = new Rotator(cfg.dim, cfg.seed, cfg.rounds);
    this.paddedDim = this.rotator.paddedDim;
    // int8 packs 4 coords/word; int4 packs 8/word; 1-bit packs 32/word.
    const perWord = cfg.bits === 8 ? 4 : cfg.bits === 4 ? 8 : 32;
    this.wordsPerRow = this.paddedDim / perWord;
    this.quantize = cfg.bits === 8 ? quantizeRow : cfg.bits === 4 ? quantizeRow4 : quantizeRow1;
    this.scratch = new Float32Array(this.paddedDim);
  }

  /**
   * Rotate + quantize `count` rows packed in `data` (count*dim floats). When
   * `wantRotated` is true the rotated fp32 rows are returned too (IVF training).
   */
  encode(data: Float32Array, count: number, wantRotated: boolean): EncodedBatch {
    if (data.length !== count * this.dim) {
      throw new Error(`expected ${count * this.dim} floats, got ${data.length}`);
    }
    const pd = this.paddedDim;
    const words = new Uint32Array(count * this.wordsPerRow);
    const scales = new Float32Array(count);
    const rotated = wantRotated ? new Float32Array(count * pd) : undefined;

    for (let i = 0; i < count; i++) {
      const src = data.subarray(i * this.dim, i * this.dim + this.dim);
      // Rotate straight into the output slice when we're keeping the rotated rows,
      // otherwise reuse a single scratch row.
      const dst = rotated ? rotated.subarray(i * pd, i * pd + pd) : this.scratch;
      this.rotator.rotateInto(src, dst);
      const q = this.quantize(dst);
      words.set(q.words, i * this.wordsPerRow);
      scales[i] = q.scale;
    }

    const out: EncodedBatch = { words, scales };
    if (rotated) out.rotated = rotated;
    return out;
  }
}
