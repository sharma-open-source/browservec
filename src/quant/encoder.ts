// Quantization encoder — the seam between synchronous in-thread encoding and the
// Web Worker offload . Indexes talk to this interface and
// never know which path ran. The worker is booted lazily on first encode and, if
// it can't be created (no Worker global, CSP block, non-Vite runtime), the encoder
// transparently falls back to running the exact same BatchEncoder on this thread.

import { BatchEncoder, type EncodeConfig, type EncodedBatch } from './encode.js';
// Vite base64-inlines the worker into the main bundle (must be a *static* import —
// the dynamic `import('…?worker&inline')` form is emitted as a separate chunk).
// The specifier only exists under Vite; `tsc` resolves it via the ambient
// declaration in src/env.d.ts, and the built dist never constructs the Worker
// unless `typeof Worker !== 'undefined'`, so Node/SSR consumers are unaffected.
import InlineQuantizeWorker from './quantize.worker?worker&inline';

export type IngestMode = 'worker' | 'main-thread' | 'pending';

export interface QuantEncoder {
  /** Padded (power-of-two) dimension the rotator works in. */
  readonly paddedDim: number;
  /** Packed u32 words per row (paddedDim/4 for int8, /8 for int4). */
  readonly wordsPerRow: number;
  /** Which path the last/next encode uses — 'pending' until the first encode resolves it. */
  mode(): IngestMode;
  /** Rotate + quantize `count` rows. `data` is copied to the worker, not retained. */
  encode(data: Float32Array, count: number, wantRotated: boolean): Promise<EncodedBatch>;
  /** Rotate a single query vector on this thread (the query path is latency-bound). */
  rotateQuery(q: Float32Array): Float32Array;
  dispose(): void;
}

interface Pending {
  resolve: (b: EncodedBatch) => void;
  reject: (e: unknown) => void;
}

class WorkerQuantEncoder implements QuantEncoder {
  readonly paddedDim: number;
  readonly wordsPerRow: number;
  private readonly local: BatchEncoder; // also the fallback if the worker never boots
  private worker: Worker | null = null;
  private booted = false;
  private fellBack = false;
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor(private readonly cfg: EncodeConfig) {
    this.local = new BatchEncoder(cfg);
    this.paddedDim = this.local.paddedDim;
    this.wordsPerRow = this.local.wordsPerRow;
  }

  mode(): IngestMode {
    if (this.fellBack) return 'main-thread';
    if (this.worker) return 'worker';
    return 'pending';
  }

  /** Boot the worker once (lazily, on first encode). Returns null if unavailable. */
  private ensureWorker(): Worker | null {
    if (this.booted) return this.worker;
    this.booted = true;
    if (typeof Worker === 'undefined') {
      this.fellBack = true;
      return null;
    }
    try {
      const w = new InlineQuantizeWorker();
      w.onmessage = (e: MessageEvent) => this.onMessage(e);
      w.onerror = () => this.failAll(new Error('quantize worker crashed'));
      w.postMessage({ type: 'init', config: this.cfg });
      this.worker = w;
      return w;
    } catch {
      this.fellBack = true;
      return null;
    }
  }

  private onMessage(e: MessageEvent): void {
    const msg = e.data as { type: string; id?: number; words?: Uint32Array; scales?: Float32Array; rotated?: Float32Array };
    if (msg.type !== 'encoded' || msg.id === undefined) return;
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    const out: EncodedBatch = { words: msg.words!, scales: msg.scales! };
    if (msg.rotated) out.rotated = msg.rotated;
    p.resolve(out);
  }

  private failAll(err: unknown): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  async encode(data: Float32Array, count: number, wantRotated: boolean): Promise<EncodedBatch> {
    const worker = this.ensureWorker();
    if (!worker) {
      // Fallback: run in-thread. Same code, just blocking.
      return this.local.encode(data, count, wantRotated);
    }
    const id = this.nextId++;
    // `data` may be a view into a large caller-owned buffer (e.g. a whole
    // snapshot). slice() gives a tight, owned copy we can transfer zero-copy —
    // cheaper than structured-cloning the entire backing ArrayBuffer, and it
    // leaves the caller's data intact. The FWHT dwarfs this one copy.
    const owned = data.slice();
    return new Promise<EncodedBatch>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ type: 'encode', id, data: owned, count, wantRotated }, [owned.buffer]);
    });
  }

  rotateQuery(q: Float32Array): Float32Array {
    return this.local.rotator.rotate(q);
  }

  dispose(): void {
    this.failAll(new Error('encoder disposed'));
    this.worker?.terminate();
    this.worker = null;
  }
}

export function createQuantEncoder(cfg: EncodeConfig): QuantEncoder {
  return new WorkerQuantEncoder(cfg);
}
