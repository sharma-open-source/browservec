// K-means trainer — the seam between the synchronous in-thread centroid-mean
// update and its Web Worker offload (§NFR-8). IVFIndex/IVFQuantIndex talk to this
// interface and never know which path ran. The worker is booted on construction
// and, if it can't be created (no Worker global, CSP block, non-Vite runtime), the
// trainer transparently runs the exact same kmeans helpers on this thread. Both
// paths call identical deterministic functions on identical inputs, so a build is
// byte-for-byte reproducible regardless of where the mean-update executed.

import { randomInitCentroids, updateCentroids } from './kmeans.js';
// Vite base64-inlines the worker into the main bundle (must be a *static* import —
// see the note in quant/encoder.ts). `tsc` resolves it via env.d.ts, and the built
// dist never constructs the Worker unless `typeof Worker !== 'undefined'`.
import InlineKMeansWorker from './kmeans.worker?worker&inline';

export type TrainMode = 'worker' | 'main-thread' | 'pending';

export interface KMeansTrainerConfig {
  /** Training sample, `trainCount * dim` floats, row-major. Copied, not retained. */
  sample: Float32Array;
  trainCount: number;
  nlist: number;
  dim: number;
  seed: number;
}

export interface KMeansTrainer {
  /** Which path the run used — 'pending' until the first call resolves it. */
  mode(): TrainMode;
  /** Initial centroids (random sample pick + normalize). Resolves once, first. */
  init(): Promise<Float32Array>;
  /** Recompute centroids from a GPU assignment for Lloyd iteration `iter`. */
  update(assign: Uint32Array, iter: number): Promise<Float32Array>;
  dispose(): void;
}

interface Pending {
  resolve: (c: Float32Array) => void;
  reject: (e: unknown) => void;
}

class WorkerKMeansTrainer implements KMeansTrainer {
  private worker: Worker | null = null;
  private fellBack = false;
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private initPending: Pending | null = null;

  constructor(private readonly cfg: KMeansTrainerConfig) {}

  mode(): TrainMode {
    if (this.fellBack) return 'main-thread';
    if (this.worker) return 'worker';
    return 'pending';
  }

  init(): Promise<Float32Array> {
    if (typeof Worker === 'undefined') return this.fallbackInit();
    let w: Worker;
    try {
      w = new InlineKMeansWorker();
    } catch {
      return this.fallbackInit();
    }
    w.onmessage = (e: MessageEvent) => this.onMessage(e);
    w.onerror = () => this.failAll(new Error('kmeans worker crashed'));
    this.worker = w;
    // The sample may be a view into IVFIndex's large reused reservoir buffer; a
    // tight owned copy is transferred zero-copy and leaves the caller's data intact.
    const owned = this.cfg.sample.slice();
    return new Promise<Float32Array>((resolve, reject) => {
      this.initPending = { resolve, reject };
      w.postMessage(
        {
          type: 'init',
          sample: owned,
          trainCount: this.cfg.trainCount,
          nlist: this.cfg.nlist,
          dim: this.cfg.dim,
          seed: this.cfg.seed,
        },
        [owned.buffer],
      );
    });
  }

  update(assign: Uint32Array, iter: number): Promise<Float32Array> {
    if (!this.worker) {
      // Fallback / never-booted: run in-thread. Same code, just blocking.
      const centroids = new Float32Array(this.cfg.nlist * this.cfg.dim);
      updateCentroids(
        this.cfg.sample,
        this.cfg.trainCount,
        assign,
        centroids,
        this.cfg.nlist,
        this.cfg.dim,
        this.cfg.seed + iter + 1,
      );
      return Promise.resolve(centroids);
    }
    const id = this.nextId++;
    const owned = assign.slice();
    return new Promise<Float32Array>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ type: 'update', id, assign: owned, iter }, [owned.buffer]);
    });
  }

  private onMessage(e: MessageEvent): void {
    const msg = e.data as { type: string; id?: number; centroids?: Float32Array };
    if (msg.type === 'ready') {
      this.initPending?.resolve(msg.centroids!);
      this.initPending = null;
      return;
    }
    if (msg.type !== 'updated' || msg.id === undefined) return;
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    p.resolve(msg.centroids!);
  }

  private failAll(err: unknown): void {
    this.initPending?.reject(err);
    this.initPending = null;
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  private fallbackInit(): Promise<Float32Array> {
    this.fellBack = true;
    return Promise.resolve(
      randomInitCentroids(this.cfg.sample, this.cfg.trainCount, this.cfg.nlist, this.cfg.dim, this.cfg.seed),
    );
  }

  dispose(): void {
    this.failAll(new Error('trainer disposed'));
    this.worker?.terminate();
    this.worker = null;
  }
}

export function createKMeansTrainer(cfg: KMeansTrainerConfig): KMeansTrainer {
  return new WorkerKMeansTrainer(cfg);
}
