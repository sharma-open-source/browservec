// IVF k-means centroid-update Web Worker (§NFR-8). The GPU-assisted Lloyd loop in
// IVFIndex/IVFQuantIndex assigns points on the GPU but recomputes the cluster
// means on the CPU each iteration — an O(trainCount·dim) sweep that, at dim≈768,
// blocks the main thread for tens of ms per iteration and janks the UI during a
// build. This worker owns a copy of the training sample and does that mean-update
// (and the one-time random init) off-thread, so only the small centroid array
// crosses back each iteration. It owns no GPU state. Bundled and inlined by Vite
// via the `?worker&inline` import in kmeansTrainer.ts, so it ships inside the
// single-file dist with no separate asset.

import { randomInitCentroids, updateCentroids } from './kmeans.js';

// In a worker `self` is the DedicatedWorkerGlobalScope (postMessage takes a
// transfer list), but the DOM lib types it as Window. Alias to the minimal
// surface we use rather than pull in the conflicting WebWorker lib.
interface WorkerScope {
  onmessage: ((e: MessageEvent<InMessage>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}
const ctx = self as unknown as WorkerScope;

type InitMessage = {
  type: 'init';
  sample: Float32Array; // trainCount * dim, transferred to the worker
  trainCount: number;
  nlist: number;
  dim: number;
  seed: number;
};
type UpdateMessage = { type: 'update'; id: number; assign: Uint32Array; iter: number };
type InMessage = InitMessage | UpdateMessage;

let sample: Float32Array | null = null;
let trainCount = 0;
let nlist = 0;
let dim = 0;
let seed = 0;

ctx.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    sample = msg.sample;
    trainCount = msg.trainCount;
    nlist = msg.nlist;
    dim = msg.dim;
    seed = msg.seed;
    const centroids = randomInitCentroids(sample, trainCount, nlist, dim, seed);
    ctx.postMessage({ type: 'ready', centroids }, [centroids.buffer]);
    return;
  }
  // update
  if (!sample) throw new Error('kmeans worker got update before init');
  // Re-seed for the empty-cluster path exactly as the main-thread loop does:
  // updateCentroids uses seed+iter+1 on iteration `iter`.
  const centroids = new Float32Array(nlist * dim);
  updateCentroids(sample, trainCount, msg.assign, centroids, nlist, dim, seed + msg.iter + 1);
  ctx.postMessage({ type: 'updated', id: msg.id, centroids }, [centroids.buffer]);
};
