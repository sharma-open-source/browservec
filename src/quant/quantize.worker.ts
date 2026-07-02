// Ingest quantization Web Worker . Runs the CPU-heavy
// rotate + quantize transform off the main thread so a large ingest doesn't
// freeze the UI. It owns no GPU state — it takes fp32 rows, returns packed codes
// (+ optional rotated rows for IVF training), and the main thread does the GPU
// upload. Bundled and inlined by Vite via the `?worker&inline` import in
// encoder.ts, so it ships inside the single-file dist with no separate asset.

import { BatchEncoder, type EncodeConfig } from './encode.js';

// In a worker `self` is the DedicatedWorkerGlobalScope (postMessage takes a
// transfer list), but the DOM lib types it as Window. Rather than pull in the
// conflicting WebWorker lib, alias to the minimal surface we use.
interface WorkerScope {
  onmessage: ((e: MessageEvent<InMessage>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}
const ctx = self as unknown as WorkerScope;

type InitMessage = { type: 'init'; config: EncodeConfig };
type EncodeMessage = {
  type: 'encode';
  id: number;
  data: Float32Array;
  count: number;
  wantRotated: boolean;
};
type InMessage = InitMessage | EncodeMessage;

let encoder: BatchEncoder | null = null;

ctx.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    encoder = new BatchEncoder(msg.config);
    ctx.postMessage({ type: 'ready', paddedDim: encoder.paddedDim, wordsPerRow: encoder.wordsPerRow });
    return;
  }
  // encode
  if (!encoder) throw new Error('quantize worker got encode before init');
  const out = encoder.encode(msg.data, msg.count, msg.wantRotated);
  const transfer: Transferable[] = [out.words.buffer, out.scales.buffer];
  if (out.rotated) transfer.push(out.rotated.buffer);
  ctx.postMessage({ type: 'encoded', id: msg.id, ...out }, transfer);
};
