// Real semantic embedder via @xenova/transformers.
//
// Loaded lazily through a dynamic import so the model + library are NEVER in the
// core bundle — you only pay for them if you call this. The model weights are
// fetched once and cached by the browser (IndexedDB), so subsequent loads are
// offline. Defaults to all-MiniLM-L6-v2 (384-dim, mean-pooled + normalized),
// which pairs with `dimension: 384` and cosine.
//
// Requires the peer dependency:  npm install @xenova/transformers

import type { Embedder } from '../types.js';

export interface TransformersEmbedderOptions {
  /** Model id. Default 'Xenova/all-MiniLM-L6-v2' (384-dim sentence embeddings). */
  model?: string;
  /** Output dimension of the chosen model. Default 384. */
  dimension?: number;
  /** Backend: 'wasm' (default, broad support) or 'webgpu' (faster, newer browsers). */
  device?: 'wasm' | 'webgpu';
  /** Optional progress callback for the one-time model download. */
  onProgress?: (info: unknown) => void;
}

export async function transformersEmbedder(options: TransformersEmbedderOptions = {}): Promise<Embedder> {
  const model = options.model ?? 'Xenova/all-MiniLM-L6-v2';
  const dimension = options.dimension ?? 384;

  // Non-literal specifier + vite-ignore: keep this an external runtime import so
  // neither the library nor the model is pulled into the core bundle.
  const spec = '@xenova/transformers';
  const mod: any = await import(/* @vite-ignore */ spec as string);
  // transformers.js defaults env.allowLocalModels to true, so it probes a
  // local `/models/...` path before falling back to the Hub. This library
  // never ships local model files, and many dev/static servers answer a
  // missing path with an HTML fallback page rather than a 404 — which then
  // fails JSON parsing with a confusing "Unexpected token '<'" error instead
  // of a clear 404. Going straight to the Hub avoids that path entirely.
  mod.env.allowLocalModels = false;
  const extractor = await mod.pipeline('feature-extraction', model, {
    device: options.device,
    progress_callback: options.onProgress,
  });

  return {
    dimension,
    async embed(texts: string[]): Promise<Float32Array[]> {
      if (texts.length === 0) return [];
      const out = await extractor(texts, { pooling: 'mean', normalize: true });
      // `out.data` is a flat Float32Array of shape [texts.length, dimension].
      const data: Float32Array = out.data;
      const res: Float32Array[] = [];
      for (let i = 0; i < texts.length; i++) {
        res.push(new Float32Array(data.subarray(i * dimension, i * dimension + dimension)));
      }
      return res;
    },
  };
}
