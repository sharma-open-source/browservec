// Zero-dependency feature-hashing embedder.
//
// Maps text to a fixed-dimension vector with the hashing trick: each token (and
// token bigram) is hashed to a coordinate with a signed contribution, then the
// vector is L2-normalized. It needs no model download and runs fully offline, so
// it's ideal for tests, demos, and keyword-ish retrieval — but it is NOT semantic
// (no notion of synonyms or meaning). Swap in `transformersEmbedder` for real
// semantic embeddings while keeping the exact same store/query code.

import type { Embedder } from '../types.js';

export interface HashingEmbedderOptions {
  /** Output dimension. Default 384 (matches all-MiniLM-L6-v2 so models are swappable). */
  dimension?: number;
  /** Add adjacent-token bigrams as features (captures short phrases). Default true. */
  bigrams?: boolean;
}

/** FNV-1a 32-bit hash of a string with a salt, for stable feature indexing. */
function fnv1a(str: string, salt: number): number {
  let h = (0x811c9dc5 ^ salt) >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 0);
}

export function hashingEmbedder(options: HashingEmbedderOptions = {}): Embedder {
  const dimension = options.dimension ?? 384;
  const bigrams = options.bigrams ?? true;

  function embedOne(text: string): Float32Array {
    const v = new Float32Array(dimension);
    const toks = tokenize(text);
    const add = (feature: string) => {
      const h = fnv1a(feature, 0);
      const idx = h % dimension;
      // Independent sign hash so collisions tend to cancel rather than accumulate.
      const sign = (fnv1a(feature, 0x9e3779b9) & 1) === 0 ? 1 : -1;
      v[idx]! += sign;
    };
    for (let i = 0; i < toks.length; i++) {
      add(toks[i]!);
      if (bigrams && i + 1 < toks.length) add(toks[i]! + ' ' + toks[i + 1]!);
    }
    // L2 normalize (cosine-ready). Empty text → zero vector (valid, just no signal).
    let s = 0;
    for (let i = 0; i < dimension; i++) s += v[i]! * v[i]!;
    const inv = 1 / (Math.sqrt(s) || 1);
    for (let i = 0; i < dimension; i++) v[i]! *= inv;
    return v;
  }

  return {
    dimension,
    embed(texts: string[]): Promise<Float32Array[]> {
      return Promise.resolve(texts.map(embedOne));
    },
  };
}
