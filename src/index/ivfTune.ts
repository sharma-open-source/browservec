// Auto-tuning for IVF's recall/latency knob (nprobe).
//
// The insight that makes this cheap: an IVF query misses a true neighbour only
// when the neighbour's cluster isn't probed, and clusters are probed in
// query→centroid score order. So for one sample query, ONE exact scan (ground
// truth) plus one centroid ranking tells us the recall at EVERY possible nprobe
// at once: a ground-truth row is found at nprobe p iff the rank of its cluster
// is < p. Tuning is then a histogram over those ranks — no per-nprobe
// re-querying, no binary search.
//
// This holds exactly for fp32 IVF (probed scores are exact, so any probed
// ground-truth row outranks every non-ground-truth candidate). For IVF×quant
// the same argument applies with quantized scores, so the tuner measures the
// IVF-induced recall loss in the space the index actually scores in — which is
// precisely the loss nprobe controls (the quantization loss is handled
// separately by the exact fp32 re-rank).

export interface TunedNprobe {
  /** Smallest nprobe whose estimated recall meets the target. */
  nprobe: number;
  /** Estimated recall@k at that nprobe (fraction of ground-truth hits probed). */
  recall: number;
}

/**
 * Pick the smallest nprobe meeting `targetRecall` from the observed cluster
 * ranks of ground-truth neighbours. `neededRanks[i]` is, for one ground-truth
 * hit, the probe rank (0 = nearest cluster to the query) of the cluster that
 * hit lives in; hits from all sample queries are pooled.
 */
export function chooseNprobe(neededRanks: ArrayLike<number>, nlist: number, targetRecall: number): TunedNprobe {
  const total = neededRanks.length;
  if (total === 0 || nlist <= 1) return { nprobe: Math.max(1, nlist), recall: 1 };
  const hist = new Uint32Array(nlist);
  for (let i = 0; i < total; i++) hist[neededRanks[i]!]!++;
  let cum = 0;
  for (let p = 1; p <= nlist; p++) {
    cum += hist[p - 1]!;
    if (cum / total >= targetRecall) return { nprobe: p, recall: cum / total };
  }
  return { nprobe: nlist, recall: 1 }; // probing everything is exact by construction
}

/**
 * Deterministic evenly-spaced pick of `want` sample indices out of `filled`
 * (the reservoir is already a uniform sample of the corpus, so a stride keeps
 * tuning reproducible without another RNG).
 */
export function pickTuneQueries(filled: number, want: number): number[] {
  const n = Math.min(filled, want);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(Math.floor((i * filled) / n));
  return out;
}

/** Sample queries drawn per tuning pass. */
export const TUNE_QUERIES = 32;
/** Recall is measured @k=10 (clamped to the corpus size). */
export const TUNE_K = 10;
