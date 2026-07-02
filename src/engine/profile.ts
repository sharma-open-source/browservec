// Per-query GPU-wait tracing (engine/profile.ts).
//
// Browsers expose no GPU utilization or per-pass timing API by default, but the
// query paths all share one structure: synchronous JS work (encode, gather,
// re-rank, merge) around one or more `mapAsync` awaits that resolve only after
// the submitted GPU work has completed and the readback is mapped. Timing those
// awaits therefore captures "GPU execution + readback transfer + queue
// scheduling" — everything that is not JS. BrowserVec.query() resets the
// collector, runs the query, and reports cpu = total − gpuWait.
//
// The collector is a module singleton: queries against a single store are
// awaited serially, and JS is single-threaded, so per-query attribution holds.
// Interleaving queries across *multiple* stores in flight at once can blur the
// split between them — acceptable for telemetry, which this is.

export const queryTrace = {
  gpuWaitMs: 0,
  reset(): void {
    this.gpuWaitMs = 0;
  },
};

/** Await a GPU readback promise, attributing the wait to the current query. */
export async function tracedGpuWait<T>(p: Promise<T>): Promise<T> {
  const t0 = performance.now();
  try {
    return await p;
  } finally {
    queryTrace.gpuWaitMs += performance.now() - t0;
  }
}
