// WASM-SIMD brute-force kernel for the CPU fallback (§NFR-7).
//
// This is the real teeth behind `fallback: 'wasm'`. The module (hand-assembled,
// 526 bytes) exports two functions over its own linear memory:
//   dot(dataPtr, qPtr, outPtr, rows, dim) — out[r] = Σ data[r]·q   (cosine/dot)
//   l2 (dataPtr, qPtr, outPtr, rows, dim) — out[r] = -Σ (data[r]-q)²  (higher = closer)
// Each inner loop accumulates four lanes at a time with f32x4 (v128), with a
// scalar tail for dim % 4. Pointers are BYTE offsets into the module's memory,
// which CpuIndex uses directly as the corpus backing store — so there's no
// per-query copy and the SIMD win (~7× over the unrolled scalar JS loop, matched
// to f32 precision) is realized end to end.
//
// The bytes were produced from src/fallback/kernel.wat via the emitter in the
// project scratchpad; regenerate both together if the kernel ever changes.

/** Signature of the exported kernels: (dataPtr, qPtr, outPtr, rows, dim) → void. */
export type KernelFn = (data: number, q: number, out: number, rows: number, dim: number) => void;

export interface SimdInstance {
  mem: WebAssembly.Memory;
  dot: KernelFn;
  l2: KernelFn;
}

// base64 of the assembled module. Decoded once below.
const WASM_BASE64 =
  'AGFzbQEAAAABCQFgBX9/f39/AAMDAgAABQMBAAEHEgMDbWVtAgADZG90AAACbDIAAQraAwLiAQMFfwF7AX0gBEF8cSEIIARBBGwhCUEAIQUCQANAIAUgA04NASAAIAUgCWxqIQb9DAAAAAAAAAAAAAAAAAAAAAAhCkEAIQcCQANAIAcgCE4NASAKIAYgB0EEbGr9AAAAIAEgB0EEbGr9AAAA/eYB/eQBIQogB0EEaiEHDAALCyAK/R8AIAr9HwGSIAr9HwIgCv0fA5KSIQsCQANAIAcgBE4NASALIAYgB0EEbGoqAAAgASAHQQRsaioAAJSSIQsgB0EBaiEHDAALCyACIAVBBGxqIAs4AAAgBUEBaiEFDAALCwvzAQMFfwJ7An0gBEF8cSEIIARBBGwhCUEAIQUCQANAIAUgA04NASAAIAUgCWxqIQb9DAAAAAAAAAAAAAAAAAAAAAAhCkEAIQcCQANAIAcgCE4NASAGIAdBBGxq/QAAACABIAdBBGxq/QAAAP3lASELIAogCyAL/eYB/eQBIQogB0EEaiEHDAALCyAK/R8AIAr9HwGSIAr9HwIgCv0fA5KSIQwCQANAIAcgBE4NASAGIAdBBGxqKgAAIAEgB0EEbGoqAACTIQ0gDCANIA2UkiEMIAdBAWohBwwACwsgAiAFQQRsaiAMjDgAACAFQQFqIQUMAAsLCw==';

function decode(b64: string): Uint8Array {
  if (typeof atob === 'function') {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  // Node.
  return new Uint8Array((globalThis as { Buffer?: { from(s: string, e: string): Uint8Array } }).Buffer!.from(b64, 'base64'));
}

// Compile once and cache. `null` means WASM or SIMD isn't available here — the
// caller (CpuIndex) then uses its pure-JS scalar path, still exact, just slower.
let cached: WebAssembly.Module | null | undefined;

function getModule(): WebAssembly.Module | null {
  if (cached !== undefined) return cached;
  try {
    const bytes = decode(WASM_BASE64);
    // validate() is false on engines without the SIMD proposal — the module uses v128.
    cached = WebAssembly.validate(bytes) ? new WebAssembly.Module(bytes) : null;
  } catch {
    cached = null;
  }
  return cached;
}

/** True if the SIMD kernel is usable in this environment. */
export function simdAvailable(): boolean {
  return getModule() !== null;
}

/**
 * Instantiate a fresh kernel instance with its own linear memory. Each CpuIndex
 * owns one so their corpora don't share memory. Returns null if unavailable.
 */
export function createSimd(): SimdInstance | null {
  const mod = getModule();
  if (!mod) return null;
  const inst = new WebAssembly.Instance(mod);
  const ex = inst.exports as unknown as { mem: WebAssembly.Memory; dot: KernelFn; l2: KernelFn };
  return { mem: ex.mem, dot: ex.dot, l2: ex.l2 };
}
