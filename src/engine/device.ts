// WebGPU device acquisition + limits probing (engine/device.ts).

export interface DeviceContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  limits: GPUSupportedLimits;
  /** True once the device has been lost; engine should re-init from source of truth. */
  lost: boolean;
}

export class WebGPUUnavailableError extends Error {
  constructor(reason: string) {
    super(`WebGPU unavailable: ${reason}`);
    this.name = 'WebGPUUnavailableError';
  }
}

export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export async function acquireDevice(existing?: GPUDevice): Promise<DeviceContext> {
  if (existing) {
    const ctx: DeviceContext = {
      // adapter is unknown when a device is injected; callers that need it should
      // pass none. We expose limits from the device itself.
      adapter: undefined as unknown as GPUAdapter,
      device: existing,
      limits: existing.limits,
      lost: false,
    };
    wireDeviceLost(ctx);
    return ctx;
  }

  if (!isWebGPUAvailable()) {
    throw new WebGPUUnavailableError('navigator.gpu is not present in this browser');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGPUUnavailableError('no GPUAdapter (try enabling WebGPU)');

  const device = await adapter.requestDevice({
    requiredLimits: pickLimits(adapter.limits),
  });

  const ctx: DeviceContext = { adapter, device, limits: device.limits, lost: false };
  wireDeviceLost(ctx);
  return ctx;
}

/** Request the headroom we want, clamped to what the adapter actually offers. */
function pickLimits(adapterLimits: GPUSupportedLimits): Record<string, number> {
  const want = {
    maxStorageBufferBindingSize: 1 << 30, // 1 GiB if available
    maxBufferSize: 1 << 30,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
  };
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(want)) {
    const cap = (adapterLimits as unknown as Record<string, number>)[k];
    if (typeof cap === 'number') out[k] = Math.min(v, cap);
  }
  return out;
}

function wireDeviceLost(ctx: DeviceContext): void {
  void ctx.device.lost.then((info) => {
    ctx.lost = true;
    // 'destroyed' is an intentional teardown; anything else is a real loss.
    if (info.reason !== 'destroyed') {
      console.warn(`[browservec] GPU device lost: ${info.reason} — ${info.message}`);
    }
  });
}
