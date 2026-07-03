// Environment probe: records what this browser/CI runner actually supports so a
// skipped GPU suite is visible in the logs, and pins the invariants the other
// suites rely on (WASM everywhere, isSupported() telling the truth).

import { describe, expect, it } from 'vitest';
import { BrowserVec, webgpuAvailable } from './helpers';

describe('BrowserVec.isSupported', () => {
  it('reports capabilities consistent with the environment', async () => {
    const info = BrowserVec.isSupported();
    // eslint-disable-next-line no-console
    console.log('[browservec tests] support:', JSON.stringify(info));

    // WASM is table stakes in any browser we test.
    expect(info.wasm).toBe(true);
    // isSupported().webgpu is a cheap presence probe; if it says no, an adapter
    // request must also fail.
    if (!info.webgpu) expect(await webgpuAvailable()).toBe(false);
    expect(typeof info.opfs).toBe('boolean');
  });
});
