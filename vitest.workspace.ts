import { defineWorkspace } from 'vitest/config';

// Chromium flags that enable WebGPU in headless CI (Linux runners have no real
// GPU — Vulkan routes to a software adapter). On macOS the default Metal backend
// just works, so only the unsafe-webgpu opt-in is needed.
const webgpuFlags =
  process.platform === 'linux'
    ? [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-angle=vulkan',
        '--disable-vulkan-surface',
      ]
    : ['--enable-unsafe-webgpu'];

export default defineWorkspace([
  {
    test: {
      name: 'unit',
      environment: 'node',
      include: ['tests/unit/**/*.test.ts'],
    },
  },
  {
    test: {
      name: 'browser',
      include: ['tests/browser/**/*.test.ts'],
      testTimeout: 120_000,
      hookTimeout: 120_000,
      browser: {
        enabled: true,
        provider: 'playwright',
        name: 'chromium',
        headless: true,
        screenshotFailures: false,
        providerOptions: {
          launch: { args: webgpuFlags },
        },
      },
    },
  },
]);
