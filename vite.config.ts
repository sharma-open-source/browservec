import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  // Dev server opens the demo directly; the root index.html also redirects there.
  server: { open: '/demo/' },
  // The ingest quantization worker (§NFR-8) is pulled in via `?worker&inline`, so
  // Vite base64-inlines it into the single-file dist rather than emitting a
  // separate asset consumers must host. Keep the default iife worker format —
  // ES-format workers can't be inlined (browsers reject module workers from data
  // URLs), so setting format:'es' would silently emit a separate chunk instead.
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'BrowserVec',
      fileName: 'browservec',
      formats: ['es'],
    },
    sourcemap: true,
    target: 'es2022',
  },
});
