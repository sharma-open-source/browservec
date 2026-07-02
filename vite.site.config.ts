import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { readdirSync } from 'node:fs';

const root = __dirname;
const exampleEntries = readdirSync(resolve(root, 'examples'))
  .filter((f) => f.endsWith('.html'))
  .reduce(
    (acc, f) => {
      acc[`examples/${f.replace(/\.html$/, '')}`] = resolve(root, 'examples', f);
      return acc;
    },
    {}
  );

// Builds the public site (landing page + demo + examples) as static, deployable
// HTML — unlike vite.config.ts (the library build), this resolves every
// `../src/index.ts` import in demo/examples into real bundled JS so the pages
// work on plain static hosting (GitHub Pages) instead of only Vite's dev server.
export default defineConfig({
  root,
  build: {
    outDir: resolve(root, 'site'),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(root, 'index.html'),
        demo: resolve(root, 'demo/index.html'),
        ...exampleEntries,
      },
    },
  },
});
