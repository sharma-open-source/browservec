// OPFS persistence backend (persist/opfs.ts).
// One file per store under a `browservec/` directory in the origin-private FS.

import type { PersistenceBackend } from './backend.js';

const DIR = 'browservec';

function fileName(name: string): string {
  return `${name}.bvec`;
}

export function opfsAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory;
}

async function dirHandle(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}

export const opfsBackend: PersistenceBackend = {
  kind: 'opfs',

  async write(name: string, data: ArrayBuffer): Promise<void> {
    const dir = await dirHandle();
    const file = await dir.getFileHandle(fileName(name), { create: true });
    const writable = await file.createWritable();
    try {
      await writable.write(data);
    } finally {
      await writable.close();
    }
  },

  async read(name: string): Promise<ArrayBuffer | null> {
    try {
      const dir = await dirHandle();
      const file = await dir.getFileHandle(fileName(name), { create: false });
      const blob = await file.getFile();
      return await blob.arrayBuffer();
    } catch (err) {
      if (err instanceof DOMException && err.name === 'NotFoundError') return null;
      throw err;
    }
  },

  async remove(name: string): Promise<void> {
    try {
      const dir = await dirHandle();
      await dir.removeEntry(fileName(name));
    } catch (err) {
      if (err instanceof DOMException && err.name === 'NotFoundError') return;
      throw err;
    }
  },
};
