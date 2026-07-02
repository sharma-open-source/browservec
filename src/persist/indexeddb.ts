// IndexedDB persistence backend — fallback when OPFS is unavailable
// (persist/indexeddb.ts, Safari gaps).

import type { PersistenceBackend } from './backend.js';

const DB_NAME = 'browservec';
const STORE = 'snapshots';
const DB_VERSION = 1;

export function indexedDBAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDB().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const req = run(t.objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
        t.oncomplete = () => db.close();
        t.onabort = () => reject(t.error);
      }),
  );
}

export const indexedDBBackend: PersistenceBackend = {
  kind: 'indexeddb',

  async write(name: string, data: ArrayBuffer): Promise<void> {
    await tx('readwrite', (s) => s.put(data, name));
  },

  async read(name: string): Promise<ArrayBuffer | null> {
    const v = await tx<ArrayBuffer | undefined>('readonly', (s) => s.get(name));
    return v ?? null;
  },

  async remove(name: string): Promise<void> {
    await tx('readwrite', (s) => s.delete(name));
  },
};
