// Persistence backend interface + selector (persist.backend 'auto').

import { opfsAvailable, opfsBackend } from './opfs.js';
import { indexedDBAvailable, indexedDBBackend } from './indexeddb.js';

export type BackendKind = 'opfs' | 'indexeddb';

export interface PersistenceBackend {
  readonly kind: BackendKind;
  write(name: string, data: ArrayBuffer): Promise<void>;
  read(name: string): Promise<ArrayBuffer | null>;
  remove(name: string): Promise<void>;
}

/** Resolve a backend. 'auto' prefers OPFS (faster, larger) and falls back to IDB. */
export function selectBackend(pref: BackendKind | 'auto'): PersistenceBackend {
  if (pref === 'opfs') {
    if (!opfsAvailable()) throw new Error('OPFS backend requested but not available in this browser');
    return opfsBackend;
  }
  if (pref === 'indexeddb') {
    if (!indexedDBAvailable()) throw new Error('IndexedDB backend requested but not available');
    return indexedDBBackend;
  }
  if (opfsAvailable()) return opfsBackend;
  if (indexedDBAvailable()) return indexedDBBackend;
  throw new Error('no persistence backend available (need OPFS or IndexedDB)');
}
