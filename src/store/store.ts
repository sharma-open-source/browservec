// Store manager: id<->row mapping, metadata, normalization, dimension validation,
// and the CPU-side raw vector buffer that is the persistence source of truth
// (store/store.ts, NFR-11 "OPFS is source of truth").

import type { Metadata, Metric, Vector } from '../types.js';

export interface RowEntry {
  id: string;
  row: number;
  metadata?: Metadata;
}

export class Store {
  private readonly byId = new Map<string, RowEntry>();
  private readonly byRow: RowEntry[] = [];
  // Rows that have been deleted. They stay in `byRow`/`raw`/the GPU index as
  // tombstones (filtered out of queries) and are only physically dropped when the
  // snapshot is compacted on save/export — cheap deletes, memory reclaimed on reload.
  private readonly deletedRows = new Set<number>();

  // Packed, post-normalization vectors (exactly what is searched/persisted).
  private raw = new Float32Array(0);
  private rawRows = 0;

  constructor(
    readonly dimension: number,
    readonly metric: Metric,
    readonly normalize: boolean,
  ) {}

  /** Live (non-deleted) vector count. */
  get count(): number {
    return this.byId.size;
  }

  /** Total rows including tombstones (matches the GPU index row count). */
  get rowCount(): number {
    return this.byRow.length;
  }

  get deletedCount(): number {
    return this.deletedRows.size;
  }

  isDeleted(row: number): boolean {
    return this.deletedRows.has(row);
  }

  has(id: string): boolean {
    return this.byId.has(id);
  }

  entryById(id: string): RowEntry | undefined {
    return this.byId.get(id);
  }

  entryByRow(row: number): RowEntry | undefined {
    return this.byRow[row];
  }

  /**
   * Tombstone the row for `id`. Returns false if the id isn't present. The row's
   * vector stays in the buffers (filtered from results) until snapshot compaction.
   */
  delete(id: string): boolean {
    const entry = this.byId.get(id);
    if (!entry) return false;
    this.byId.delete(id);
    this.deletedRows.add(entry.row);
    return true;
  }

  /** Live entries in row order (deleted rows skipped), aligned with {@link liveVectors}. */
  liveEntries(): RowEntry[] {
    if (this.deletedRows.size === 0) return this.byRow.slice(0, this.rawRows);
    const out: RowEntry[] = [];
    for (let row = 0; row < this.byRow.length; row++) {
      if (!this.deletedRows.has(row)) out.push(this.byRow[row]!);
    }
    return out;
  }

  /** Compacted live vectors (deleted rows removed), aligned with {@link liveEntries}. */
  liveVectors(): Float32Array {
    if (this.deletedRows.size === 0) return this.rawView();
    const dim = this.dimension;
    const out = new Float32Array((this.byId.size) * dim);
    let w = 0;
    for (let row = 0; row < this.byRow.length; row++) {
      if (this.deletedRows.has(row)) continue;
      out.set(this.raw.subarray(row * dim, row * dim + dim), w * dim);
      w++;
    }
    return out;
  }

  /** The packed raw vectors for the current rows (count * dim floats). */
  rawView(): Float32Array {
    return this.raw.subarray(0, this.rawRows * this.dimension);
  }

  /** Copy of one row's vector, or undefined if the row doesn't exist. */
  vectorAt(row: number): Float32Array | undefined {
    if (row < 0 || row >= this.rawRows) return undefined;
    const off = row * this.dimension;
    return this.raw.slice(off, off + this.dimension);
  }

  /** Validate length and (optionally) L2-normalize, returning a fresh Float32Array. */
  prepare(vector: Vector): Float32Array {
    const v = vector instanceof Float32Array ? new Float32Array(vector) : Float32Array.from(vector);
    if (v.length !== this.dimension) {
      throw new Error(`vector dim ${v.length} != store dim ${this.dimension}`);
    }
    if (this.normalize) normalizeInPlace(v);
    return v;
  }

  /**
   * Register a new id, store its prepared vector in the raw buffer, and return
   * the assigned row. Throws on duplicate id. `prepared` must already be the
   * right dimension and (if applicable) normalized.
   */
  insert(id: string, prepared: Float32Array, metadata?: Metadata): number {
    if (this.byId.has(id)) throw new Error(`duplicate id: ${id}`);
    const row = this.byRow.length;
    const entry: RowEntry = { id, row, ...(metadata !== undefined ? { metadata } : {}) };
    this.byId.set(id, entry);
    this.byRow.push(entry);

    this.ensureRawCapacity(row + 1);
    this.raw.set(prepared, row * this.dimension);
    this.rawRows = row + 1;
    return row;
  }

  /** Drop everything — used before an in-place compaction rebuilds the store dense. */
  clear(): void {
    this.byId.clear();
    this.byRow.length = 0;
    this.deletedRows.clear();
    this.raw = new Float32Array(0);
    this.rawRows = 0;
  }

  private ensureRawCapacity(rows: number): void {
    const needed = rows * this.dimension;
    if (needed <= this.raw.length) return;
    const nextRows = Math.max(rows, Math.ceil((this.raw.length / this.dimension) * 1.5), 1024);
    const next = new Float32Array(nextRows * this.dimension);
    next.set(this.raw.subarray(0, this.rawRows * this.dimension));
    this.raw = next;
  }
}

export function normalizeInPlace(v: Float32Array): void {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i]! * v[i]!;
  const norm = Math.sqrt(sum);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < v.length; i++) v[i]! *= inv;
  }
}
