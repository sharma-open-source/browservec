// Growable GPU storage buffer for the corpus + small helpers (engine/buffers.ts).
//
// keeps the whole corpus in one storage buffer. When rows * dim * 4 bytes would
// exceed the device limit we throw with a clear message; chunking across dispatches
// is an M4 concern (NFR-10). Buffer pooling for query/result scratch lives here too.

export class CorpusBuffer {
  private buffer: GPUBuffer | null = null;
  private capacityRows = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly dim: number,
    private readonly maxBindingBytes: number,
  ) {}

  get rowStrideBytes(): number {
    return this.dim * 4; // fp32
  }

  get gpu(): GPUBuffer {
    if (!this.buffer) throw new Error('CorpusBuffer not allocated yet');
    return this.buffer;
  }

  /** Ensure capacity for at least `rows`, growing (and copying) geometrically. */
  ensureCapacity(rows: number): void {
    if (rows <= this.capacityRows) return;

    const nextRows = Math.max(rows, Math.ceil(this.capacityRows * 1.5), 1024);
    const bytes = nextRows * this.rowStrideBytes;
    if (bytes > this.maxBindingBytes) {
      const maxRows = Math.floor(this.maxBindingBytes / this.rowStrideBytes);
      throw new Error(
        `corpus would need ${bytes} bytes (>${this.maxBindingBytes} device limit). ` +
          `Max ~${maxRows} rows at dim=${this.dim} in M1. Quantization (§6) and ` +
          `chunking (§NFR-10) raise this; not yet implemented.`,
      );
    }

    const next = this.device.createBuffer({
      label: 'browservec:corpus',
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    if (this.buffer) {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.buffer, 0, next, 0, this.capacityRows * this.rowStrideBytes);
      this.device.queue.submit([enc.finish()]);
      this.buffer.destroy();
    }

    this.buffer = next;
    this.capacityRows = nextRows;
  }

  /** Write `data` (length dim) at row index `row`. */
  writeRow(row: number, data: Float32Array): void {
    this.device.queue.writeBuffer(this.gpu, row * this.rowStrideBytes, data);
  }

  /** Write a contiguous block of rows starting at `startRow`. */
  writeRows(startRow: number, data: Float32Array): void {
    this.device.queue.writeBuffer(this.gpu, startRow * this.rowStrideBytes, data);
  }

  destroy(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.capacityRows = 0;
  }
}

/**
 * Corpus split across several GPU buffers. A single
 * storage buffer is capped at `maxStorageBufferBindingSize` — only 128 MiB by
 * default on many devices — so past ~40k fp32×768 rows (or ~500k int8 rows) one
 * buffer overflows. This spreads rows over N fixed-stride "chunks" (each ≤ the
 * limit); a query dispatches the distance kernel once per chunk into a shared
 * scores buffer. Row payload is generic: fp32 vectors (`Float32Array`, 4 B/elem)
 * for the flat path, or packed u32 codes (`Uint32Array`, 4 B/elem) for quant.
 *
 * Layout invariant: every chunk but the last holds exactly `rowsPerChunk` rows,
 * so global row r lives in chunk `floor(r / rowsPerChunk)` at local row
 * `r % rowsPerChunk`. The last (current) chunk grows geometrically up to
 * `rowsPerChunk` before a new one is opened — so a small store still uses a
 * small buffer rather than eagerly allocating a full-size chunk.
 */
export type RowData = Float32Array | Uint32Array;

export class ChunkedCorpus {
  private buffers: GPUBuffer[] = [];
  private caps: number[] = []; // capacity (rows) of each chunk
  private rows = 0;
  readonly rowsPerChunk: number;
  private readonly strideBytes: number;

  constructor(
    private readonly device: GPUDevice,
    /** Elements (fp32 coords or u32 code-words) per row. */
    private readonly elementsPerRow: number,
    /** Bytes per element (4 for both Float32Array and Uint32Array). */
    bytesPerElement: number,
    maxBindingBytes: number,
    /** Force a smaller chunk size (rows) — for tests/demos, to exercise chunking cheaply. */
    forcedRowsPerChunk?: number,
  ) {
    this.strideBytes = elementsPerRow * bytesPerElement;
    const limit = Math.max(1, Math.floor(maxBindingBytes / this.strideBytes));
    this.rowsPerChunk = forcedRowsPerChunk
      ? Math.max(1, Math.min(forcedRowsPerChunk, limit))
      : limit;
  }

  get count(): number {
    return this.rows;
  }
  get chunkCount(): number {
    return this.buffers.length;
  }

  private totalCapacity(): number {
    if (this.buffers.length === 0) return 0;
    return (this.buffers.length - 1) * this.rowsPerChunk + this.caps[this.caps.length - 1]!;
  }

  private ensure(target: number): void {
    while (this.totalCapacity() < target) {
      const last = this.buffers.length - 1;
      const lastFull = last < 0 || this.caps[last] === this.rowsPerChunk;
      if (lastFull) {
        // Open a new chunk, sized geometrically to what this chunk needs.
        const base = this.buffers.length * this.rowsPerChunk;
        const need = Math.min(this.rowsPerChunk, target - base);
        const cap = Math.min(this.rowsPerChunk, Math.max(1024, need));
        this.buffers.push(this.newChunk(cap));
        this.caps.push(cap);
      } else {
        // Grow the current (last) chunk toward rowsPerChunk, copying its rows.
        const base = last * this.rowsPerChunk;
        const need = Math.min(this.rowsPerChunk, target - base);
        const newCap = Math.min(this.rowsPerChunk, Math.max(Math.ceil(this.caps[last]! * 2), need));
        const nb = this.newChunk(newCap);
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers[last]!, 0, nb, 0, this.caps[last]! * this.strideBytes);
        this.device.queue.submit([enc.finish()]);
        this.buffers[last]!.destroy();
        this.buffers[last] = nb;
        this.caps[last] = newCap;
      }
    }
  }

  private newChunk(capRows: number): GPUBuffer {
    return this.device.createBuffer({
      label: `browservec:corpus-chunk-${this.buffers.length}`,
      size: capRows * this.strideBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /** Append `count` rows packed in `data`, splitting across chunk boundaries. */
  append(data: RowData, count: number): void {
    this.ensure(this.rows + count);
    let written = 0;
    while (written < count) {
      const globalRow = this.rows + written;
      const chunkIdx = Math.floor(globalRow / this.rowsPerChunk);
      const localRow = globalRow % this.rowsPerChunk;
      const room = this.rowsPerChunk - localRow;
      const n = Math.min(room, count - written);
      this.device.queue.writeBuffer(
        this.buffers[chunkIdx]!,
        localRow * this.strideBytes,
        data,
        written * this.elementsPerRow,
        n * this.elementsPerRow,
      );
      written += n;
    }
    this.rows += count;
  }

  /** Overwrite a single existing row (update path). */
  writeRow(row: number, data: RowData): void {
    const chunkIdx = Math.floor(row / this.rowsPerChunk);
    const localRow = row % this.rowsPerChunk;
    this.device.queue.writeBuffer(this.buffers[chunkIdx]!, localRow * this.strideBytes, data);
  }

  /** The GPU buffer holding chunk `i` (chunk of global row r is `floor(r / rowsPerChunk)`). */
  bufferAt(i: number): GPUBuffer {
    return this.buffers[i]!;
  }

  /** Visit each chunk's buffer with its global base row + live row count. */
  eachChunk(cb: (buffer: GPUBuffer, baseRow: number, rowCount: number) => void): void {
    for (let i = 0; i < this.buffers.length; i++) {
      const baseRow = i * this.rowsPerChunk;
      const rowCount = Math.min(this.rowsPerChunk, this.rows - baseRow);
      if (rowCount <= 0) break;
      cb(this.buffers[i]!, baseRow, rowCount);
    }
  }

  destroy(): void {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
    this.caps = [];
    this.rows = 0;
  }
}

/** Create a reusable uniform/storage buffer sized to hold one query vector. */
export function createQueryBuffer(device: GPUDevice, dim: number): GPUBuffer {
  return device.createBuffer({
    label: 'browservec:query',
    size: dim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

/** Result scores buffer (one f32 per corpus row) + a mappable readback buffer. */
export interface ScorePair {
  scores: GPUBuffer; // STORAGE, written by the kernel
  readback: GPUBuffer; // MAP_READ, copied from scores
}

export function createScoreBuffers(device: GPUDevice, rows: number): ScorePair {
  const size = Math.max(rows, 1) * 4;
  return {
    scores: device.createBuffer({
      label: 'browservec:scores',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    readback: device.createBuffer({
      label: 'browservec:scores-readback',
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  };
}
