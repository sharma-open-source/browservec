import { describe, it, expect } from 'vitest';
import { serialize, deserialize, FORMAT_VERSION, type SerializeInput } from '../../src/persist/format';
import { HNSWGraph } from '../../src/index/hnswGraph';
import { mulberry32 } from '../../src/quant/prng';

function makeInput(count = 3, dimension = 4): SerializeInput {
  const rng = mulberry32(1);
  const vectors = new Float32Array(count * dimension);
  for (let i = 0; i < vectors.length; i++) vectors[i] = rng() * 2 - 1;
  return {
    dimension,
    metric: 'cosine' as const,
    normalize: true,
    count,
    entries: Array.from({ length: count }, (_, i) => ({
      id: `id-${i}`,
      metadata: { n: i, tag: `t${i}`, flag: i % 2 === 0, nil: null },
    })),
    vectors,
  };
}

describe('snapshot format', () => {
  it('round-trips vectors, ids, and metadata', () => {
    const input = makeInput();
    const snap = deserialize(serialize(input));
    expect(snap.dimension).toBe(4);
    expect(snap.metric).toBe('cosine');
    expect(snap.normalize).toBe(true);
    expect(snap.count).toBe(3);
    expect(snap.entries).toEqual(input.entries);
    expect(snap.vectors).toEqual(input.vectors);
    expect(snap.graph).toBeUndefined();
  });

  it('round-trips every metric code and the normalize flag', () => {
    for (const metric of ['cosine', 'dot', 'l2'] as const) {
      for (const normalize of [true, false]) {
        const snap = deserialize(serialize({ ...makeInput(), metric, normalize }));
        expect(snap.metric).toBe(metric);
        expect(snap.normalize).toBe(normalize);
      }
    }
  });

  it('round-trips unicode metadata and awkward JSON lengths', () => {
    const input = makeInput(1);
    input.entries = [{ id: '日本語-🚀', metadata: { s: 'héllo "quoted" \\ 中文' } }];
    const snap = deserialize(serialize(input));
    expect(snap.entries[0]!.id).toBe('日本語-🚀');
    expect(snap.entries[0]!.metadata).toEqual({ s: 'héllo "quoted" \\ 中文' });
    // Vectors must still be intact behind the padded metadata region.
    expect(snap.vectors).toEqual(input.vectors);
  });

  it('handles entries without metadata and an empty store', () => {
    const one = makeInput(1);
    one.entries = [{ id: 'bare' }];
    expect(deserialize(serialize(one)).entries[0]).toEqual({ id: 'bare' });

    const empty = { ...makeInput(0), entries: [], vectors: new Float32Array(0) };
    const snap = deserialize(serialize(empty));
    expect(snap.count).toBe(0);
    expect(snap.entries).toEqual([]);
  });

  it('writes v1 when no graph is present', () => {
    const buf = serialize(makeInput());
    expect(new DataView(buf).getUint32(4, true)).toBe(1);
  });

  it('writes v2 with a graph and round-trips it losslessly', () => {
    const dim = 8;
    const rows = 100;
    const rng = mulberry32(2);
    const vectors = new Float32Array(rows * dim);
    for (let i = 0; i < vectors.length; i++) vectors[i] = rng() * 2 - 1;
    const g = new HNSWGraph(dim, 'l2', { seed: 3, M: 8 });
    g.append(vectors, rows);
    const graph = g.serializeGraph();

    const buf = serialize({
      dimension: dim,
      metric: 'l2',
      normalize: false,
      count: rows,
      entries: Array.from({ length: rows }, (_, i) => ({ id: `v${i}` })),
      vectors,
      graph,
    });
    expect(new DataView(buf).getUint32(4, true)).toBe(FORMAT_VERSION);

    const snap = deserialize(buf);
    expect(snap.graph).toBeDefined();
    expect(snap.graph!.M).toBe(graph.M);
    expect(snap.graph!.entry).toBe(graph.entry);
    expect(snap.graph!.top).toBe(graph.top);
    expect(snap.graph!.levels).toEqual(graph.levels);
    expect(snap.graph!.links0).toEqual(graph.links0);
    expect(snap.graph!.upper).toEqual(graph.upper);
  });

  it('rejects a vectors/count mismatch at serialize time', () => {
    const input = makeInput();
    input.vectors = new Float32Array(5);
    expect(() => serialize(input)).toThrow(/vectors length/);
  });

  it('rejects a graph that does not match the snapshot count', () => {
    const input = makeInput(3, 4);
    const g = new HNSWGraph(4, 'l2', { M: 8 });
    g.append(new Float32Array(8), 2); // 2 rows != count 3
    expect(() => serialize({ ...input, graph: g.serializeGraph() })).toThrow(/does not match/);
  });

  it('rejects bad magic, truncation, and unknown versions', () => {
    const good = serialize(makeInput());

    const badMagic = good.slice(0);
    new DataView(badMagic).setUint32(0, 0x12345678, true);
    expect(() => deserialize(badMagic)).toThrow(/bad magic/);

    expect(() => deserialize(good.slice(0, 16))).toThrow(/too small/);
    expect(() => deserialize(good.slice(0, good.byteLength - 8))).toThrow(/truncated/);

    const badVersion = good.slice(0);
    new DataView(badVersion).setUint32(4, 99, true);
    expect(() => deserialize(badVersion)).toThrow(/unsupported snapshot version 99/);

    const badMetric = good.slice(0);
    new DataView(badMetric).setUint32(12, 7, true);
    expect(() => deserialize(badMetric)).toThrow(/unknown metric/);
  });

  it('returned vectors do not pin the snapshot buffer', () => {
    const input = makeInput();
    const buf = serialize(input);
    const snap = deserialize(buf);
    new Uint8Array(buf).fill(0); // trash the source
    expect(snap.vectors).toEqual(input.vectors);
  });
});
