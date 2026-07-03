import { describe, it, expect } from 'vitest';
import { compileFilter } from '../../src/store/filter';

describe('compileFilter', () => {
  it('treats a bare value as $eq', () => {
    const p = compileFilter({ lang: 'en' });
    expect(p({ lang: 'en' })).toBe(true);
    expect(p({ lang: 'de' })).toBe(false);
    expect(p({})).toBe(false);
    expect(p(undefined)).toBe(false);
  });

  it('$eq is strict — no type coercion', () => {
    const p = compileFilter({ n: 1 });
    expect(p({ n: 1 })).toBe(true);
    expect(p({ n: '1' })).toBe(false);
    expect(p({ n: true })).toBe(false);
  });

  it('supports null equality', () => {
    const p = compileFilter({ x: null });
    expect(p({ x: null })).toBe(true);
    expect(p({ x: 0 })).toBe(false);
    expect(p({})).toBe(false); // missing field is undefined, not null
  });

  it('$ne also matches records missing the field', () => {
    const p = compileFilter({ lang: { $ne: 'en' } });
    expect(p({ lang: 'de' })).toBe(true);
    expect(p({})).toBe(true);
    expect(p(undefined)).toBe(true);
    expect(p({ lang: 'en' })).toBe(false);
  });

  it('$in matches any listed value', () => {
    const p = compileFilter({ tag: { $in: ['a', 'b'] } });
    expect(p({ tag: 'a' })).toBe(true);
    expect(p({ tag: 'b' })).toBe(true);
    expect(p({ tag: 'c' })).toBe(false);
    expect(p({})).toBe(false);
  });

  it('numeric range operators only match stored numbers', () => {
    const p = compileFilter({ year: { $gte: 2020 } });
    expect(p({ year: 2020 })).toBe(true);
    expect(p({ year: 2019 })).toBe(false);
    expect(p({ year: '2021' })).toBe(false);
    expect(p({ year: null })).toBe(false);
    expect(p({})).toBe(false);
  });

  it('covers all four range operators', () => {
    expect(compileFilter({ n: { $gt: 5 } })({ n: 6 })).toBe(true);
    expect(compileFilter({ n: { $gt: 5 } })({ n: 5 })).toBe(false);
    expect(compileFilter({ n: { $lt: 5 } })({ n: 4 })).toBe(true);
    expect(compileFilter({ n: { $lt: 5 } })({ n: 5 })).toBe(false);
    expect(compileFilter({ n: { $lte: 5 } })({ n: 5 })).toBe(true);
    expect(compileFilter({ n: { $gte: 5 } })({ n: 5 })).toBe(true);
  });

  it('multiple operators on one field AND together', () => {
    const p = compileFilter({ year: { $gte: 2000, $lt: 2010 } });
    expect(p({ year: 2005 })).toBe(true);
    expect(p({ year: 2010 })).toBe(false);
    expect(p({ year: 1999 })).toBe(false);
  });

  it('multiple fields AND together', () => {
    const p = compileFilter({ lang: 'en', year: { $gte: 2020 } });
    expect(p({ lang: 'en', year: 2021 })).toBe(true);
    expect(p({ lang: 'en', year: 2019 })).toBe(false);
    expect(p({ lang: 'de', year: 2021 })).toBe(false);
  });

  it('an empty filter matches everything', () => {
    const p = compileFilter({});
    expect(p({ any: 'thing' })).toBe(true);
    expect(p(undefined)).toBe(true);
  });

  it('throws at compile time on unknown operators', () => {
    expect(() => compileFilter({ x: { $regex: 'a' } as never })).toThrow(/unknown operator "\$regex"/);
  });

  it('throws at compile time on malformed arguments', () => {
    expect(() => compileFilter({ x: { $in: 'not-array' as never } })).toThrow(/expects an array/);
    expect(() => compileFilter({ x: { $gt: 'nope' as never } })).toThrow(/expects a number/);
  });
});
