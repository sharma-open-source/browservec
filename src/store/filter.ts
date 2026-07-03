// Metadata predicate evaluation (FR-7).
//
// Compiles a Mongo-ish MetadataFilter into a plain predicate over a record's
// metadata. Fields AND together; a bare value is shorthand for { $eq: value }.
// Validation (unknown operators, malformed arguments) happens at compile time so
// a typo throws once per query instead of silently matching nothing.

import type { FilterOps, FilterValue, Metadata, MetadataFilter } from '../types.js';

type Predicate = (metadata?: Metadata) => boolean;

const OPS = new Set(['$eq', '$ne', '$in', '$gt', '$gte', '$lt', '$lte']);

function isOpsObject(cond: FilterValue | FilterValue[] | FilterOps): cond is FilterOps {
  return cond !== null && typeof cond === 'object' && !Array.isArray(cond);
}

function compileOp(field: string, op: string, arg: unknown): Predicate {
  switch (op) {
    case '$eq':
      return (m) => m?.[field] === arg;
    case '$ne':
      // Mongo-ish: $ne also matches records where the field is missing.
      return (m) => m?.[field] !== arg;
    case '$in': {
      if (!Array.isArray(arg)) {
        throw new Error(`filter: $in on "${field}" expects an array, got ${typeof arg}`);
      }
      const set = new Set(arg as FilterValue[]);
      return (m) => set.has(m?.[field] as FilterValue);
    }
    case '$gt':
    case '$gte':
    case '$lt':
    case '$lte': {
      if (typeof arg !== 'number') {
        throw new Error(`filter: ${op} on "${field}" expects a number, got ${typeof arg}`);
      }
      // Range operators only match stored numbers — a missing field or a
      // string/bool/null value never satisfies a numeric comparison.
      if (op === '$gt') return (m) => typeof m?.[field] === 'number' && (m[field] as number) > arg;
      if (op === '$gte') return (m) => typeof m?.[field] === 'number' && (m[field] as number) >= arg;
      if (op === '$lt') return (m) => typeof m?.[field] === 'number' && (m[field] as number) < arg;
      return (m) => typeof m?.[field] === 'number' && (m[field] as number) <= arg;
    }
    default:
      throw new Error(`filter: unknown operator "${op}" on field "${field}"`);
  }
}

/** Compile a MetadataFilter into a single predicate (AND across fields/operators). */
export function compileFilter(filter: MetadataFilter): Predicate {
  const tests: Predicate[] = [];
  for (const [field, cond] of Object.entries(filter)) {
    if (isOpsObject(cond)) {
      for (const [op, arg] of Object.entries(cond)) {
        if (!OPS.has(op)) throw new Error(`filter: unknown operator "${op}" on field "${field}"`);
        tests.push(compileOp(field, op, arg));
      }
    } else {
      tests.push(compileOp(field, '$eq', cond));
    }
  }
  return (metadata) => {
    for (const t of tests) if (!t(metadata)) return false;
    return true;
  };
}
