# BrowserVec for frontend developers

You already know **embeddings** — you've used an API like OpenAI's, got back a
list of numbers, and sent it to a vector database somewhere on the internet.

BrowserVec is that database, but it runs inside your user's browser. No server,
no API key, no round-trip. This guide explains what's happening in plain
language — no approximate-nearest-neighbor papers required.

---

## What's a vector?

A **vector** is just a list of numbers. Think of it as coordinates on a map:

```
[0.12, -0.54, 0.87, 0.33, ...]   ← 768 numbers
```

If you've used embedding APIs, the numbers represent **meaning** — similar
concepts produce similar lists of numbers. An embedding of "dog" is closer to
"puppy" than to "toaster."

BrowserVec stores these lists and, given a new one, finds the closest matches
from everything you've saved.

---

## What does "similarity search" mean?

You give it a query vector (a list of numbers). It finds the stored vectors
that are **nearest** to it.

"Nearest" depends on a **metric** you choose:

| Metric | What it measures | When to use |
|---|---|---|
| `cosine` | Direction — are they pointing the same way? | Most embeddings |
| `dot` | Direction × length — same as cosine when vectors are normalized | Legacy models |
| `l2` | Straight-line distance on the map | Some custom models |

The result is a list of `{ id, score }` pairs. Higher score = closer match.

---

## Core concepts (no jargon)

### Index

An **index** is how the library organizes your vectors so that lookups are
fast. There are several strategies:

- **Flat** — checks every single vector, one by one. Simple, exact, works
  great up to ~hundreds of thousands of rows.
- **Clustered** — groups similar vectors together into "neighborhoods." A
  query only checks the closest few neighborhoods instead of the whole city.
  Much faster when you have millions of vectors.
- **Graph** — connects each vector to its nearest neighbors, like a
  navigation map. A query hops from neighbor to neighbor, reaching the
  closest ones in a few steps. Works well even without a GPU.

You don't have to pick at first — `flat` is the default. When you need more
speed, you enable one of the faster strategies.

### Quantization

**Quantization** is compression for vectors. Instead of storing each number as
a full 32-bit float (like a high-res photo), you can store it as:

- **int8** — 8 bits per number (~4× smaller)
- **int4** — 4 bits per number (~8× smaller)
- **1-bit** — just the sign, ±1 (~32× smaller)

Smaller means more vectors fit in the same memory. The trade-off: compression
adds a little noise, so the library does a quick "second pass" on the best
candidates to restore accuracy.

### GPU acceleration

Modern browsers have access to your graphics card through WebGPU. The library
uses it to run the math (millions of dot-products per query) in parallel on
the GPU — the same chip that renders games. This makes search fast even with
a million vectors.

If WebGPU isn't available (older phones, some browsers), the library falls
back to optimized CPU code.

### Persistence

**Persistence** means saving your vectors to the device's storage so they're
still there when the user comes back. The library uses:

- **OPFS** (Origin Private File System) — fast, modern browser storage
- **IndexedDB** — older but universally supported fallback

It's all local. Nothing is sent to a server.

### Encryption

If you're saving sensitive data, you can **encrypt** the snapshot with a
passphrase. The library uses the same encryption standards banks use
(AES-256-GCM). Without the passphrase, the stored data is unreadable.

---

## How a query works (the short version)

```
You call:   db.query(myVector, { k: 5 })

1. Validate — is the vector the right size?
2. Normalize — if using cosine metric, make it unit length
3. Score — compare myVector against all stored vectors (or the closest groups)
             └── runs on the GPU if available
4. Pick top-k — keep the 5 highest scores
5. Re-rank — if using quantization, re-score those 5 with exact math
6. Return — { id, score } for each match
```

The whole thing takes a few milliseconds.

---

## The lifecycle of a store

```
Create → Add vectors → Query → Save → (close tab)
                                      ↓
                               Create → (auto-loads) → Query
```

1. **Create** — pick a dimension and metric (`BrowserVec.create(...)`)
2. **Add** — insert vectors (`db.addBatch(...)`)
3. **Query** — search (`db.query(...)`)
4. **Save** — persist to disk (`db.save()`)
5. **Next visit** — same name auto-loads, no re-insertion needed

---

## What all those acronyms mean

You'll see these in the advanced docs. Here's a quick decoder:

| Term | Stands for | What it is |
|---|---|---|
| **ANN** | Approximate Nearest Neighbor | A search that returns results that are very close to the best ones, but faster than checking everything |
| **IVF** | Inverted File Index | The "clustered" strategy — groups vectors into neighborhoods |
| **HNSW** | Hierarchical Navigable Small World | The "graph" strategy — connects vectors in a navigation map |
| **WGSL** | WebGPU Shading Language | The programming language used to write GPU code for the browser |
| **OPFS** | Origin Private File System | Fast browser storage for saving data locally |
| **PBKDF2** | Password-Based Key Derivation Function 2 | The algorithm that turns your passphrase into an encryption key |
| **AES-256-GCM** | Advanced Encryption Standard (256-bit, Galois/Counter Mode) | The encryption used to protect saved data |
| **SIMD** | Single Instruction, Multiple Data | A CPU feature that processes multiple numbers at once — makes the CPU fallback faster |
| **FWHT** | Fast Walsh-Hadamard Transform | The math that "rotates" vectors before compression (makes quantization work better) |

---

## Moving to the advanced docs

Once you're comfortable, here's where to go next:

- [Integration guide](./integration.md) — bundle it, CSP headers, error handling
- [Configuration guide](./configuration.md) — tuning, decision trees, parameter reference
- [Feature deep-dives](../features.md) — code examples for every capability
- [Architecture](../architecture.md) — how the pieces fit together
- [Internals](../internals.md) — how every kernel, codec, and algorithm works
