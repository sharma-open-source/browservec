// Snapshot encryption at rest (private offline storage).
//
// An in-browser vector store keeps embeddings on the device precisely so data
// never leaves it — but a persisted OPFS/IndexedDB snapshot (or an exported blob)
// is otherwise plaintext vectors + metadata, which leak content if the device is
// shared or the storage is exfiltrated. This wraps a serialized snapshot in an
// authenticated-encryption envelope: a passphrase is stretched with PBKDF2-SHA256
// into an AES-256-GCM key, and GCM's auth tag means a wrong passphrase or any
// tampering fails loudly rather than returning garbage.
//
// Envelope layout (little-endian):
//   [0..4)    magic "BVCE" (BrowserVec Crypto Envelope)
//   [4..8)    u32 version (= 1)
//   [8..12)   u32 PBKDF2 iterations
//   [12..16)  u32 reserved (0)
//   [16..32)  salt (16 bytes)   — PBKDF2
//   [32..44)  iv   (12 bytes)   — AES-GCM nonce
//   [44..)    ciphertext (WebCrypto appends the 16-byte GCM tag)

const CRYPTO_MAGIC = 0x45435642; // "BVCE" little-endian
const CRYPTO_VERSION = 1;
const HEADER_BYTES = 44;
const SALT_BYTES = 16;
const IV_BYTES = 12;
const PBKDF2_ITERATIONS = 210_000; // OWASP-ish floor for PBKDF2-SHA256

function subtle(): SubtleCrypto {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (!c?.subtle) {
    throw new Error('encryption requires WebCrypto (crypto.subtle), unavailable in this environment');
  }
  return c.subtle;
}

/** True if `buf` starts with the encrypted-envelope magic. */
export function isEncrypted(buf: ArrayBuffer): boolean {
  if (buf.byteLength < 4) return false;
  return new DataView(buf).getUint32(0, true) === CRYPTO_MAGIC;
}

async function deriveKey(passphrase: string, salt: Uint8Array, iterations: number): Promise<CryptoKey> {
  const s = subtle();
  const material = await s.importKey('raw', new TextEncoder().encode(passphrase), 'PBKDF2', false, [
    'deriveKey',
  ]);
  return s.deriveKey(
    { name: 'PBKDF2', salt, iterations, hash: 'SHA-256' },
    material,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  );
}

/** Encrypt a serialized snapshot into a self-describing envelope. */
export async function encryptSnapshot(plain: ArrayBuffer, passphrase: string): Promise<ArrayBuffer> {
  if (!passphrase) throw new Error('encryption passphrase must be a non-empty string');
  const c = (globalThis as { crypto?: Crypto }).crypto!;
  const salt = c.getRandomValues(new Uint8Array(SALT_BYTES));
  const iv = c.getRandomValues(new Uint8Array(IV_BYTES));
  const key = await deriveKey(passphrase, salt, PBKDF2_ITERATIONS);
  const ct = await subtle().encrypt({ name: 'AES-GCM', iv }, key, plain);

  const out = new ArrayBuffer(HEADER_BYTES + ct.byteLength);
  const dv = new DataView(out);
  dv.setUint32(0, CRYPTO_MAGIC, true);
  dv.setUint32(4, CRYPTO_VERSION, true);
  dv.setUint32(8, PBKDF2_ITERATIONS, true);
  dv.setUint32(12, 0, true);
  const bytes = new Uint8Array(out);
  bytes.set(salt, 16);
  bytes.set(iv, 32);
  bytes.set(new Uint8Array(ct), HEADER_BYTES);
  return out;
}

/** Decrypt an envelope produced by {@link encryptSnapshot}. Wrong passphrase/tamper throws. */
export async function decryptSnapshot(buf: ArrayBuffer, passphrase: string): Promise<ArrayBuffer> {
  if (!isEncrypted(buf)) throw new Error('not an encrypted BrowserVec snapshot (bad magic)');
  if (buf.byteLength < HEADER_BYTES) throw new Error('encrypted snapshot too small / corrupt');
  const dv = new DataView(buf);
  const version = dv.getUint32(4, true);
  if (version !== CRYPTO_VERSION) {
    throw new Error(`unsupported encryption version ${version} (this build reads ${CRYPTO_VERSION})`);
  }
  const iterations = dv.getUint32(8, true);
  const salt = new Uint8Array(buf, 16, SALT_BYTES);
  const iv = new Uint8Array(buf, 32, IV_BYTES);
  const ct = buf.slice(HEADER_BYTES);
  const key = await deriveKey(passphrase, salt, iterations);
  try {
    return await subtle().decrypt({ name: 'AES-GCM', iv }, key, ct);
  } catch {
    // GCM auth failure — wrong passphrase or corrupted/tampered data.
    throw new Error('decryption failed: wrong passphrase or corrupted snapshot');
  }
}
