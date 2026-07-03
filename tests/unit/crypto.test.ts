import { describe, it, expect } from 'vitest';
import { encryptSnapshot, decryptSnapshot, isEncrypted } from '../../src/persist/crypto';

function payload(): ArrayBuffer {
  const bytes = new Uint8Array(256);
  for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 7) & 0xff;
  return bytes.buffer;
}

// PBKDF2 at 210k iterations runs a few times here; each call is ~100ms in Node.
describe('snapshot encryption', () => {
  it('round-trips with the right passphrase', async () => {
    const plain = payload();
    const env = await encryptSnapshot(plain, 'correct horse');
    expect(isEncrypted(env)).toBe(true);
    expect(isEncrypted(plain)).toBe(false);
    const back = await decryptSnapshot(env, 'correct horse');
    expect(new Uint8Array(back)).toEqual(new Uint8Array(plain));
  });

  it('produces a different envelope every time (random salt/iv)', async () => {
    const plain = payload();
    const a = new Uint8Array(await encryptSnapshot(plain, 'p'));
    const b = new Uint8Array(await encryptSnapshot(plain, 'p'));
    const identical = a.length === b.length && a.every((x, i) => x === b[i]);
    expect(identical).toBe(false);
  });

  it('rejects a wrong passphrase loudly', async () => {
    const env = await encryptSnapshot(payload(), 'right');
    await expect(decryptSnapshot(env, 'wrong')).rejects.toThrow(/wrong passphrase or corrupted/);
  });

  it('rejects tampered ciphertext (GCM auth)', async () => {
    const env = await encryptSnapshot(payload(), 'p');
    const tampered = env.slice(0);
    const bytes = new Uint8Array(tampered);
    bytes[bytes.length - 1] ^= 0xff;
    await expect(decryptSnapshot(tampered, 'p')).rejects.toThrow(/wrong passphrase or corrupted/);
  });

  it('rejects an empty passphrase and non-envelope input', async () => {
    await expect(encryptSnapshot(payload(), '')).rejects.toThrow(/non-empty/);
    await expect(decryptSnapshot(payload(), 'p')).rejects.toThrow(/bad magic/);
  });

  it('rejects unsupported envelope versions', async () => {
    const env = await encryptSnapshot(payload(), 'p');
    const bumped = env.slice(0);
    new DataView(bumped).setUint32(4, 2, true);
    await expect(decryptSnapshot(bumped, 'p')).rejects.toThrow(/unsupported encryption version 2/);
  });
});
