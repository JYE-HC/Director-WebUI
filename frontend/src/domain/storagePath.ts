/**
 * Validates an opaque storage identity string returned by the backend (the
 * active database path). The frontend never resolves these values — they
 * feed localStorage keys and digests — so the check only guarantees a
 * non-empty, control-character-free, platform-absolute path: POSIX absolute
 * (`/srv/...`), Windows drive-letter (`D:\...` / `D:/...`), or UNC
 * (`\\server\share\...`).
 */
export function isStoragePath(value: unknown): value is string {
  return typeof value === "string" &&
    value.length > 1 &&
    value.length <= 4096 &&
    !/[\u0000-\u001f\u007f]/.test(value) &&
    (value.startsWith("/") ||
      /^[A-Za-z]:[\\/]/.test(value) ||
      value.startsWith("\\\\"));
}
