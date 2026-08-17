import "@testing-library/jest-dom/vitest";
import { configure } from "@testing-library/react";

// Slow shared CI runners can take far more than the 1s default for async UI
// state (conflict dialogs, project option lists) to settle after hydration;
// 5s still proved flaky under push+PR double runs. Keep well below the
// Vitest per-test timeout so the RTL error wins over a hard kill.
configure({ asyncUtilTimeout: 15000 });

// Node 25 exposes an incomplete experimental localStorage object when it is
// started without --localstorage-file. Install the browser Storage contract
// explicitly so hydration tests exercise the same API used in production.
const values = new Map<string, string>();
const storage: Storage = {
  get length() {
    return values.size;
  },
  clear() {
    values.clear();
  },
  getItem(key) {
    return values.get(String(key)) ?? null;
  },
  key(index) {
    return Array.from(values.keys())[index] ?? null;
  },
  removeItem(key) {
    values.delete(String(key));
  },
  setItem(key, value) {
    values.set(String(key), String(value));
  },
};

Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: storage,
});
