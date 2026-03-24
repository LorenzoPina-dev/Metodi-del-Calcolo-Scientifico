// engine/MatrixEngine.ts - Public API helpers

export { globalDispatcher } from "./BackendDispatcher.js";
export { LazyMatrix } from "./LazyMatrix.js";
export { globalPool } from "./BufferPool.js";
export type {
  IBackend, ComputeNode, MatrixHandle,
  DispatcherThresholds, BackendCapabilities
} from "./types.js";

import { globalDispatcher } from "./BackendDispatcher.js";
import { LazyMatrix } from "./LazyMatrix.js";
import type { Matrix } from "../Matrix.js";
import type { INumeric } from "../type/interface.js";
import type { DispatcherThresholds } from "./types.js";

export async function initEngine(options?: {
  gpu?: boolean;
  workers?: boolean;
  wasm?: boolean;
  thresholds?: Partial<DispatcherThresholds>;
}): Promise<{ webgpu: boolean; workers: boolean; wasm: boolean }> {
  if (options?.thresholds) {
    globalDispatcher.configure(options.thresholds);
  }
  return globalDispatcher.init({
    gpu: options?.gpu,
    workers: options?.workers,
    wasm: options?.wasm,
  });
}

export function lazy<T extends INumeric<T>>(m: Matrix<T>): LazyMatrix<T> {
  return LazyMatrix.fromMatrix(m);
}
