// engine/backends/WebGPUBackend.ts - WebGPU backend wrapper

import type { IBackend, BackendCapabilities } from "../types.js";
import { globalPool } from "../BufferPool.js";
import {
  initGPU,
  isGPUAvailable,
  gpuMatmul,
  gpuElementwise,
  gpuScalarOp,
  gpuUnary
} from "../../gpu/webgpu_backend.js";

export class WebGPUBackend implements IBackend {
  readonly name = "webgpu" as const;
  readonly priority = 0;

  async probe(): Promise<BackendCapabilities> {
    try {
      const ok = await initGPU();
      return { available: ok && isGPUAvailable(), perfScore: 0.01 };
    } catch {
      return { available: false };
    }
  }

  async matmul(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
  ): Promise<Float64Array> {
    return gpuMatmul(A, B, M, K, N);
  }

  async eltwise(
    op: "add" | "sub" | "dotmul" | "dotdiv",
    A: Float64Array, B: Float64Array | number,
    len: number
  ): Promise<Float64Array> {
    if (typeof B === "number") {
      const opIdx = this._scalarOpIdx(op);
      return gpuScalarOp(A.subarray(0, len), B, opIdx);
    }
    const opIdx = this._eltwiseOpIdx(op);
    return gpuElementwise(A.subarray(0, len), B.subarray(0, len), opIdx);
  }

  async unary(
    op: "negate" | "abs" | "sqrt" | "exp" | "transpose",
    A: Float64Array,
    rows: number, cols: number
  ): Promise<Float64Array> {
    if (op === "transpose") {
      const len = rows * cols;
      const result = globalPool.acquire(len);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          result[c * rows + r] = A[r * cols + c];
        }
      }
      return result;
    }
    const opIdx = this._unaryOpIdx(op);
    return gpuUnary(A.subarray(0, rows * cols), opIdx);
  }

  dispose(): void {
    // no explicit dispose for the shared GPU device
  }

  private _eltwiseOpIdx(op: "add" | "sub" | "dotmul" | "dotdiv"): 0 | 1 | 2 | 3 {
    switch (op) {
      case "add": return 0;
      case "sub": return 1;
      case "dotmul": return 2;
      case "dotdiv": return 3;
    }
  }

  private _scalarOpIdx(op: "add" | "sub" | "dotmul" | "dotdiv"): 0 | 1 | 2 | 3 {
    switch (op) {
      case "add": return 0;
      case "sub": return 1;
      case "dotmul": return 2;
      case "dotdiv": return 3;
    }
  }

  private _unaryOpIdx(op: "negate" | "abs" | "sqrt" | "exp"): number {
    switch (op) {
      case "abs": return 0;
      case "negate": return 1;
      case "sqrt": return 2;
      case "exp": return 3;
    }
  }
}
