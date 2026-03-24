// engine/backends/WorkersBackend.ts - Worker pool backend wrapper

import type { IBackend, BackendCapabilities } from "../types.js";
import { globalPool } from "../BufferPool.js";
import { WorkerPool } from "../../parallel/worker_pool.js";

export class WorkersBackend implements IBackend {
  readonly name = "workers" as const;
  readonly priority = 1;

  async probe(): Promise<BackendCapabilities> {
    try {
      // Usa l'istanza globale se disponibile, altrimenti crea una temporanea per il probe
      if (WorkerPool.instance) {
        return { available: true, perfScore: 0.3 };
      }
      const pool = new WorkerPool();
      await pool.init();
      WorkerPool.instance = pool;
      return { available: true, perfScore: 0.3 };
    } catch {
      return { available: false };
    }
  }

  async matmul(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
  ): Promise<Float64Array> {
    const pool = WorkerPool.instance;
    if (!pool) {
      return this._matmulSerial(A, B, new Float64Array(M * N), M, K, N);
    }
    return pool.matmul(A, B, M, K, N);
  }

  async eltwise(
    op: "add" | "sub" | "dotmul" | "dotdiv",
    A: Float64Array, B: Float64Array | number,
    len: number
  ): Promise<Float64Array> {
    const C = globalPool.acquire(len);
    const isScalar = typeof B === "number";
    switch (op) {
      case "add":
        for (let i = 0; i < len; i++) C[i] = A[i] + (isScalar ? (B as number) : (B as Float64Array)[i]);
        break;
      case "sub":
        for (let i = 0; i < len; i++) C[i] = A[i] - (isScalar ? (B as number) : (B as Float64Array)[i]);
        break;
      case "dotmul":
        for (let i = 0; i < len; i++) C[i] = A[i] * (isScalar ? (B as number) : (B as Float64Array)[i]);
        break;
      case "dotdiv":
        for (let i = 0; i < len; i++) C[i] = A[i] / (isScalar ? (B as number) : (B as Float64Array)[i]);
        break;
    }
    return C;
  }

  async unary(
    op: "negate" | "abs" | "sqrt" | "exp" | "transpose",
    A: Float64Array,
    rows: number, cols: number
  ): Promise<Float64Array> {
    const len = rows * cols;
    const result = globalPool.acquire(len);
    switch (op) {
      case "negate":
        for (let i = 0; i < len; i++) result[i] = -A[i];
        break;
      case "abs":
        for (let i = 0; i < len; i++) result[i] = Math.abs(A[i]);
        break;
      case "sqrt":
        for (let i = 0; i < len; i++) result[i] = Math.sqrt(A[i]);
        break;
      case "exp":
        for (let i = 0; i < len; i++) result[i] = Math.exp(A[i]);
        break;
      case "transpose":
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            result[c * rows + r] = A[r * cols + c];
          }
        }
        break;
    }
    return result;
  }

  dispose(): void {
    if (WorkerPool.instance) {
      WorkerPool.instance.shutdown();
    }
  }

  private _matmulSerial(
    A: Float64Array, B: Float64Array, C: Float64Array,
    M: number, K: number, N: number
  ): Float64Array {
    C.fill(0);
    for (let i = 0; i < M; i++) {
      const iK = i * K;
      const iN = i * N;
      for (let k = 0; k < K; k++) {
        const aik = A[iK + k];
        if (aik === 0) continue;
        const kN = k * N;
        for (let j = 0; j < N; j++) C[iN + j] += aik * B[kN + j];
      }
    }
    return C;
  }
}
