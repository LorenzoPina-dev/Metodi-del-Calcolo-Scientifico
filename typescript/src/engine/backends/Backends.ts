// engine/backends/Backends.ts - WASM and vanilla CPU backends

import type { IBackend, BackendCapabilities } from "../types.js";
import { globalPool } from "../BufferPool.js";
import { initWasm, getBridgeSync } from "../../wasm/wasm_bridge.js";

type WasmOp =
  | "addMatrix"
  | "subMatrix"
  | "dotMul"
  | "dotDiv"
  | "addScalar"
  | "subScalar"
  | "mulScalar"
  | "unaryNeg"
  | "unaryAbs"
  | "unarySqrt"
  | "unaryExp"
  | "transpose"
  | "matmul";

export class WasmBackend implements IBackend {
  readonly name = "wasm" as const;
  readonly priority = 2;

  private bridge: ReturnType<typeof getBridgeSync> | null = null;
  private queue: Promise<unknown> = Promise.resolve();

  async probe(): Promise<BackendCapabilities> {
    try {
      await initWasm();
      this.bridge = getBridgeSync();
      if (!this.bridge) return { available: false };
      return { available: true, perfScore: 0.15 };
    } catch {
      return { available: false };
    }
  }

  async matmul(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
  ): Promise<Float64Array> {
    return this._enqueue(() => {
      const w = this._requireBridge();
      const aPtr = this._write(w, A);
      const bPtr = this._write(w, B);
      const cPtr = w.allocOutput(M * N);

      w.exports.matmul(aPtr, bPtr, cPtr, M, K, N);

      const result = globalPool.acquire(M * N);
      const heap = new Float64Array(w.exports.memory.buffer, cPtr, M * N);
      result.set(heap);
      w.reset();
      return result;
    });
  }

  async eltwise(
    op: "add" | "sub" | "dotmul" | "dotdiv",
    A: Float64Array, B: Float64Array | number,
    len: number
  ): Promise<Float64Array> {
    return this._enqueue(() => {
      const w = this._requireBridge();
      const aPtr = this._write(w, A.subarray(0, len));

      let cPtr = 0;
      if (typeof B === "number") {
        const scalar = op === "dotdiv" ? (1 / B) : B;
        cPtr = w.allocOutput(len);
        const fn = this._scalarFn(op);
        (w.exports as any)[fn](aPtr, cPtr, len, scalar);
      } else {
        const bPtr = this._write(w, B.subarray(0, len));
        cPtr = w.allocOutput(len);
        const fn = this._binFn(op);
        (w.exports as any)[fn](aPtr, bPtr, cPtr, len);
      }

      const result = globalPool.acquire(len);
      const heap = new Float64Array(w.exports.memory.buffer, cPtr, len);
      result.set(heap);
      w.reset();
      return result;
    });
  }

  async unary(
    op: "negate" | "abs" | "sqrt" | "exp" | "transpose",
    A: Float64Array,
    rows: number, cols: number
  ): Promise<Float64Array> {
    return this._enqueue(() => {
      const w = this._requireBridge();
      const len = rows * cols;
      const aPtr = this._write(w, A.subarray(0, len));
      const cPtr = w.allocOutput(len);

      switch (op) {
        case "negate":
          w.exports.unaryNeg(aPtr, cPtr, len);
          break;
        case "abs":
          w.exports.unaryAbs(aPtr, cPtr, len);
          break;
        case "sqrt":
          w.exports.unarySqrt(aPtr, cPtr, len);
          break;
        case "exp":
          w.exports.unaryExp(aPtr, cPtr, len);
          break;
        case "transpose":
          w.exports.transpose(aPtr, cPtr, rows, cols);
          break;
      }

      const result = globalPool.acquire(len);
      const heap = new Float64Array(w.exports.memory.buffer, cPtr, len);
      result.set(heap);
      w.reset();
      return result;
    });
  }

  dispose(): void {
    this.bridge = null;
  }

  private _enqueue<T>(fn: () => T | Promise<T>): Promise<T> {
    const run = () => Promise.resolve().then(fn);
    const p = this.queue.then(run, run);
    this.queue = p.catch(() => undefined);
    return p;
  }

  private _requireBridge() {
    if (!this.bridge) throw new Error("WASM not initialized");
    return this.bridge;
  }

  private _write(w: NonNullable<ReturnType<typeof getBridgeSync>>, data: Float64Array): number {
    const ptr = w.alloc(data.length);
    const heap = new Float64Array(w.exports.memory.buffer);
    heap.set(data, ptr >> 3);
    return ptr;
  }

  private _binFn(op: "add" | "sub" | "dotmul" | "dotdiv"): WasmOp {
    switch (op) {
      case "add": return "addMatrix";
      case "sub": return "subMatrix";
      case "dotmul": return "dotMul";
      case "dotdiv": return "dotDiv";
    }
  }

  private _scalarFn(op: "add" | "sub" | "dotmul" | "dotdiv"): WasmOp {
    switch (op) {
      case "add": return "addScalar";
      case "sub": return "subScalar";
      case "dotmul":
      case "dotdiv":
        return "mulScalar";
    }
  }
}

export class VanillaBackend implements IBackend {
  readonly name = "vanilla" as const;
  readonly priority = 3;

  async probe(): Promise<BackendCapabilities> {
    return { available: true, perfScore: 1.0 };
  }

  async matmul(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
  ): Promise<Float64Array> {
    const C = globalPool.acquire(M * N);
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
    // noop
  }
}
