// engine/types.ts - Compute graph and backend interfaces

export type OpCode =
  | "matmul"
  | "add"
  | "sub"
  | "dotmul"
  | "dotdiv"
  | "transpose"
  | "negate"
  | "abs"
  | "sqrt"
  | "exp"
  // Algebra lineare — implementate async nel dispatcher
  | "inv"
  | "solve_jacobi"
  | "lu"
  | "lup"
  | "cholesky"
  | "qr";

export interface ComputeNode {
  op: OpCode;
  inputs: MatrixHandle[];
  meta?: Record<string, number | string>;
}

export interface MatrixHandle {
  readonly rows: number;
  readonly cols: number;
  readonly state: "READY" | "PENDING";
  readonly rawData?: Float64Array;
  readonly node?: ComputeNode;
}

export interface BackendCapabilities {
  available: boolean;
  perfScore?: number;
}

export interface IBackend {
  readonly name: "webgpu" | "workers" | "wasm" | "vanilla";
  readonly priority: number;

  probe(): Promise<BackendCapabilities>;

  matmul(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
  ): Promise<Float64Array>;

  eltwise(
    op: "add" | "sub" | "dotmul" | "dotdiv",
    A: Float64Array, B: Float64Array | number,
    len: number
  ): Promise<Float64Array>;

  unary(
    op: "negate" | "abs" | "sqrt" | "exp" | "transpose",
    A: Float64Array,
    rows: number, cols: number
  ): Promise<Float64Array>;

  dispose(): void;
}

export interface DispatcherThresholds {
  gpuMinElements: number;
  workersMinElements: number;
  wasmMinElements: number;
}

export const DEFAULT_THRESHOLDS: DispatcherThresholds = {
  gpuMinElements: 1_000_000,
  workersMinElements: 100_000,
  wasmMinElements: 5_000,
};
