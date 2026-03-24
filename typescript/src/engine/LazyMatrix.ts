// engine/LazyMatrix.ts - Lazy evaluation wrapper

import type { ComputeNode, MatrixHandle, OpCode } from "./types.js";
import { globalDispatcher } from "./BackendDispatcher.js";
import { globalPool } from "./BufferPool.js";
import type { INumeric } from "../type/interface.js";
import type { Matrix } from "../Matrix.js";
import type { Float64M } from "../type/float.js";

type ReadyState = {
  kind: "READY";
  rawF64: Float64Array;
  rows: number;
  cols: number;
};

type PendingState = {
  kind: "PENDING";
  node: ComputeNode;
  rows: number;
  cols: number;
};

type LMState = ReadyState | PendingState;

export class LazyMatrix<T extends INumeric<T> = Float64M> implements MatrixHandle {
  private _state: LMState;

  static fromMatrix<T extends INumeric<T>>(m: Matrix<T>): LazyMatrix<T> {
    if (!(m as any).isFloat64) {
      throw new Error("LazyMatrix supports only Float64M matrices.");
    }
    const len = m.rows * m.cols;
    const out = globalPool.acquire(len);
    const d = m.data as Array<{ value?: number; toNumber?: () => number }>;
    for (let i = 0; i < len; i++) {
      const v = d[i] as any;
      out[i] = typeof v.value === "number" ? v.value : v.toNumber();
    }
    const lm = new LazyMatrix<T>();
    lm._state = { kind: "READY", rawF64: out, rows: m.rows, cols: m.cols };
    return lm;
  }

  static fromRaw<T extends INumeric<T>>(
    data: Float64Array, rows: number, cols: number
  ): LazyMatrix<T> {
    const lm = new LazyMatrix<T>();
    lm._state = { kind: "READY", rawF64: data, rows, cols };
    return lm;
  }

  private static pending<T extends INumeric<T>>(
    node: ComputeNode, rows: number, cols: number
  ): LazyMatrix<T> {
    const lm = new LazyMatrix<T>();
    lm._state = { kind: "PENDING", node, rows, cols };
    return lm;
  }

  private constructor() {
    this._state = undefined as unknown as LMState;
  }

  get rows(): number { return this._state.rows; }
  get cols(): number { return this._state.cols; }
  get state(): "READY" | "PENDING" { return this._state.kind; }
  get rawData(): Float64Array | undefined {
    return this._state.kind === "READY" ? this._state.rawF64 : undefined;
  }
  get node(): ComputeNode | undefined {
    return this._state.kind === "PENDING" ? this._state.node : undefined;
  }

  lazyMul(B: LazyMatrix<T>): LazyMatrix<T> {
    if (this.cols !== B.rows) {
      throw new Error(`lazyMul: dimension mismatch (${this.cols} != ${B.rows})`);
    }
    return LazyMatrix.pending<T>(
      { op: "matmul", inputs: [this, B] },
      this.rows, B.cols
    );
  }

  lazyAdd(B: LazyMatrix<T> | number): LazyMatrix<T> {
    if (typeof B !== "number" && (this.rows !== B.rows || this.cols !== B.cols)) {
      throw new Error("lazyAdd: dimension mismatch");
    }
    const inputs: MatrixHandle[] = [this];
    if (typeof B !== "number") inputs.push(B);
    return LazyMatrix.pending<T>(
      { op: "add", inputs, meta: typeof B === "number" ? { scalar: B } : undefined },
      this.rows, this.cols
    );
  }

  lazySub(B: LazyMatrix<T> | number): LazyMatrix<T> {
    if (typeof B !== "number" && (this.rows !== B.rows || this.cols !== B.cols)) {
      throw new Error("lazySub: dimension mismatch");
    }
    const inputs: MatrixHandle[] = [this];
    if (typeof B !== "number") inputs.push(B);
    return LazyMatrix.pending<T>(
      { op: "sub", inputs, meta: typeof B === "number" ? { scalar: B } : undefined },
      this.rows, this.cols
    );
  }

  lazyDotMul(B: LazyMatrix<T> | number): LazyMatrix<T> {
    if (typeof B !== "number" && (this.rows !== B.rows || this.cols !== B.cols)) {
      throw new Error("lazyDotMul: dimension mismatch");
    }
    const inputs: MatrixHandle[] = [this];
    if (typeof B !== "number") inputs.push(B);
    return LazyMatrix.pending<T>(
      { op: "dotmul", inputs, meta: typeof B === "number" ? { scalar: B } : undefined },
      this.rows, this.cols
    );
  }

  lazyDotDiv(B: LazyMatrix<T> | number): LazyMatrix<T> {
    if (typeof B !== "number" && (this.rows !== B.rows || this.cols !== B.cols)) {
      throw new Error("lazyDotDiv: dimension mismatch");
    }
    const inputs: MatrixHandle[] = [this];
    if (typeof B !== "number") inputs.push(B);
    return LazyMatrix.pending<T>(
      { op: "dotdiv", inputs, meta: typeof B === "number" ? { scalar: B } : undefined },
      this.rows, this.cols
    );
  }

  lazyT(): LazyMatrix<T> {
    return LazyMatrix.pending<T>(
      { op: "transpose", inputs: [this] },
      this.cols, this.rows
    );
  }

  lazyNegate(): LazyMatrix<T> {
    return LazyMatrix.pending<T>(
      { op: "negate", inputs: [this] },
      this.rows, this.cols
    );
  }

  lazyAbs(): LazyMatrix<T> {
    return LazyMatrix.pending<T>(
      { op: "abs", inputs: [this] },
      this.rows, this.cols
    );
  }

  lazySqrt(): LazyMatrix<T> {
    return LazyMatrix.pending<T>(
      { op: "sqrt", inputs: [this] },
      this.rows, this.cols
    );
  }

  lazyExp(): LazyMatrix<T> {
    return LazyMatrix.pending<T>(
      { op: "exp", inputs: [this] },
      this.rows, this.cols
    );
  }

  // ── Operazioni di algebra lineare lazy ─────────────────────────────────────

  /** Inversa: richiede matrice quadrata */
  lazyInv(): LazyMatrix<T> {
    if (this.rows !== this.cols)
      throw new Error(`lazyInv: matrice non quadrata (${this.rows}x${this.cols})`);
    return LazyMatrix.pending<T>(
      { op: "inv", inputs: [this] },
      this.rows, this.cols
    );
  }

  /**
   * Solver Jacobi lazy: risolve Ax = b in modo asincrono.
   * `b` deve essere una colonna (n x 1).
   */
  lazySolveJacobi(
    b: LazyMatrix<T>,
    tol = 1e-10, maxIter = 5000
  ): LazyMatrix<T> {
    if (this.rows !== this.cols)
      throw new Error(`lazySolveJacobi: matrice non quadrata (${this.rows}x${this.cols})`);
    if (b.rows !== this.rows || b.cols !== 1)
      throw new Error(`lazySolveJacobi: b deve essere ${this.rows}x1`);
    return LazyMatrix.pending<T>(
      { op: "solve_jacobi", inputs: [this, b], meta: { tol, maxIter } },
      this.rows, 1
    );
  }

  async data(): Promise<Matrix<Float64M>> {
    const f64 = await this._evaluate();
    const mod = await import("../Matrix.js");
    return new mod.Matrix<Float64M>(this.rows, this.cols, f64);
  }

  async rawF64(): Promise<Float64Array> {
    return this._evaluate();
  }

  private _evalCache: WeakMap<LazyMatrix<T>, Float64Array> | null = null;

  private async _evaluate(): Promise<Float64Array> {
    if (!this._evalCache) this._evalCache = new WeakMap();
    return this._evalNode(this, this._evalCache);
  }

  private async _evalNode(
    node: LazyMatrix<T>,
    cache: WeakMap<LazyMatrix<T>, Float64Array>
  ): Promise<Float64Array> {
    if (cache.has(node)) return cache.get(node)!;

    if (node._state.kind === "READY") {
      const raw = node._state.rawF64;
      cache.set(node, raw);
      return raw;
    }

    const state = node._state as PendingState;
    const cNode = state.node;
    const inputs = await Promise.all(
      cNode.inputs.map(h => this._evalNode(h as LazyMatrix<T>, cache))
    );

    const result = await this._execOp(cNode.op, inputs, cNode.meta, node.rows, node.cols);
    cache.set(node, result);
    return result;
  }

  private async _execOp(
    op: OpCode,
    inputs: Float64Array[],
    meta: Record<string, number | string> | undefined,
    outRows: number, outCols: number
  ): Promise<Float64Array> {
    const A = inputs[0];
    const B = inputs[1];

    switch (op) {
      case "matmul": {
        const K = A.length / outRows;
        return globalDispatcher.matmul(A, B, outRows, K, outCols);
      }
      case "add": {
        const bVal = meta?.scalar !== undefined ? (meta.scalar as number) : B;
        return globalDispatcher.eltwise("add", A, bVal, outRows * outCols);
      }
      case "sub": {
        const bVal = meta?.scalar !== undefined ? (meta.scalar as number) : B;
        return globalDispatcher.eltwise("sub", A, bVal, outRows * outCols);
      }
      case "dotmul": {
        const bVal = meta?.scalar !== undefined ? (meta.scalar as number) : B;
        return globalDispatcher.eltwise("dotmul", A, bVal, outRows * outCols);
      }
      case "dotdiv": {
        const bVal = meta?.scalar !== undefined ? (meta.scalar as number) : B;
        return globalDispatcher.eltwise("dotdiv", A, bVal, outRows * outCols);
      }
      case "transpose":
        return globalDispatcher.unary("transpose", A, outCols, outRows);
      case "negate":
        return globalDispatcher.unary("negate", A, outRows, outCols);
      case "abs":
        return globalDispatcher.unary("abs", A, outRows, outCols);
      case "sqrt":
        return globalDispatcher.unary("sqrt", A, outRows, outCols);
      case "exp":
        return globalDispatcher.unary("exp", A, outRows, outCols);

      // ── Algebra lineare: implementazioni async tramite import dinamico ───────
      case "inv": {
        const { smartInverse } = await import("../algoritm/inverse.js");
        const { Matrix } = await import("../Matrix.js");
        const m = new Matrix<any>(outRows, outCols, A);
        const inv = smartInverse(m);
        return _matrixToF64(inv);
      }
      case "solve_jacobi": {
        const { solveJacobiAsync } = await import("../solver/jacobi.js");
        const { Matrix } = await import("../Matrix.js");
        const tol = typeof meta?.tol === "number" ? meta.tol : 1e-10;
        const maxIter = typeof meta?.maxIter === "number" ? meta.maxIter : 5000;
        const Amat = new Matrix<any>(outRows, outRows, A);
        const bmat = new Matrix<any>(outRows, 1, B);
        const x = await solveJacobiAsync(Amat, bmat, tol, maxIter);
        return _matrixToF64(x);
      }
      default:
        throw new Error(`[LazyMatrix] Unknown op: ${op}`);
    }
  }
}

// ─── Helper: Matrix<T> → Float64Array ────────────────────────────────────────
function _matrixToF64(m: { data: Array<{ value?: number; toNumber?: () => number }> }): Float64Array {
  const len = m.data.length;
  const out = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    const v = m.data[i] as any;
    out[i] = typeof v.value === "number" ? v.value : v.toNumber();
  }
  return out;
}
