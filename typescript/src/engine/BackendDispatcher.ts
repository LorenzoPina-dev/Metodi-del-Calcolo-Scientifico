// engine/BackendDispatcher.ts - Cascading backend router

import { IBackend, DispatcherThresholds, DEFAULT_THRESHOLDS } from "./types.js";
import { WebGPUBackend } from "./backends/WebGPUBackend.js";
import { WorkersBackend } from "./backends/WorkersBackend.js";
import { WasmBackend, VanillaBackend } from "./backends/Backends.js";

export class BackendDispatcher {
  private backends: IBackend[] = [];
  private available: Set<string> = new Set();
  private initialized = false;
  private thresholds: DispatcherThresholds;
  private initPromise: Promise<{ webgpu: boolean; workers: boolean; wasm: boolean }> | null = null;

  constructor(thresholds: Partial<DispatcherThresholds> = {}) {
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...thresholds };
  }

  async init(options?: { gpu?: boolean; workers?: boolean; wasm?: boolean }): Promise<{
    webgpu: boolean; workers: boolean; wasm: boolean;
  }> {
    if (this.initialized && this.initPromise) return this.initPromise;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this._doInit(options);
    return this.initPromise;
  }

  private async _doInit(options?: { gpu?: boolean; workers?: boolean; wasm?: boolean }) {
    const enabled = {
      gpu: options?.gpu ?? true,
      workers: options?.workers ?? true,
      wasm: options?.wasm ?? true,
    };

    const candidates: IBackend[] = [];
    if (enabled.gpu) candidates.push(new WebGPUBackend());
    if (enabled.workers) candidates.push(new WorkersBackend());
    if (enabled.wasm) candidates.push(new WasmBackend());
    candidates.push(new VanillaBackend());

    const results = await Promise.all(
      candidates.map(b => b.probe().then(caps => ({ b, caps })))
    );

    this.backends = [];
    this.available = new Set();

    for (const { b, caps } of results) {
      if (caps.available) {
        this.backends.push(b);
        this.available.add(b.name);
      } else {
        b.dispose();
      }
    }

    this.backends.sort((a, b) => a.priority - b.priority);
    this.initialized = true;

    return {
      webgpu: this.available.has("webgpu"),
      workers: this.available.has("workers"),
      wasm: this.available.has("wasm"),
    };
  }

  async matmul(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
  ): Promise<Float64Array> {
    await this._ensureInit();
    const elements = M * K + K * N;
    return this._cascade("matmul", elements, backend =>
      backend.matmul(A, B, M, K, N)
    );
  }

  async eltwise(
    op: "add" | "sub" | "dotmul" | "dotdiv",
    A: Float64Array, B: Float64Array | number,
    len: number
  ): Promise<Float64Array> {
    await this._ensureInit();
    return this._cascade("eltwise", len, backend =>
      backend.eltwise(op, A, B, len)
    );
  }

  async unary(
    op: "negate" | "abs" | "sqrt" | "exp" | "transpose",
    A: Float64Array,
    rows: number, cols: number
  ): Promise<Float64Array> {
    await this._ensureInit();
    return this._cascade("unary", rows * cols, backend =>
      backend.unary(op, A, rows, cols)
    );
  }

  configure(thresholds: Partial<DispatcherThresholds>): void {
    Object.assign(this.thresholds, thresholds);
  }

  getAvailableBackends(): string[] {
    return this.backends.map(b => b.name);
  }

  dispose(): void {
    for (const b of this.backends) b.dispose();
    this.backends = [];
    this.available = new Set();
    this.initialized = false;
    this.initPromise = null;
  }

  private async _ensureInit(): Promise<void> {
    if (this.initialized) return;
    await this.init();
  }

  private async _cascade<T>(
    opType: "matmul" | "eltwise" | "unary",
    elements: number,
    execute: (backend: IBackend) => Promise<T>
  ): Promise<T> {
    if (!this.initialized) {
      throw new Error("BackendDispatcher not initialized. Call initEngine() first.");
    }

    const eligible = this._eligibleBackends(opType, elements);

    let lastError: unknown;
    for (const backend of eligible) {
      try {
        const result = await execute(backend);
        return result;
      } catch (err) {
        console.warn(
          `[MatrixTS] Backend "${backend.name}" failed for ${opType}(${elements} elems), ` +
          `falling back. Error: ${(err as Error).message ?? err}`
        );
        lastError = err;
      }
    }

    throw new Error(
      `[MatrixTS] All backends failed for ${opType}. Last error: ${lastError}`
    );
  }

  private _eligibleBackends(
    opType: "matmul" | "eltwise" | "unary",
    elements: number
  ): IBackend[] {
    const { gpuMinElements, workersMinElements, wasmMinElements } = this.thresholds;

    return this.backends.filter(b => {
      switch (b.name) {
        case "webgpu":
          return opType === "matmul" && elements >= gpuMinElements;
        case "workers":
          return opType === "matmul" && elements >= workersMinElements;
        case "wasm":
          return elements >= wasmMinElements;
        case "vanilla":
          return true;
        default:
          return true;
      }
    });
  }
}

export const globalDispatcher = new BackendDispatcher();
