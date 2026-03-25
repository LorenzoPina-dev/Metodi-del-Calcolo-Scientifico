// solver/jacobi.ts
//
// Gerarchia di esecuzione per solveJacobiMat / solveJacobiAsync:
//
//   solveJacobiMat() — sincrono:
//     1. WASM + SIMD (n² ≥ WASM_THRESHOLD.ITERATIVE)  → ~5× TS
//     2. TS Float64 fast-path (unroll ×4)              → baseline
//     3. Path generico (Complex, Rational)
//
//   solveJacobiAsync() — async, massimo throughput:
//     1. GPU  (WebGPU f32, n² ≥ GPU_THRESHOLD.JACOBI)
//     2. WasmWorkerPool con soglia interna (500×500):
//          n² < 250 000  → WASM single-thread (jacobiSolve SIMD, zero GC)
//          n² >= 250 000 → TS workers paralleli (SAB zero-copy)
//     3. WASM → TS sincrono                            → fallback
//
import type { Matrix }          from "../Matrix.js";
import type { Float64M, INumeric } from "../type/index.js";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge.js";
import { WasmWorkerPool } from "../parallel/wasm_worker_pool.js";
import { isGPUAvailable, gpuJacobi, GPU_THRESHOLD } from "../gpu/webgpu_backend.js";
import { _hasConverged } from "./_hasConverged.js";

// ─── solveJacobiMat — sincrono ────────────────────────────────────────────────
export function solveJacobiMat<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 5000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiMat: matrice non quadrata.");

    if (A.isFloat64) {
        // 1. WASM (sincrono)
        if (n * n >= WASM_THRESHOLD.ITERATIVE) {
            const w = getBridgeSync();
            if (w) {
                const aPtr       = w.writeFloat64M(A.data as any);
                const bPtr       = w.writeFloat64M(b.data as any);
                const xPtr       = w.allocOutput(n);
                const xNewPtr    = w.allocOutput(n);
                const diagInvPtr = w.allocOutput(n);
                w.exports.jacobiSolve(aPtr, bPtr, xPtr, xNewPtr, diagInvPtr, n, tol, maxIter);
                const flat = w.readF64(xPtr, n);
                w.reset();
                const out = A.like(n, 1);
                for (let i = 0; i < n; i++) out.data[i] = A.zero.fromNumber(flat[i]);
                return out;
            }
        }
        // 2. TS Float64 fast-path
        return _jacobiF64(A as any, b as any, tol, maxIter);
    }
    return _jacobiGeneric(A, b, tol, maxIter);
}

// ─── solveJacobiAsync — GPU → WasmWorkerPool → fallback ─────────────────────
export async function solveJacobiAsync<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 5000
): Promise<Matrix<T>> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiAsync: matrice non quadrata.");
    if (!A.isFloat64) return solveJacobiMat(A, b, tol, maxIter);

    const ad = A.data as any as Array<{ value: number }>;
    const bd = b.data as any as Array<{ value: number }>;
    const aFlat   = new Float64Array(n * n);
    const bFlat   = new Float64Array(n);
    const diagInv = new Float64Array(n);

    for (let i = 0; i < n * n; i++) aFlat[i] = ad[i].value;
    for (let i = 0; i < n;     i++) bFlat[i] = bd[i].value;
    for (let i = 0; i < n;     i++) {
        const d = aFlat[i * n + i];
        if (Math.abs(d) < 1e-300) throw new Error(`Jacobi: pivot nullo alla riga ${i + 1}.`);
        diagInv[i] = 1.0 / d;
    }

    // 1. GPU (WebGPU, f32)
    if (n * n >= GPU_THRESHOLD.JACOBI && isGPUAvailable()) {
        try {
            const xFlat = await gpuJacobi(aFlat, bFlat, diagInv, n, tol, maxIter);
            return _fromF64Vec(A, n, xFlat);
        } catch (e) {
            console.warn("[GPU] Jacobi fallback a WasmWorkerPool:", (e as Error).message);
        }
    }

    // 2. WasmWorkerPool (soglia interna 500×500):
    //    • n² < 250 000  → WASM single-thread (jacobiSolve SIMD+diagInv precomp.)
    //    • n² >= 250 000 → TS workers paralleli (SAB zero-copy, efficiente per iter)
    const pool = WasmWorkerPool.instance;
    if (pool) {
        try {
            const xFlat = await pool.jacobi(aFlat, bFlat, diagInv, n, tol, maxIter);
            return _fromF64Vec(A, n, xFlat);
        } catch (e) {
            console.warn("[WasmWorkerPool] Jacobi fallback WASM ST:", (e as Error).message);
        }
    }

    // 3. WASM → TS sincrono
    return solveJacobiMat(A, b, tol, maxIter);
}

// ─── Helper ──────────────────────────────────────────────────────────────────
function _fromF64Vec<T extends INumeric<T>>(
    A: Matrix<T>, n: number, flat: Float64Array
): Matrix<T> {
    const out = A.like(n, 1);
    for (let i = 0; i < n; i++) out.data[i] = A.zero.fromNumber(flat[i]);
    return out;
}

// ─── TS Float64 fast-path (unroll ×4) ────────────────────────────────────────
function _jacobiF64(
    A: Matrix<Float64M>, b: Matrix<Float64M>,
    tol: number, maxIter: number
): Matrix<any> {
    const n = A.rows, ad = A.data, bd = b.data;
    const a_raw   = new Float64Array(n * n);
    const b_raw   = new Float64Array(n);
    const diagInv = new Float64Array(n);

    for (let i = 0; i < n * n; i++) a_raw[i] = ad[i].value;
    for (let i = 0; i < n;     i++) b_raw[i] = bd[i].value;
    for (let i = 0; i < n;     i++) {
        const d = a_raw[i * n + i];
        if (Math.abs(d) < 1e-300) throw new Error(`Jacobi: pivot nullo alla riga ${i + 1}.`);
        diagInv[i] = 1.0 / d;
    }

    let x = new Float64Array(n), xNew = new Float64Array(n);
    for (let iter = 0; iter < maxIter; iter++) {
        let maxDiff = 0, maxAbsX = 0;
        for (let i = 0; i < n; i++) {
            const off = i * n; let s = 0;
            let j = 0;
            const i4 = i & ~3;
            for (; j < i4; j += 4) {
                s += a_raw[off + j]     * x[j]
                   + a_raw[off + j + 1] * x[j + 1]
                   + a_raw[off + j + 2] * x[j + 2]
                   + a_raw[off + j + 3] * x[j + 3];
            }
            for (; j < i; j++) s += a_raw[off + j] * x[j];
            j = i + 1;
            const n4 = n & ~3;
            for (; j < n4; j += 4) {
                s += a_raw[off + j]     * x[j]
                   + a_raw[off + j + 1] * x[j + 1]
                   + a_raw[off + j + 2] * x[j + 2]
                   + a_raw[off + j + 3] * x[j + 3];
            }
            for (; j < n; j++) s += a_raw[off + j] * x[j];

            const xi = (b_raw[i] - s) * diagInv[i];
            xNew[i] = xi;
            const diff = xi - x[i]; const absDiff = diff < 0 ? -diff : diff;
            if (absDiff > maxDiff) maxDiff = absDiff;
            const ax = xi < 0 ? -xi : xi; if (ax > maxAbsX) maxAbsX = ax;
        }
        [x, xNew] = [xNew, x];
        if (maxDiff / (maxAbsX > 1 ? maxAbsX : 1) < tol) break;
    }

    return _fromF64Vec(A as any, n, x);
}

// ─── Path generico (Complex, Rational) ────────────────────────────────────────
function _jacobiGeneric<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>, tol: number, maxIter: number
): Matrix<T> {
    const n = A.rows, ad = A.data, bd = b.data;
    let x = A.like(n, 1);
    for (let iter = 0; iter < maxIter; iter++) {
        const xNext = A.like(n, 1); const nd = xNext.data, xd = x.data;
        for (let i = 0; i < n; i++) {
            const off = i * n; let s = bd[i];
            for (let j = 0; j < i; j++) s = s.subtract(ad[off + j].multiply(xd[j]));
            for (let j = i + 1; j < n; j++) s = s.subtract(ad[off + j].multiply(xd[j]));
            nd[i] = s.divide(ad[off + i]);
        }
        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }
    return x;
}
