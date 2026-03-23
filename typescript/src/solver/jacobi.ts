// solver/jacobi.ts
//
// Gerarchia di esecuzione:
//   1. GPU  (WebGPU, n ≥ 200, f32)         — ~10-50× WASM per grandi n
//   2. Workers (CPU threads, n ≥ 100)       — ~N× WASM  (N = # core fisici / 2)
//   3. WASM + SIMD (n ≥ 8)                  — ~5× TS con diagInv precompilato
//   4. TS Float64 fast-path                 — baseline
//
// Jacobi è EMBARRASSINGLY PARALLEL per riga:
//   x_new[i] dipende solo da x_old[j], non da x_new[j] (a differenza di GS/SOR).
//   → Ogni riga può essere calcolata in parallelo senza dipendenze.
//   → Perfetto per GPU (n thread paralleli = n righe) e Worker threads.
//
import { Matrix }        from "..";
import { Float64M, INumeric } from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";
import { getPoolSync, PARALLEL_THRESHOLD, _jacobiSerial } from "../parallel/worker_pool";
import { isGPUAvailable, gpuJacobi, GPU_THRESHOLD } from "../gpu/webgpu_backend";
import { _hasConverged } from "./_hasConverged";

export function solveJacobiMat<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 5000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiMat: matrice non quadrata.");

    if (A.isFloat64) {
        // ── WASM (sincrono, n ≥ 8) ───────────────────────────────────────────
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
        // ── TS Float64 fast-path ─────────────────────────────────────────────
        return _jacobiF64(A as any, b as any, tol, maxIter);
    }
    return _jacobiGeneric(A, b, tol, maxIter);
}

// ─── Versione ASINCRONA con GPU e Worker threads ──────────────────────────────
// Da usare per n ≥ 100 dove il parallelismo porta beneficio netto.
//
// Esempio:
//   const x = await solveJacobiAsync(A, b, 1e-10, 5000);
//
export async function solveJacobiAsync<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 5000
): Promise<Matrix<T>> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiAsync: matrice non quadrata.");
    if (!A.isFloat64) return solveJacobiMat(A, b, tol, maxIter);

    // Prepara dati raw
    const ad = A.data as any as Array<{ value: number }>;
    const bd = b.data as any as Array<{ value: number }>;
    const aFlat    = new Float64Array(n * n);
    const bFlat    = new Float64Array(n);
    const diagInv  = new Float64Array(n);

    for (let i = 0; i < n * n; i++) aFlat[i] = ad[i].value;
    for (let i = 0; i < n;     i++) bFlat[i] = bd[i].value;
    for (let i = 0; i < n;     i++) {
        const d = aFlat[i * n + i];
        if (Math.abs(d) < 1e-300) throw new Error(`Jacobi: pivot nullo alla riga ${i + 1}.`);
        diagInv[i] = 1.0 / d;
    }

    let xFlat: Float64Array;

    // ── 1. GPU (WebGPU, f32, n ≥ 200) ────────────────────────────────────────
    if (n * n >= GPU_THRESHOLD.JACOBI && isGPUAvailable()) {
        try {
            xFlat = await gpuJacobi(aFlat, bFlat, diagInv, n, tol, maxIter);
            const out = A.like(n, 1);
            for (let i = 0; i < n; i++) out.data[i] = A.zero.fromNumber(xFlat[i]);
            return out;
        } catch (e) {
            console.warn("[GPU] Jacobi fallback a Workers:", (e as Error).message);
        }
    }

    // ── 2. Worker threads (CPU parallelo, n ≥ 100) ───────────────────────────
    if (n * n >= PARALLEL_THRESHOLD.JACOBI) {
        const pool = getPoolSync();
        if (pool?.available) {
            xFlat = await pool.jacobi(aFlat, bFlat, diagInv, n, tol, maxIter);
            const out = A.like(n, 1);
            for (let i = 0; i < n; i++) out.data[i] = A.zero.fromNumber(xFlat[i]);
            return out;
        }
    }

    // ── 3. WASM sincrono ──────────────────────────────────────────────────────
    return solveJacobiMat(A, b, tol, maxIter);
}

// ─── TS Float64 fast-path (diagInv precompilato) ────────────────────────────
function _jacobiF64(
    A: Matrix<Float64M>, b: Matrix<Float64M>,
    tol: number, maxIter: number
): Matrix<any> {
    const n = A.rows, ad = A.data, bd = b.data;
    const a_raw  = new Float64Array(n * n);
    const b_raw  = new Float64Array(n);
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
            for (let j = 0; j < i; j++) s += a_raw[off + j] * x[j];
            for (let j = i + 1; j < n; j++) s += a_raw[off + j] * x[j];
            const xi = (b_raw[i] - s) * diagInv[i];
            xNew[i] = xi;
            const diff = xi - x[i]; const absDiff = diff < 0 ? -diff : diff;
            if (absDiff > maxDiff) maxDiff = absDiff;
            const ax = xi < 0 ? -xi : xi; if (ax > maxAbsX) maxAbsX = ax;
        }
        const tmp = x; x = xNew; xNew = tmp;
        const denom = maxAbsX > 1 ? maxAbsX : 1;
        if (maxDiff / denom < tol) break;
    }
    const out = A.like(n, 1);
    for (let i = 0; i < n; i++) out.data[i] = A.zero.fromNumber(x[i]);
    return out;
}

// ─── Path generico (Complex, Rational) ─────────────────────────────────────
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
