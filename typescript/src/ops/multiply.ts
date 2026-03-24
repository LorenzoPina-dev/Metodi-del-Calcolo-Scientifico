// ops/multiply.ts
//
// Gerarchia di esecuzione per Float64M (A × B):
//
//   mul()  — sincrono:
//     1. WASM + SIMD (n² ≥ WASM_THRESHOLD.MATMUL)   → ~3-8× TS
//     2. TS Float64 fast-path (tiled, unroll ×4)     → baseline
//     3. Path generico (Complex, Rational)            → correttezza
//
//   mulAsync() — async, massimo throughput:
//     1. GPU  (WebGPU f32, n² ≥ GPU_THRESHOLD.MATMUL)  → ~10-50× WASM
//     2. Workers (pool, n² ≥ PARALLEL_THRESHOLD.MATMUL) → ~N× WASM
//     3. WASM → TS                                       → fallback
//
// NOTA: il pool viene riutilizzato tra chiamate (non viene spento dopo mulAsync).
// Chiamare WorkerPool.instance.shutdown() solo all'uscita del processo.
//
import type { Matrix }    from "../Matrix.js";
import type { INumeric }  from "../type/index.js";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge.js";
import {  WorkerPool } from "../parallel/worker_pool.js";
import { isGPUAvailable, gpuMatmul, GPU_THRESHOLD } from "../gpu/webgpu_backend.js";

// ─── mul() sincrono ───────────────────────────────────────────────────────────
export function multiply<T extends INumeric<T>>(
    this: Matrix<T>,
    B: Matrix<T> | number
): Matrix<T> {

    // ── Scalare ───────────────────────────────────────────────────────────────
    if (typeof B === "number") {
        if (this.isFloat64) {
            const d = this.data, len = d.length;
            if (len >= WASM_THRESHOLD.MATMUL) {
                const w = getBridgeSync();
                if (w) {
                    const aPtr = w.writeFloat64M(d as any);
                    const cPtr = w.allocOutput(len);
                    w.exports.mulScalar(aPtr, cPtr, len, B);
                    const flat = w.readF64(cPtr, len);
                    w.reset();
                    const out = new Array<T>(len);
                    for (let i = 0; i < len; i++) out[i] = this.zero.fromNumber(flat[i]);
                    return this.likeWithData(this.rows, this.cols, out);
                }
            }
            const outData = new Array<T>(len);
            for (let i = 0; i < len; i++) outData[i] = this.zero.fromNumber((d[i] as any).value * B);
            return this.likeWithData(this.rows, this.cols, outData);
        }
        const s = this.zero.fromNumber(B), d = this.data, len = d.length;
        const outData = new Array<T>(len);
        for (let i = 0; i < len; i++) outData[i] = d[i].multiply(s);
        return this.likeWithData(this.rows, this.cols, outData);
    }

    // ── Dimensioni ────────────────────────────────────────────────────────────
    if (this.cols !== B.rows)
        throw new Error(`multiply: dimensioni interne non coincidono (${this.cols} ≠ ${B.rows})`);

    // ── Float64: WASM → TS ────────────────────────────────────────────────────
    if (this.isFloat64 && (B as any).isFloat64) {
        const M = this.rows, K = this.cols, N = B.cols;
        const nSq = M * K;

        // 1. WASM + SIMD
        if (nSq >= WASM_THRESHOLD.MATMUL) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(this.data as any);
                const bPtr = w.writeFloat64M(B.data as any);
                const cPtr = w.allocOutput(M * N);
                w.exports.matmul(aPtr, bPtr, cPtr, M, K, N);
                const flat = w.readF64(cPtr, M * N);
                w.reset();
                const outData = new Array<T>(M * N);
                for (let i = 0; i < M * N; i++) outData[i] = this.zero.fromNumber(flat[i]);
                return this.likeWithData(M, N, outData);
            }
        }

        // 2. TS Float64 fast-path (tiled + unroll ×4)
        return _matMulF64(this, B as any) as any;
    }

    // ── Path generico (Complex, Rational) ─────────────────────────────────────
    return _matMulGeneric(this, B);
}

// ─── mulAsync() — GPU → Workers → WASM → TS ──────────────────────────────────
export async function mulAsync<T extends INumeric<T>>(
    A: Matrix<T>,
    B: Matrix<T>
): Promise<Matrix<T>> {
    if (A.cols !== B.rows)
        throw new Error(`mulAsync: dimensioni interne non coincidono (${A.cols} ≠ ${B.rows})`);

    if (!A.isFloat64 || !(B as any).isFloat64)
        return A.mul(B);

    const M = A.rows, K = A.cols, N = B.cols;
    const aFlat = _extractF64(A.data as any);
    const bFlat = _extractF64(B.data as any);

    // 1. GPU (WebGPU, f32)
    if (M * K >= GPU_THRESHOLD.MATMUL && isGPUAvailable()) {
        try {
            const cFlat = await gpuMatmul(aFlat, bFlat, M, K, N);
            return _fromF64Flat(A, M, N, cFlat);
        } catch (e) {
            console.warn("[GPU] matmul fallback a Workers:", (e as Error).message);
        }
    }

    // 2. Worker pool asincrono (pool persistente — NON spento dopo ogni op)
    if (M * K >= 500_000) {
        const pool = new WorkerPool();
        await pool.init();
        const cFlat = new Float64Array(M * N);
        await pool.matmul(aFlat, bFlat, M, K, N);
        return _fromF64Flat(A, M, N, cFlat);
    }

    // 3. Sincrono (WASM → TS)
    return A.mul(B);
}

// ─── Helpers pubblici ─────────────────────────────────────────────────────────
export function _extractF64(data: Array<{ value: number }>): Float64Array {
    const out = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = data[i].value;
    return out;
}

function _fromF64Flat<T extends INumeric<T>>(
    A: Matrix<T>, rows: number, cols: number, flat: Float64Array
): Matrix<T> {
    const len = rows * cols;
    const out = new Array<T>(len);
    for (let i = 0; i < len; i++) out[i] = A.zero.fromNumber(flat[i]);
    return A.likeWithData(rows, cols, out);
}

// ─── TS Float64 tiled (kernel locale, unroll ×4) ─────────────────────────────
function _matMulF64<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const af = A.data, bf = B.data;
    const cf = new Float64Array(M * N);

    // Trasponi B per accesso riga-riga (L1-friendly)
    const BT = new Float64Array(K * N);
    for (let k = 0; k < K; k++) {
        const kN = k * N;
        for (let j = 0; j < N; j++) BT[j * K + k] = (bf[kN + j] as any).value;
    }

    for (let i = 0; i < M; i++) {
        const iK = i * K, iN = i * N;
        for (let j = 0; j < N; j++) {
            const jK = j * K;
            let s = 0.0;
            let k = 0;
            const K4 = K & ~3;
            for (; k < K4; k += 4) {
                s += (af[iK + k] as any).value     * BT[jK + k]
                   + (af[iK + k + 1] as any).value * BT[jK + k + 1]
                   + (af[iK + k + 2] as any).value * BT[jK + k + 2]
                   + (af[iK + k + 3] as any).value * BT[jK + k + 3];
            }
            for (; k < K; k++) s += (af[iK + k] as any).value * BT[jK + k];
            cf[iN + j] = s;
        }
    }

    return _fromF64Flat(A, M, N, cf);
}

function _matMulGeneric<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const out = A.like(M, N);
    const ad = A.data, bd = B.data, od = out.data;
    for (let i = 0; i < M; i++) {
        const iOff = i * K, outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = ad[iOff + k], kOff = k * N;
            for (let j = 0; j < N; j++)
                od[outOff + j] = od[outOff + j].add(aik.multiply(bd[kOff + j]));
        }
    }
    return out;
}
