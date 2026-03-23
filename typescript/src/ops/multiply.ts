// ops/multiply.ts
//
// Gerarchia di esecuzione per Float64M (A × B, n grande):
//
//   1. GPU  (WebGPU, n ≥ 300, f32 ~7 cifre sig.)   → ~10-50× WASM
//   2. Workers (CPU threads, n ≥ 300)               → ~N× WASM  (N = # core / 2)
//   3. WASM + SIMD (n ≥ 16)                         → ~3-8× TS
//   4. TS Float64 fast-path (Float64Array, i-k-j)   → baseline
//   5. Path generico (Complex, Rational)             → correttezza > performance
//
// Le soglie sono calibrate per garantire guadagno netto dopo l'overhead
// di trasferimento dati (serializzazione JS↔GPU/Workers).
//
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";
import { getPoolSync, PARALLEL_THRESHOLD, _matmulSerial } from "../parallel/worker_pool";
import { isGPUAvailable, gpuMatmul, GPU_THRESHOLD } from "../gpu/webgpu_backend";

export function multiply<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {

    // ── Scalare ───────────────────────────────────────────────────────────────
    if (typeof B === "number") {
        if (this.isFloat64) {
            const d = this.data, len = d.length;
            // GPU per scalare: non conveniente (overhead > calcolo), usa WASM/TS
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

    // ── Float64M: usa il percorso ottimale ────────────────────────────────────
    if (this.isFloat64 && (B as any).isFloat64) {
        const M = this.rows, K = this.cols, N = B.cols;
        const nSq = M * K;   // proxy per la dimensione del problema

        // ── Percorso ASINCRONO (GPU/Workers): non chiamabile da operatori sincroni
        //    Per usare GPU/Workers su operazioni sincrone, wrappare in mulAsync().
        //    Qui offriamo il percorso sincrono WASM + TS.
        //    (vedi mulAsync() sotto per il percorso asincrono completo)

        // ── WASM + SIMD ────────────────────────────────────────────────────────
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

        // ── TS Float64 fast-path ─────────────────────────────────────────────
        return _matMulF64(this, B as any) as any;
    }

    // ── Path generico (Complex, Rational) ─────────────────────────────────────
    return _matMulGeneric(this, B);
}

// ─── Versione ASINCRONA con GPU e Worker threads ──────────────────────────────
// Da usare per grandi matrici (n > 300) dove il guadagno GPU/Workers è netto.
//
// Esempio:
//   const C = await mulAsync(A, B);
//
export async function mulAsync<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Promise<Matrix<T>> {
    if (A.cols !== B.rows)
        throw new Error(`mulAsync: dimensioni interne non coincidono (${A.cols} ≠ ${B.rows})`);

    if (!A.isFloat64 || !(B as any).isFloat64)
        return A.mul(B);  // percorso generico sincrono

    const M = A.rows, K = A.cols, N = B.cols;

    // Estrai dati raw come Float64Array
    const aFlat = _extractF64(A.data as any);
    const bFlat = _extractF64(B.data as any);

    // ── 1. GPU (WebGPU, f32, n ≥ 300) ────────────────────────────────────────
    if (M * K >= GPU_THRESHOLD.MATMUL && isGPUAvailable()) {
        try {
            const cFlat = await gpuMatmul(aFlat, bFlat, M, K, N);
            const outData = new Array<T>(M * N);
            for (let i = 0; i < M * N; i++) outData[i] = A.zero.fromNumber(cFlat[i]);
            return A.likeWithData(M, N, outData);
        } catch (e) {
            console.warn("[GPU] matmul fallback a Workers:", (e as Error).message);
        }
    }

    // ── 2. Worker threads (CPU parallelo, n ≥ 300) ───────────────────────────
    if (M * K >= PARALLEL_THRESHOLD.MATMUL) {
        const pool = getPoolSync();
        if (pool?.available) {
            const cFlat = new Float64Array(M * N);
            await pool.matmul(aFlat, bFlat, cFlat, M, K, N);
            const outData = new Array<T>(M * N);
            for (let i = 0; i < M * N; i++) outData[i] = A.zero.fromNumber(cFlat[i]);
            return A.likeWithData(M, N, outData);
        }
    }

    // ── 3. WASM sincrono ──────────────────────────────────────────────────────
    return A.mul(B);
}

// ─── Helper ───────────────────────────────────────────────────────────────────
function _extractF64(data: Array<{ value: number }>): Float64Array {
    const out = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = data[i].value;
    return out;
}

function _matMulF64<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const af = A.data, bf = B.data;
    const cf = new Float64Array(M * N);
    for (let i = 0; i < M; i++) {
        const iOff = i * K, outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = (af[iOff + k] as any).value as number;
            if (aik === 0) continue;
            const kOff = k * N;
            for (let j = 0; j < N; j++) cf[outOff + j] += aik * (bf[kOff + j] as any).value;
        }
    }
    const outData = new Array<T>(M * N);
    for (let i = 0; i < M * N; i++) outData[i] = A.zero.fromNumber(cf[i]);
    return A.likeWithData(M, N, outData);
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
