// ops/multiply.ts
//
// Fast-path Float64M:
//   Lettura: (data[i] as any).value  → nessuna allocazione
//   Calcolo: Float64Array cf          → zero allocazioni intermedie
//   Output:  Float64M per elemento   → M*N allocazioni (vs M*K*N nel path generico)
//   Per 100×100: 10K allocazioni vs 1M (100× meno).
//
import { Matrix } from "..";
import { INumeric } from "../type";

export function multiply<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    // ---- Scalare ----
    if (typeof B === "number") {
        if (this.isFloat64) return _scalarF64(this, B);
        const s = this.zero.fromNumber(B);
        const len = this.data.length;
        const outData = new Array<T>(len);
        for (let i = 0; i < len; i++) outData[i] = this.data[i].multiply(s);
        return this.likeWithData(this.rows, this.cols, outData);
    }

    if (this.cols !== B.rows) {
        throw new Error(`multiply: dimensioni interne non coincidono (${this.cols} ≠ ${B.rows})`);
    }

    // ---- Matrice × Matrice — Float64M fast path ----
    if (this.isFloat64 && (B as any).isFloat64) return _matMulF64(this, B as any) as any;

    // ---- Matrice × Matrice — path generico ----
    return _matMulGeneric(this, B);
}

// ---------------------------------------------------------------------------
// Float64M — scalare
// ---------------------------------------------------------------------------
function _scalarF64<T extends INumeric<T>>(A: Matrix<T>, s: number): Matrix<T> {
    const d = A.data;
    const len = d.length;
    const outData = new Array<T>(len);
    for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber((d[i] as any).value * s);
    return A.likeWithData(A.rows, A.cols, outData);
}

// ---------------------------------------------------------------------------
// Float64M — moltiplicazione matriciale con Float64Array intermedia
// ---------------------------------------------------------------------------
function _matMulF64<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const af = A.data, bf = B.data;
    const cf = new Float64Array(M * N);  // zero-inizializzato

    // Layout i-k-j: massimizza la riutilizzazione delle cache L1/L2
    for (let i = 0; i < M; i++) {
        const iOff = i * K, outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = (af[iOff + k] as any).value as number;
            if (aik === 0) continue;                 // skip esplicito degli zeri
            const kOff = k * N;
            for (let j = 0; j < N; j++) cf[outOff + j] += aik * (bf[kOff + j] as any).value;
        }
    }

    // Wrap: una sola passata O(M*N) — vs O(M*K*N) nel path generico
    const outData = new Array<T>(M * N);
    for (let i = 0; i < M * N; i++) outData[i] = A.zero.fromNumber(cf[i]);
    return A.likeWithData(M, N, outData);
}

// ---------------------------------------------------------------------------
// Path generico (Complex, Rational, …)
// ---------------------------------------------------------------------------
function _matMulGeneric<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const out = A.like(M, N);
    const ad = A.data, bd = B.data, od = out.data;

    for (let i = 0; i < M; i++) {
        const iOff = i * K, outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = ad[iOff + k];
            const kOff = k * N;
            for (let j = 0; j < N; j++) {
                od[outOff + j] = od[outOff + j].add(aik.multiply(bd[kOff + j]));
            }
        }
    }
    return out;
}
