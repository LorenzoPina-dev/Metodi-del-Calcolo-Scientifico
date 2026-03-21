// ops/norm.ts
//
// Float64M fast-path: usa (.value) invece di .abs().toNumber()
// eliminando n² allocazioni Float64M per Frobenius, Inf, 1-norm.
//
import { Matrix } from "..";
import { INumeric } from "../type";

type NormType = "1" | "2" | "inf" | "fro";

export function norm<T extends INumeric<T>>(this: Matrix<T>, type: NormType = "2"): number {
    const isVec = this.rows === 1 || this.cols === 1;
    return isVec ? _normVec(this, type) : _normMat(this, type);
}

// ==================== VETTORI ====================

function _normVec<T extends INumeric<T>>(A: Matrix<T>, type: NormType): number {
    const d = A.data, len = d.length;
    const f64 = A.isFloat64;

    switch (type.toUpperCase()) {
        case "1": {
            let s = 0;
            if (f64) { for (let i = 0; i < len; i++) { const v = (d[i] as any).value; s += v < 0 ? -v : v; } }
            else      { for (let i = 0; i < len; i++) s += d[i].abs().toNumber(); }
            return s;
        }
        case "INF": {
            let max = 0;
            if (f64) { for (let i = 0; i < len; i++) { const v = Math.abs((d[i] as any).value); if (v > max) max = v; } }
            else      { for (let i = 0; i < len; i++) { const v = d[i].abs().toNumber(); if (v > max) max = v; } }
            return max;
        }
        case "2":
        case "FRO": {
            let ss = 0;
            if (f64) {
                let i = 0;
                for (; i <= len - 4; i += 4) {
                    const a = (d[i] as any).value, b = (d[i+1] as any).value;
                    const c = (d[i+2] as any).value, e = (d[i+3] as any).value;
                    ss += a*a + b*b + c*c + e*e;
                }
                for (; i < len; i++) { const v = (d[i] as any).value; ss += v*v; }
            } else {
                for (let i = 0; i < len; i++) { const v = d[i].abs().toNumber(); ss += v*v; }
            }
            return Math.sqrt(ss);
        }
        default:
            throw new Error(`norm: tipo '${type}' non supportato per vettori.`);
    }
}

// ==================== MATRICI ====================

function _normMat<T extends INumeric<T>>(A: Matrix<T>, type: NormType): number {
    switch (type.toUpperCase()) {
        case "1":   return _norm1(A);
        case "INF": return _normInf(A);
        case "FRO": return _normFro(A);
        case "2":   throw new Error("norm 2 matriciale richiede SVD (non implementato).");
        default:    throw new Error(`norm: tipo '${type}' non supportato per matrici.`);
    }
}

function _norm1<T extends INumeric<T>>(A: Matrix<T>): number {
    const R = A.rows, C = A.cols;
    const d = A.data;
    const f64 = A.isFloat64;
    let max = 0;
    for (let j = 0; j < C; j++) {
        let s = 0;
        if (f64) {
            for (let i = 0; i < R; i++) { const v = (d[i * C + j] as any).value; s += v < 0 ? -v : v; }
        } else {
            for (let i = 0; i < R; i++) s += A.get(i, j).abs().toNumber();
        }
        if (s > max) max = s;
    }
    return max;
}

function _normInf<T extends INumeric<T>>(A: Matrix<T>): number {
    const R = A.rows, C = A.cols;
    const d = A.data;
    const f64 = A.isFloat64;
    let max = 0;
    for (let i = 0; i < R; i++) {
        const off = i * C;
        let s = 0;
        if (f64) {
            let j = 0;
            for (; j <= C - 4; j += 4) {
                const a = (d[off+j] as any).value, b = (d[off+j+1] as any).value;
                const c = (d[off+j+2] as any).value, e = (d[off+j+3] as any).value;
                s += (a < 0 ? -a : a) + (b < 0 ? -b : b) + (c < 0 ? -c : c) + (e < 0 ? -e : e);
            }
            for (; j < C; j++) { const v = (d[off+j] as any).value; s += v < 0 ? -v : v; }
        } else {
            for (let j = 0; j < C; j++) s += A.get(i, j).abs().toNumber();
        }
        if (s > max) max = s;
    }
    return max;
}

function _normFro<T extends INumeric<T>>(A: Matrix<T>): number {
    const d = A.data, len = d.length;
    const f64 = A.isFloat64;
    let ss = 0;
    if (f64) {
        let i = 0;
        for (; i <= len - 4; i += 4) {
            const a = (d[i] as any).value, b = (d[i+1] as any).value;
            const c = (d[i+2] as any).value, e = (d[i+3] as any).value;
            ss += a*a + b*b + c*c + e*e;
        }
        for (; i < len; i++) { const v = (d[i] as any).value; ss += v*v; }
    } else {
        for (let i = 0; i < len; i++) { const v = d[i].abs().toNumber(); ss += v*v; }
    }
    return Math.sqrt(ss);
}
