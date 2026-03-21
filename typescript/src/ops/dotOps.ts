// ops/dotOps.ts
import { Matrix } from "..";
import { INumeric } from "../type";

// --- DOT MULTIPLY (.* ) ---
export function dotMultiply<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") {
        const s = this.zero.fromNumber(B);
        return applyScalar(this, (a) => a.multiply(s));
    }
    return applyBroadcast(this, B, (a, b) => a.multiply(b));
}

// --- DOT DIVIDE (./) ---
export function dotDivide<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") {
        const s = this.zero.fromNumber(B);
        return applyScalar(this, (a) => a.divide(s));
    }
    return applyBroadcast(this, B, (a, b) => a.divide(b));
}

// --- DOT POWER (.^) ---
export function dotPow<T extends INumeric<T>>(this: Matrix<T>, exp: number | Matrix<T>): Matrix<T> {
    if (typeof exp === "number") {
        return applyScalar(this, (a) => powGeneric(a, exp));
    }
    return applyBroadcast(this, exp, (a, b) => powGeneric(a, b.toNumber()));
}

// ---- helper ----

function applyScalar<T extends INumeric<T>>(
    A: Matrix<T>,
    op: (a: T) => T
): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.data.length; i++) out.data[i] = op(A.data[i]);
    return out;
}

function applyBroadcast<T extends INumeric<T>>(
    A: Matrix<T>,
    B: Matrix<T>,
    op: (a: T, b: T) => T
): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    if (A.rows === B.rows && A.cols === B.cols) {
        for (let i = 0; i < A.data.length; i++) out.data[i] = op(A.data[i], B.data[i]);
    } else if (B.rows === 1 && B.cols === A.cols) {
        for (let i = 0; i < A.rows; i++) {
            const off = i * A.cols;
            for (let j = 0; j < A.cols; j++) out.data[off + j] = op(A.data[off + j], B.data[j]);
        }
    } else if (B.cols === 1 && B.rows === A.rows) {
        for (let i = 0; i < A.rows; i++) {
            const off = i * A.cols;
            for (let j = 0; j < A.cols; j++) out.data[off + j] = op(A.data[off + j], B.data[i]);
        }
    } else {
        throw new Error("dotOp: dimensioni incompatibili");
    }
    return out;
}

/** Potenza intera generica tramite metodi INumeric. */
function powGeneric<T extends INumeric<T>>(base: T, exp: number): T {
    if (!Number.isInteger(exp)) {
        // Fallback float per esponenti non interi
        return base.fromNumber(Math.pow(base.toNumber(), exp));
    }
    if (exp === 0) return base.fromNumber(1);
    if (exp < 0)  return base.fromNumber(1).divide(powGeneric(base, -exp));
    let result = base.fromNumber(1);
    let b = base;
    let e = exp;
    while (e > 0) {
        if (e % 2 === 1) result = result.multiply(b);
        b = b.multiply(b);
        e = Math.floor(e / 2);
    }
    return result;
}
