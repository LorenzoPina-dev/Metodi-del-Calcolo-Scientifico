// ops/dotOps.ts
import { Matrix } from "..";
import { INumeric } from "../type";

// --- DOT MULTIPLY (.*) ---
export function dotMultiply<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _applyScalarF64(this, B, (a, b) => a * b, (a, b) => a.multiply(b));
    return _applyBroadcast(this, B, (a, b) => a * b, (a, b) => a.multiply(b));
}

// --- DOT DIVIDE (./) ---
export function dotDivide<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _applyScalarF64(this, B, (a, b) => a / b, (a, b) => a.divide(b));
    return _applyBroadcast(this, B, (a, b) => a / b, (a, b) => a.divide(b));
}

// --- DOT POWER (.^) ---
export function dotPow<T extends INumeric<T>>(this: Matrix<T>, exp: number | Matrix<T>): Matrix<T> {
    if (typeof exp === "number") {
        if (this.isFloat64) {
            const d = this.data, len = d.length;
            const outData = new Array<T>(len);
            for (let i = 0; i < len; i++)
                outData[i] = this.zero.fromNumber(Math.pow((d[i] as any).value, exp));
            return this.likeWithData(this.rows, this.cols, outData);
        }
        return _applyScalarGeneric(this, (a) => _powGeneric(a, exp));
    }
    return _applyBroadcast(this, exp, (a, b) => Math.pow(a, b), (a, b) => _powGeneric(a, b.toNumber()));
}

// ---- helper interni ----

function _applyScalarF64<T extends INumeric<T>>(
    A: Matrix<T>, s: number,
    opF64: (a: number, b: number) => number,
    opGen: (a: T, b: T) => T
): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
        for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(opF64((d[i] as any).value, s));
    } else {
        const sv = A.zero.fromNumber(s);
        for (let i = 0; i < len; i++) outData[i] = opGen(d[i], sv);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

function _applyScalarGeneric<T extends INumeric<T>>(
    A: Matrix<T>, op: (a: T) => T
): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    for (let i = 0; i < len; i++) outData[i] = op(d[i]);
    return A.likeWithData(A.rows, A.cols, outData);
}

function _applyBroadcast<T extends INumeric<T>>(
    A: Matrix<T>, B: Matrix<T>,
    opF64: (a: number, b: number) => number,
    opGen: (a: T, b: T) => T
): Matrix<T> {
    const ad = A.data, bd = B.data;
    const R = A.rows, C = A.cols;
    const f64 = A.isFloat64 && B.isFloat64;
    const outData = new Array<T>(R * C);

    if (A.rows === B.rows && A.cols === B.cols) {
        const len = ad.length;
        if (f64) {
            for (let i = 0; i < len; i++)
                outData[i] = A.zero.fromNumber(opF64((ad[i] as any).value, (bd[i] as any).value));
        } else {
            for (let i = 0; i < len; i++) outData[i] = opGen(ad[i], bd[i]);
        }
    } else if (B.rows === 1 && B.cols === C) {         // row broadcast
        for (let i = 0; i < R; i++) {
            const off = i * C;
            if (f64) {
                for (let j = 0; j < C; j++)
                    outData[off + j] = A.zero.fromNumber(opF64((ad[off + j] as any).value, (bd[j] as any).value));
            } else {
                for (let j = 0; j < C; j++) outData[off + j] = opGen(ad[off + j], bd[j]);
            }
        }
    } else if (B.cols === 1 && B.rows === R) {         // column broadcast
        for (let i = 0; i < R; i++) {
            const off = i * C;
            if (f64) {
                const bv = (bd[i] as any).value as number;
                for (let j = 0; j < C; j++)
                    outData[off + j] = A.zero.fromNumber(opF64((ad[off + j] as any).value, bv));
            } else {
                const bv = bd[i];
                for (let j = 0; j < C; j++) outData[off + j] = opGen(ad[off + j], bv);
            }
        }
    } else {
        throw new Error("dotOp: dimensioni incompatibili");
    }
    return A.likeWithData(R, C, outData);
}

/** Esponenziazione intera per tipi generici (binary exponentiation). */
function _powGeneric<T extends INumeric<T>>(base: T, exp: number): T {
    if (!Number.isInteger(exp)) return base.fromNumber(Math.pow(base.toNumber(), exp));
    if (exp === 0) return base.fromNumber(1);
    if (exp < 0)  return base.fromNumber(1).divide(_powGeneric(base, -exp));
    let result = base.fromNumber(1);
    let b = base;
    let e = exp;
    while (e > 0) {
        if (e & 1) result = result.multiply(b);
        b = b.multiply(b);
        e >>= 1;
    }
    return result;
}
