// ops/unary.ts
//
// Tutte le operazioni unarie usano fromNumber() per produrre l'output,
// evitando allocazioni intermedie (es. .abs() alloca un Float64M; qui no).
//
import { Matrix } from "..";
import { INumeric } from "../type";

// ---- Helper centrale ----
function _applyUnary<T extends INumeric<T>>(
    A: Matrix<T>,
    opF64: (v: number) => number,
    opGen: (v: T) => T
): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
        for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(opF64((d[i] as any).value));
    } else {
        for (let i = 0; i < len; i++) outData[i] = opGen(d[i]);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

export function abs<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.abs, (x) => x.abs());
}
export function sqrt<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.sqrt, (x) => x.sqrt());
}
export function round<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.round, (x) => x.round());
}
export function negate<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, (v) => -v, (x) => x.negate());
}
export function exp<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.exp, (x) => x.fromNumber(Math.exp(x.toNumber())));
}
export function floor<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.floor, (x) => x.fromNumber(Math.floor(x.toNumber())));
}
export function ceil<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.ceil, (x) => x.fromNumber(Math.ceil(x.toNumber())));
}
export function sin<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.sin, (x) => x.fromNumber(Math.sin(x.toNumber())));
}
export function cos<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.cos, (x) => x.fromNumber(Math.cos(x.toNumber())));
}
export function tan<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return _applyUnary(this, Math.tan, (x) => x.fromNumber(Math.tan(x.toNumber())));
}

/** Traccia: somma degli elementi diagonali. */
export function trace<T extends INumeric<T>>(this: Matrix<T>): T {
    const n = Math.min(this.rows, this.cols);
    const d = this.data, C = this.cols;
    let t = this.zero;
    for (let i = 0; i < n; i++) t = t.add(d[i * C + i]);
    return t;
}
