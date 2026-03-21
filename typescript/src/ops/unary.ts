// ops/unary.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Applica una funzione unaria T→T a ogni elemento. */
function applyUnary<T extends INumeric<T>>(A: Matrix<T>, fn: (x: T) => T): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.data.length; i++) out.data[i] = fn(A.data[i]);
    return out;
}

export function abs<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.abs());
}

export function sqrt<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.sqrt());
}

export function round<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.round());
}

export function negate<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.negate());
}

export function exp<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.fromNumber(Math.exp(x.toNumber())));
}

export function floor<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.fromNumber(Math.floor(x.toNumber())));
}

export function ceil<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.fromNumber(Math.ceil(x.toNumber())));
}

export function sin<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.fromNumber(Math.sin(x.toNumber())));
}

export function cos<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.fromNumber(Math.cos(x.toNumber())));
}

export function tan<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    return applyUnary(this, (x) => x.fromNumber(Math.tan(x.toNumber())));
}

/** Traccia: somma degli elementi diagonali. Restituisce T. */
export function trace<T extends INumeric<T>>(this: Matrix<T>): T {
    const n = Math.min(this.rows, this.cols);
    let t = this.zero;
    for (let i = 0; i < n; i++) t = t.add(this.get(i, i));
    return t;
}
