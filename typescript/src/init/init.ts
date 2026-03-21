// init/init.ts
import { Matrix } from "..";
import { INumeric, Float64M } from "../type";

const F0 = Float64M.zero;
const F1 = Float64M.one;

export function zeros(rows: number, cols: number): Matrix<Float64M> {
    return new Matrix<Float64M>(rows, cols);
}

export function ones(rows: number, cols: number): Matrix<Float64M> {
    const m = new Matrix<Float64M>(rows, cols);
    m.data.fill(F1);
    return m;
}

export function identity(n: number): Matrix<Float64M> {
    const m = new Matrix<Float64M>(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = F1;
    return m;
}

export function diag(n: number, k: number): Matrix<Float64M> {
    const m = new Matrix<Float64M>(n, n);
    const kv = new Float64M(k);
    for (let i = 0; i < n; i++) m.data[i * n + i] = kv;
    return m;
}

export function diagFromArray(arr: number[]): Matrix<Float64M> {
    const n = arr.length;
    const m = new Matrix<Float64M>(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = new Float64M(arr[i]);
    return m;
}

// ---- Versioni generiche ----

export function zerosLike<T extends INumeric<T>>(
    rows: number, cols: number, zero: T, one: T
): Matrix<T> {
    return new Matrix<T>(rows, cols, zero, one);
}

export function identityLike<T extends INumeric<T>>(n: number, zero: T, one: T): Matrix<T> {
    const m = new Matrix<T>(n, n, zero, one);
    for (let i = 0; i < n; i++) m.data[i * n + i] = one;
    return m;
}
