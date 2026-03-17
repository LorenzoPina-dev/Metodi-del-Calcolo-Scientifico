import { Matrix } from "../core";

export function zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
}

export function ones(rows: number, cols: number): Matrix {
    const m = new Matrix(rows, cols);
    m.data.fill(1);
    return m;
}

export function identity(n: number): Matrix {
    return diag(n, 1);
}

export function diag(n: number, k: number): Matrix {
    const m = zeros(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = k;
    return m;
}

export function diagFromArray(arr: number[]): Matrix {
    const n = arr.length;
    const m = zeros(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = arr[i];
    return m;
}

