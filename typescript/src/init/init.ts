// init/init.ts
import { Matrix } from "..";
import { INumeric, Float64M } from "../type";

// ============================================================
// Helper interno: crea una Matrix di default.
// Usato dalle factory statiche su Matrix quando non viene
// specificato un tipo (backward-compat con il codice float).
// ============================================================

const F0 = Float64M.zero;
const F1 = Float64M.one;

/** Crea una matrice di zeri Float64M di dimensioni rows×cols. */
export function zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols, F0, F1);
}

/** Crea una matrice di uni Float64M di dimensioni rows×cols. */
export function ones(rows: number, cols: number): Matrix {
    const m = new Matrix(rows, cols, F0, F1);
    m.data.fill(F1);
    return m;
}

/** Crea la matrice identità Float64M n×n. */
export function identity(n: number): Matrix {
    const m = zeros(n, n);
    for (let i = 0; i < n; i++) m.set(i, i, F1);
    return m;
}

/** Crea una matrice Float64M n×n con k sulla diagonale principale. */
export function diag(n: number, k: number): Matrix {
    const m = zeros(n, n);
    const kVal = new Float64M(k);
    for (let i = 0; i < n; i++) m.set(i, i, kVal);
    return m;
}

/** Crea una matrice Float64M diagonale dai valori di un array. */
export function diagFromArray(arr: number[]): Matrix {
    const n = arr.length;
    const m = zeros(n, n);
    for (let i = 0; i < n; i++) m.setNum(i, i, arr[i]);
    return m;
}

// ============================================================
// Versioni generiche (per tipi custom)
// ============================================================

/** Crea una matrice di zeri di tipo T dato uno zero e uno sample. */
export function zerosLike<T extends INumeric<T>>(
    rows: number,
    cols: number,
    zero: T,
    one: T
): Matrix<T> {
    return new Matrix<T>(rows, cols, zero, one);
}

export function identityLike<T extends INumeric<T>>(
    n: number,
    zero: T,
    one: T
): Matrix<T> {
    const m = zerosLike<T>(n, n, zero, one);
    for (let i = 0; i < n; i++) m.set(i, i, one);
    return m;
}
