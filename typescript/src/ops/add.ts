// ops/add.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Somma universale (A + B).
 * Gestisce: scalare, matrici identiche, vettore riga (1×N), vettore colonna (M×1).
 */
export function add<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return addScalar(this, B);
    if (this.rows === B.rows && this.cols === B.cols) return addMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return addRowVector(this, B);
    if (B.cols === 1 && B.rows === this.rows) return addColVector(this, B);
    throw new Error(`add: dimensioni incompatibili ${this.rows}×${this.cols} e ${B.rows}×${B.cols}`);
}

export function totalSum<T extends INumeric<T>>(this: Matrix<T>): T {
    let s = this.zero;
    for (let i = 0; i < this.data.length; i++) s = s.add(this.data[i]);
    return s;
}

// ---- helper privati ----

function addScalar<T extends INumeric<T>>(A: Matrix<T>, scalar: number): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    const s = A.zero.fromNumber(scalar);
    for (let i = 0; i < A.data.length; i++) out.data[i] = A.data[i].add(s);
    return out;
}

function addMatrix<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.data.length; i++) out.data[i] = A.data[i].add(B.data[i]);
    return out;
}

function addRowVector<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        const off = i * A.cols;
        for (let j = 0; j < A.cols; j++) {
            out.data[off + j] = A.data[off + j].add(B.data[j]);
        }
    }
    return out;
}

function addColVector<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        const off = i * A.cols;
        const bVal = B.data[i];
        for (let j = 0; j < A.cols; j++) {
            out.data[off + j] = A.data[off + j].add(bVal);
        }
    }
    return out;
}
