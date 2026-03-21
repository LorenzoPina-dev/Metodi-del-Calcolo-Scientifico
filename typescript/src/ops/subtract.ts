// ops/subtract.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function subtract<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return subScalar(this, B);
    if (this.rows === B.rows && this.cols === B.cols) return subMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return subRowVector(this, B);
    if (B.cols === 1 && B.rows === this.rows) return subColVector(this, B);
    throw new Error(`subtract: dimensioni incompatibili ${this.rows}×${this.cols} e ${B.rows}×${B.cols}`);
}

function subScalar<T extends INumeric<T>>(A: Matrix<T>, scalar: number): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    const s = A.zero.fromNumber(scalar);
    for (let i = 0; i < A.data.length; i++) out.data[i] = A.data[i].subtract(s);
    return out;
}

function subMatrix<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.data.length; i++) out.data[i] = A.data[i].subtract(B.data[i]);
    return out;
}

function subRowVector<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        const off = i * A.cols;
        for (let j = 0; j < A.cols; j++) {
            out.data[off + j] = A.data[off + j].subtract(B.data[j]);
        }
    }
    return out;
}

function subColVector<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        const off = i * A.cols;
        const bVal = B.data[i];
        for (let j = 0; j < A.cols; j++) {
            out.data[off + j] = A.data[off + j].subtract(bVal);
        }
    }
    return out;
}
