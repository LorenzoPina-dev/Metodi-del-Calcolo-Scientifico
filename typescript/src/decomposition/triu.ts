// decomposition/triu.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Estrae la parte triangolare superiore (con offset k, default 0). */
export function triu<T extends INumeric<T>>(A: Matrix<T>, k = 0): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        for (let j = Math.max(i + k, 0); j < A.cols; j++) {
            out.set(i, j, A.get(i, j));
        }
    }
    return out;
}
