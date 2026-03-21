// decomposition/tril.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Estrae la parte triangolare inferiore (con offset k, default 0). */
export function tril<T extends INumeric<T>>(A: Matrix<T>, k = 0): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j <= Math.min(i + k, A.cols - 1); j++) {
            if (j >= 0) out.set(i, j, A.get(i, j));
        }
    }
    return out;
}
