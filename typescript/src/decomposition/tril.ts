// decomposition/tril.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Triangolare inferiore con offset k (default 0). Accesso diretto a data[]. */
export function tril<T extends INumeric<T>>(A: Matrix<T>, k = 0): Matrix<T> {
    const R = A.rows, C = A.cols;
    const out = A.like(R, C);
    const ad = A.data, od = out.data;
    for (let i = 0; i < R; i++) {
        const off = i * C;
        const limit = Math.min(i + k + 1, C);
        for (let j = 0; j < limit; j++) od[off + j] = ad[off + j];
    }
    return out;
}
