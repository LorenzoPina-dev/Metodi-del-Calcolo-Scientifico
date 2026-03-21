// decomposition/triu.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Triangolare superiore con offset k (default 0). Accesso diretto a data[]. */
export function triu<T extends INumeric<T>>(A: Matrix<T>, k = 0): Matrix<T> {
    const R = A.rows, C = A.cols;
    const out = A.like(R, C);
    const ad = A.data, od = out.data;
    for (let i = 0; i < R; i++) {
        const off = i * C;
        const start = Math.max(i + k, 0);
        for (let j = start; j < C; j++) od[off + j] = ad[off + j];
    }
    return out;
}
