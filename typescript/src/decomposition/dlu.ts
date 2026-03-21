// decomposition/dlu.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function decomposeDLU<T extends INumeric<T>>(
    A: Matrix<T>
): { D: Matrix<T>; L: Matrix<T>; U: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("decomposeDLU: matrice non quadrata.");
    const n = A.rows;
    const D = A.like(n, n);
    const L = A.like(n, n);
    const U = A.like(n, n);
    const ad = A.data, dd = D.data, ld = L.data, ud = U.data;

    for (let i = 0; i < n; i++) {
        const off = i * n;
        for (let j = 0; j < n; j++) {
            const v = ad[off + j];
            if (i === j)    dd[off + j] = v;
            else if (i > j) ld[off + j] = v;
            else            ud[off + j] = v;
        }
    }
    return { D, L, U };
}
