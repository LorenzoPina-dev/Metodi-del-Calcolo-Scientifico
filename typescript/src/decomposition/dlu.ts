// decomposition/dlu.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Scompone A = D + L_strict + U_strict (forma additiva).
 * D:  diagonale
 * L:  triangolare inferiore stretta (off-diagonale inferiore)
 * U:  triangolare superiore stretta (off-diagonale superiore)
 */
export function decomposeDLU<T extends INumeric<T>>(
    A: Matrix<T>
): { D: Matrix<T>; L: Matrix<T>; U: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("decomposeDLU: la matrice deve essere quadrata.");

    const D = A.like(A.rows, A.cols);
    const L = A.like(A.rows, A.cols);
    const U = A.like(A.rows, A.cols);

    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j < A.cols; j++) {
            const v = A.get(i, j);
            if (i === j)      D.set(i, j, v);
            else if (i > j)   L.set(i, j, v);
            else              U.set(i, j, v);
        }
    }
    return { D, L, U };
}
